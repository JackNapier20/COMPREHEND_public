"""Wait for a local model server and validate the controlled 256-leaf inputs."""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path

from openai import APIConnectionError, APIStatusError, OpenAI


def wait_for_model(config: dict, *, wait_seconds: float = 600,
                   retry_interval: float = 10, probe_timeout: float = 5) -> None:
    base_url = str(config.get("base_url", "")).strip().rstrip("/")
    model = str(config.get("model", "")).strip()
    if not base_url or not model or "<" in base_url or "<" in model:
        raise ValueError("Set GEMMA_BASE_URL and GEMMA_MODEL in .env first.")
    if min(wait_seconds, retry_interval, probe_timeout) <= 0:
        raise ValueError("Readiness wait, retry interval, and probe timeout must be positive.")

    print(f"Client node: {socket.gethostname()}", flush=True)
    print(f"Checking model endpoint: {base_url}/models", flush=True)
    print(f"Expected served model: {model}", flush=True)
    print(f"Waiting up to {wait_seconds:g} seconds for server startup.", flush=True)
    client = OpenAI(
        api_key=config.get("api_key", "DUMMY"), base_url=base_url,
        timeout=probe_timeout, max_retries=0,
    )
    deadline = time.monotonic() + wait_seconds
    last_error = None
    last_progress_minute = -1
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                models = client.models.list(timeout=min(probe_timeout, remaining))
            except APIConnectionError as exc:
                last_error = exc
            except APIStatusError as exc:
                if exc.status_code not in {502, 503, 504}:
                    raise RuntimeError(
                        f"Model endpoint {base_url}/models returned HTTP {exc.status_code}; "
                        "check its API key and URL."
                    ) from exc
                last_error = exc
            else:
                model_ids = {item.id for item in models.data}
                if model not in model_ids:
                    raise ValueError(
                        f"GEMMA_MODEL {model!r} is not served by {base_url}; "
                        f"available model IDs: {sorted(model_ids)}"
                    )
                print(f"Model server ready: {model}", flush=True)
                return

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            progress_minute = int((wait_seconds - remaining) // 60)
            if progress_minute != last_progress_minute:
                print(
                    f"Server not ready: {last_error}; "
                    f"{remaining:.0f} seconds left to wait.", flush=True,
                )
                last_progress_minute = progress_minute
            time.sleep(min(retry_interval, remaining))
    finally:
        client.close()

    cause = getattr(last_error, "__cause__", None) or last_error
    raise RuntimeError(
        f"Cannot reach a ready model server at {base_url}/models after "
        f"{wait_seconds:g} seconds from client node {socket.gethostname()}. "
        f"Last error: {cause}. Check the serving job is still running, its "
        "actual node and port, and connectivity from this client node. "
        "Already-exported GEMMA_BASE_URL/GEMMA_MODEL override .env values."
    ) from last_error


def main() -> None:
    from main import API_KEYS
    from tree import Tree

    key = API_KEYS["openai"]
    if not key or key.endswith("..."):
        raise SystemExit("Set OPENAI_API_KEY in .env for GPT evaluation.")
    for dataset in ("WB", "UF"):
        source = Path(
            f"comparison_256_outputs/{dataset}_current_k2/evaltree_capability_base.json"
        )
        tree = Tree.load(source)
        if len(tree.levels) != 1 or len(tree.levels[0].nodes) != 256:
            raise SystemExit(f"Expected exactly 256 source leaves: {source}")
        leaves = tree.levels[0].nodes
        if any(not node.capability or node.mainIdea for node in leaves):
            raise SystemExit(f"Expected fixed capabilities and fresh main ideas: {source}")
    wait_for_model(
        API_KEYS["gemma"],
        wait_seconds=float(os.environ.get("GEMMA_WAIT_SECONDS", "600")),
    )
    print("Inputs verified: WB256 and UF256; Gemma generation, OpenAI evaluation.")


if __name__ == "__main__":
    main()
