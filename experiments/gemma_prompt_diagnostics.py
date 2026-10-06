"""Compare seven bounded Gemma prompt conditions without changing the pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from openai import APIError, OpenAI, Timeout

from retrieval.blossom_matching import gemma_main_idea_messages


V4_MAIN_IDEA_SYSTEM = (
    "You are a professional editor. Your task is to extract the single most important "
    "main idea from a given text in one short sentence."
)
V4_SUMMARY_SYSTEM = (
    "You are a professional editor. Your task is to generate a high-quality summary "
    "that generalizes two prompts."
)


def diagnostic_cases() -> dict[str, str]:
    task = "Write Python code to read CSV files."
    current = gemma_main_idea_messages(task)
    current_prefix, current_user = (message["content"] for message in current)
    training_user = (
        f"Text:\n{task}\n\n"
        "Extract the single most important main idea in one short sentence.\n"
        "Return only that sentence."
    )
    capability = (
        "Give one short capability phrase common to these tasks: "
        "(1) Write Python code to read a CSV file. "
        "(2) Write Python code to filter CSV rows. Output only the capability phrase."
    )
    pair = (
        f"Prompt A:\n{task}\n\nPrompt B:\nWrite Python code to filter CSV rows.\n\n"
        "Write a concise generalized prompt that is implied by both Prompt A and Prompt B.\n"
    )
    # The training-format conditions mirror the available finetuneDatasetV4.jsonl.
    return {
        "known_working_capability": capability,
        "current_extraction_no_prefix": current_user,
        "current_extraction_current_prefix": f"{current_prefix}\n\n{current_user}",
        "current_extraction_v4_prefix": f"{V4_MAIN_IDEA_SYSTEM}\n\n{current_user}",
        "v4_extraction_current_prefix": f"{current_prefix}\n\n{training_user}",
        "v4_extraction_v4_prefix": f"{V4_MAIN_IDEA_SYSTEM}\n\n{training_user}",
        "v4_pair_summary": f"{V4_SUMMARY_SYSTEM}\n\n{pair}",
    }


def run_diagnostics(config: dict, *, max_tokens: int = 128, timeout: float = 60,
                    max_api_calls: int = 7) -> dict:
    cases = diagnostic_cases()
    if max_tokens < 1 or timeout <= 0:
        raise ValueError("Output-token limit and timeout must be positive")
    if max_api_calls < len(cases):
        raise ValueError(f"This diagnostic needs {len(cases)} calls; ceiling is {max_api_calls}")
    report = {
        "model": config["model"], "base_url": config["base_url"],
        "temperature": 0, "max_tokens": max_tokens, "timeout_seconds": timeout,
        "stop": ["<end_of_turn>"], "ignore_eos": False,
        "planned_calls": len(cases), "attempts_per_condition": 1,
        "training_template_source": "available finetuneDatasetV4.jsonl (confirm checkpoint lineage)",
        "results": [],
    }
    print(f"Endpoint: {config['base_url']}; model: {config['model']}", flush=True)
    print(f"Exactly {len(cases)} generation calls, no retries; max {max_tokens} output tokens each.", flush=True)
    with OpenAI(
        api_key=config.get("api_key", "DUMMY"), base_url=config["base_url"],
        timeout=Timeout(timeout, connect=min(10.0, timeout)), max_retries=0,
    ) as client:
        for name, prompt in cases.items():
            messages = [{"role": "user", "content": prompt}]
            result = {
                "condition": name, "messages": messages,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
            started = time.monotonic()
            print(f"\nCondition: {name}", flush=True)
            try:
                response = client.chat.completions.create(
                    model=config["model"], messages=messages, max_tokens=max_tokens,
                    temperature=0, stop=["<end_of_turn>"], extra_body={"ignore_eos": False},
                )
                if not response.choices:
                    raise ValueError("Server returned no completion choices")
                choice = response.choices[0]
                result.update(
                    response=choice.message.content or "", finish_reason=choice.finish_reason,
                    stop_reason=getattr(choice, "stop_reason", None),
                    usage=response.usage.model_dump() if response.usage is not None else None,
                )
                print(f"Finish reason: {result['finish_reason']}; stop reason: {result['stop_reason']}", flush=True)
                print(f"Response: {result['response'][:800]!r}", flush=True)
            except (APIError, ValueError) as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                print(f"Request failed: {result['error']}", flush=True)
            result["elapsed_seconds"] = time.monotonic() - started
            report["results"].append(result)
    print("\nDiagnostic complete. These responses are evidence, not approval for a full run.", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="Overrides GEMMA_BASE_URL/.env")
    parser.add_argument("--model", help="Overrides GEMMA_MODEL/.env")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-api-calls", type=int, default=7)
    parser.add_argument("--output", type=Path, help="Optional NEW JSON file; existing files are refused")
    args = parser.parse_args()
    if args.max_tokens < 1 or args.timeout <= 0 or args.max_api_calls < 7:
        parser.error("Use positive token/timeout limits and --max-api-calls of at least 7")
    if args.output and args.output.exists():
        parser.error(f"Output already exists; use a new filename: {args.output}")
    if args.base_url:
        os.environ["GEMMA_BASE_URL"] = args.base_url
    if args.model:
        os.environ["GEMMA_MODEL"] = args.model
    from main import API_KEYS

    report = run_diagnostics(
        API_KEYS["gemma"], max_tokens=args.max_tokens, timeout=args.timeout,
        max_api_calls=args.max_api_calls,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        print(f"Full responses saved to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
