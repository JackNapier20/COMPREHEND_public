"""Small provider-response helpers shared across hierarchy pipelines."""

from __future__ import annotations

import os
import signal
import threading
import time
from contextlib import contextmanager

from openai import OpenAI


API_TIMEOUT_SECONDS = float(os.environ.get("LLM_API_TIMEOUT_SECONDS", "120"))


class APICallTimeoutError(TimeoutError):
    """Raised when a synchronous provider call exceeds its hard deadline."""


@contextmanager
def _hard_deadline(seconds: float):
    """Interrupt blocking socket reads that SDK timeouts fail to stop."""
    if (
        seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def handle_timeout(_signum, _frame):
        raise APICallTimeoutError(
            f"Provider call exceeded {seconds:g}-second hard deadline"
        )

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def call_with_deadline(call, /, *args, timeout_seconds: float | None = None,
                       **kwargs):
    timeout = API_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    with _hard_deadline(timeout):
        return call(*args, **kwargs)


def anthropic_text(client, *, attempts: int = 3, **request) -> str:
    """Return joined Anthropic text blocks, retrying empty/transient replies."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            message = call_with_deadline(client.messages.create, **request)
            text = "\n".join(
                block.text.strip()
                for block in (message.content or [])
                if getattr(block, "text", "").strip()
            )
            if text:
                return text
            stop_reason = getattr(message, "stop_reason", None)
            last_error = RuntimeError(
                f"Anthropic returned no text blocks (stop_reason={stop_reason!r})"
            )
        except Exception as exc:
            last_error = exc
        if attempt < attempts:
            print(f"Anthropic call attempt {attempt} failed: {last_error}; retrying")
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Anthropic call failed after {attempts} attempts") from last_error


def openai_chat_text(client, *, attempts: int = 3, **request) -> str:
    """Return OpenAI-compatible chat text with bounded empty/error retries."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            response = call_with_deadline(client.chat.completions.create, **request)
            choice = response.choices[0]
            text = (choice.message.content or "").strip()
            if text:
                return text
            reasoning = getattr(choice.message, "reasoning_content", "") or ""
            last_error = RuntimeError(
                "OpenAI-compatible API returned empty text "
                f"(finish_reason={getattr(choice, 'finish_reason', None)!r}, "
                f"reasoning_chars={len(reasoning)})"
            )
        except Exception as exc:
            last_error = exc
        if attempt < attempts:
            print(f"OpenAI call attempt {attempt} failed: {last_error}; retrying")
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"OpenAI call failed after {attempts} attempts") from last_error


def together_chat_text(config: dict, *, messages: list[dict], attempts: int = 3,
                       max_tokens: int | None = None) -> str:
    """Call a Together-hosted model through its OpenAI-compatible endpoint."""
    api_key = (config or {}).get("api_key", "")
    model = (config or {}).get("model", "")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY is not configured")
    if not model:
        raise ValueError("TOGETHER_MODEL is not configured")

    client = OpenAI(
        api_key=api_key,
        base_url=config.get("base_url", "https://api.together.xyz/v1"),
        timeout=API_TIMEOUT_SECONDS,
    )
    token_limit = max_tokens or int(config.get("max_tokens", 4096))
    retry_ceiling = max(
        token_limit,
        int(config.get("empty_retry_max_tokens", 16384)),
    )
    request = {
        "model": model,
        "messages": messages,
        "max_tokens": token_limit,
        "temperature": 0,
    }
    reasoning_effort = config.get("reasoning_effort")
    if reasoning_effort:
        request["extra_body"] = {"reasoning_effort": reasoning_effort}

    while True:
        try:
            return openai_chat_text(client, attempts=attempts, **request)
        except RuntimeError as exc:
            cause = exc
            empty_response = False
            while cause is not None:
                if "returned empty text" in str(cause):
                    empty_response = True
                    break
                cause = cause.__cause__
            current_limit = int(request["max_tokens"])
            if not empty_response or current_limit >= retry_ceiling:
                raise
            next_limit = min(current_limit * 2, retry_ceiling)
            print(
                "Together produced reasoning but no visible text; retrying "
                f"the same model with max_tokens={next_limit}"
            )
            request["max_tokens"] = next_limit
