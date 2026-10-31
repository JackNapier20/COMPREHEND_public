"""Probe a local OpenAI-compatible server with one short chat request."""

import argparse
import os
import socket
import time

from openai import APIError, OpenAI


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://gpu014:5050/v1")
    parser.add_argument("--model", help="Defaults to the sole model returned by /v1/models")
    parser.add_argument("--timeout", type=float, default=15, help="Timeout per request in seconds")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    base_url = args.base_url.strip().rstrip("/")
    print(f"Client node: {socket.gethostname()}", flush=True)
    print(f"Endpoint: {base_url}", flush=True)
    stage = "GET /v1/models"
    started = time.monotonic()

    try:
        with OpenAI(
            api_key=os.environ.get("GEMMA_API_KEY", "DUMMY"),
            base_url=base_url,
            timeout=args.timeout,
            max_retries=0,
        ) as client:
            print(f"Sending {stage}...", flush=True)
            model_ids = sorted({model.id for model in client.models.list().data})
            print(f"Available model IDs: {model_ids}", flush=True)
            if args.model:
                model = args.model
                if model not in model_ids:
                    raise ValueError(f"Requested model {model!r} is not served.")
            elif len(model_ids) == 1:
                model = model_ids[0]
            else:
                raise ValueError("Expected one served model; specify --model using an ID above.")

            stage = "POST /v1/chat/completions"
            print(f"Sending {stage} with model {model!r}...", flush=True)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Follow instructions exactly."},
                    {"role": "user", "content": "Reply with exactly: connection OK"},
                ],
                max_tokens=64,
                temperature=0,
            )
            if not response.choices:
                raise ValueError("Server returned no completion choices.")
            choice = response.choices[0]
            reply = (choice.message.content or "").strip()
            print(f"Finish reason: {choice.finish_reason}", flush=True)
            if not reply:
                raise ValueError("Server returned an empty chat response.")
            print(f"Response: {reply}", flush=True)
            print(f"SUCCESS: model answered in {time.monotonic() - started:.2f}s.", flush=True)
            return 0
    except (APIError, ValueError) as exc:
        print(f"FAILED at {stage}: {type(exc).__name__}: {exc}", flush=True)
        cause = exc.__cause__
        while cause is not None:
            print(f"Caused by: {type(cause).__name__}: {cause}", flush=True)
            cause = cause.__cause__
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
