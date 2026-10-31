import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from llm_utils import (
    APICallTimeoutError,
    anthropic_text,
    call_with_deadline,
    openai_chat_text,
    together_chat_text,
)


class DeadlineTests(unittest.TestCase):
    def test_interrupts_blocking_call(self):
        with self.assertRaises(APICallTimeoutError):
            call_with_deadline(time.sleep, 1, timeout_seconds=0.01)

    def test_worker_thread_defers_to_provider_timeout(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                call_with_deadline, lambda: "complete", timeout_seconds=0.01
            )
            self.assertEqual(future.result(), "complete")


class _Messages:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
        self.requests = []

    def create(self, **request):
        self.calls += 1
        self.requests.append(request)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class AnthropicTextTests(unittest.TestCase):
    @patch("llm_utils.time.sleep")
    def test_retries_empty_response_and_joins_text_blocks(self, _sleep):
        messages = _Messages([
            SimpleNamespace(content=[], stop_reason="refusal"),
            SimpleNamespace(
                content=[SimpleNamespace(text=" first "), SimpleNamespace(text="second")],
                stop_reason="end_turn",
            ),
        ])
        result = anthropic_text(SimpleNamespace(messages=messages), model="test")
        self.assertEqual(result, "first\nsecond")
        self.assertEqual(messages.calls, 2)

    @patch("llm_utils.time.sleep")
    def test_raises_after_bounded_attempts(self, _sleep):
        messages = _Messages([RuntimeError("network"), RuntimeError("network")])
        with self.assertRaisesRegex(RuntimeError, "after 2 attempts"):
            anthropic_text(SimpleNamespace(messages=messages), attempts=2, model="test")


class OpenAITextTests(unittest.TestCase):
    @patch("llm_utils.time.sleep")
    def test_retries_empty_chat_response(self, _sleep):
        empty = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=""))])
        good = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=" fallback "))])
        completions = _Messages([empty, good])
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        self.assertEqual(openai_chat_text(client, model="test"), "fallback")
        self.assertEqual(completions.calls, 2)

    @patch("llm_utils.OpenAI")
    def test_together_uses_configured_endpoint_model_and_reasoning(self, openai):
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=" summary "))])
        completions = _Messages([response])
        openai.return_value = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

        result = together_chat_text(
            {
                "api_key": "key",
                "base_url": "https://example.test/v1",
                "model": "org/model",
                "max_tokens": 2048,
                "reasoning_effort": "high",
            },
            messages=[{"role": "user", "content": "prompt"}],
        )

        self.assertEqual(result, "summary")
        openai.assert_called_once_with(
            api_key="key", base_url="https://example.test/v1", timeout=120.0
        )
        self.assertEqual(completions.calls, 1)

    @patch("llm_utils.time.sleep")
    @patch("llm_utils.OpenAI")
    def test_together_expands_budget_after_reasoning_only_responses(
        self, openai, _sleep
    ):
        empty = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="", reasoning_content="thinking"),
            finish_reason="length",
        )])
        good = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=" answer "),
            finish_reason="stop",
        )])
        completions = _Messages([empty, empty, empty, good])
        openai.return_value = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

        result = together_chat_text(
            {
                "api_key": "key",
                "model": "org/model",
                "max_tokens": 2048,
                "empty_retry_max_tokens": 4096,
            },
            messages=[{"role": "user", "content": "prompt"}],
        )

        self.assertEqual(result, "answer")
        self.assertEqual(
            [request["max_tokens"] for request in completions.requests],
            [2048, 2048, 2048, 4096],
        )
