import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import APITimeoutError

from experiments import gemma_prompt_diagnostics as diagnostic
from retrieval.blossom_matching import gemma_main_idea_messages


class GemmaPromptDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.config = {"base_url": "http://gpu013:5050/v1", "model": "gemma-v4"}

    def response(self, text="short answer", finish_reason="stop"):
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(
            message=SimpleNamespace(content=text), finish_reason=finish_reason,
            stop_reason=106 if finish_reason == "stop" else None,
        )])

    def test_cases_include_current_request_and_crossed_training_format(self):
        cases = diagnostic.diagnostic_cases()
        current = gemma_main_idea_messages("Write Python code to read CSV files.")
        self.assertEqual(len(cases), 7)
        self.assertEqual(len(set(cases.values())), 7)
        self.assertEqual(cases["current_extraction_current_prefix"],
                         "\n\n".join(message["content"] for message in current))
        self.assertTrue(cases["v4_extraction_v4_prefix"].startswith(diagnostic.V4_MAIN_IDEA_SYSTEM))
        training_body = cases["v4_extraction_v4_prefix"].split("\n\n", 1)[1]
        self.assertTrue(training_body.startswith("Text:\n"))
        self.assertTrue(training_body.endswith("Return only that sentence."))

    @patch("experiments.gemma_prompt_diagnostics.OpenAI")
    def test_truncated_response_is_recorded_without_retries(self, openai):
        client = openai.return_value.__enter__.return_value
        client.chat.completions.create.return_value = self.response("repeated text", "length")
        report = diagnostic.run_diagnostics(self.config)
        self.assertEqual(openai.call_args.kwargs["max_retries"], 0)
        self.assertEqual(client.chat.completions.create.call_count, 7)
        self.assertEqual(len(report["results"]), 7)
        self.assertNotIn("api_key", report)
        for result in report["results"]:
            self.assertEqual(result["finish_reason"], "length")
            self.assertEqual(result["response"], "repeated text")
            self.assertEqual(len(result["prompt_sha256"]), 64)
        for request in client.chat.completions.create.call_args_list:
            self.assertEqual(request.kwargs["max_tokens"], 128)
            self.assertEqual(request.kwargs["temperature"], 0)
            self.assertEqual(request.kwargs["extra_body"], {"ignore_eos": False})
            self.assertEqual([message["role"] for message in request.kwargs["messages"]], ["user"])

    @patch("experiments.gemma_prompt_diagnostics.OpenAI")
    def test_failed_condition_does_not_hide_other_conditions(self, openai):
        client = openai.return_value.__enter__.return_value
        client.chat.completions.create.side_effect = [
            APITimeoutError(request=httpx.Request("POST", "http://gpu013:5050/v1")),
            *[self.response() for _ in range(6)],
        ]
        report = diagnostic.run_diagnostics(self.config)
        self.assertIn("APITimeoutError", report["results"][0]["error"])
        self.assertTrue(all(result["response"] == "short answer" for result in report["results"][1:]))
        self.assertEqual(client.chat.completions.create.call_count, 7)

    @patch("experiments.gemma_prompt_diagnostics.OpenAI")
    def test_call_ceiling_fails_before_client_creation(self, openai):
        with self.assertRaisesRegex(ValueError, "needs 7 calls"):
            diagnostic.run_diagnostics(self.config, max_api_calls=6)
        openai.assert_not_called()

    @patch("experiments.gemma_prompt_diagnostics.run_diagnostics")
    def test_cli_refuses_existing_output_before_requests(self, run):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.json"
            path.touch()
            with patch("sys.argv", ["probe", "--output", str(path)]):
                with self.assertRaises(SystemExit) as failure:
                    diagnostic.main()
            self.assertEqual(failure.exception.code, 2)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
