import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from openai import APIConnectionError, APIStatusError

from experiments.local_model_preflight import wait_for_model


class LocalModelPreflightTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "api_key": "DUMMY", "base_url": "http://gpu014:5050/v1", "model": "gemma-v4",
        }
        self.client = Mock()
        self.ready = SimpleNamespace(data=[SimpleNamespace(id="gemma-v4")])
        self.clock = 0.0
        for name, replacement in (
            ("OpenAI", Mock(return_value=self.client)),
            ("time.monotonic", Mock(side_effect=lambda: self.clock)),
            ("time.sleep", Mock(side_effect=self.advance)),
        ):
            patcher = patch(f"experiments.local_model_preflight.{name}", replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
            if name == "time.sleep":
                self.sleep = replacement

    def advance(self, seconds):
        self.clock += seconds

    def refused(self):
        error = APIConnectionError(request=Mock())
        error.__cause__ = ConnectionRefusedError(111, "Connection refused")
        return error

    def status_error(self, status):
        return APIStatusError(
            "server error",
            response=SimpleNamespace(status_code=status, headers={}, request=Mock()),
            body={},
        )

    def test_waits_through_connection_refusal_until_model_is_ready(self):
        self.client.models.list.side_effect = [self.refused(), self.ready]
        wait_for_model(self.config, wait_seconds=30, retry_interval=10)
        self.assertEqual(self.client.models.list.call_count, 2)
        self.sleep.assert_called_once_with(10)
        self.client.close.assert_called_once()

    def test_stopped_server_has_bounded_wait_and_specific_diagnostics(self):
        self.client.models.list.side_effect = self.refused()
        with self.assertRaisesRegex(RuntimeError, "gpu014:5050/v1/models after 3 seconds"):
            wait_for_model(self.config, wait_seconds=3, retry_interval=2)
        self.assertEqual(self.clock, 3)
        self.assertEqual(self.client.models.list.call_args_list, [
            call(timeout=3), call(timeout=1),
        ])
        self.client.close.assert_called_once()

    def test_wrong_registered_model_stops_immediately(self):
        self.client.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id="different-checkpoint")],
        )
        with self.assertRaisesRegex(ValueError, "different-checkpoint"):
            wait_for_model(self.config)
        self.sleep.assert_not_called()

    def test_retries_service_unavailable_during_startup(self):
        self.client.models.list.side_effect = [self.status_error(503), self.ready]
        wait_for_model(self.config)
        self.assertEqual(self.client.models.list.call_count, 2)

    def test_authentication_error_does_not_spend_startup_wait(self):
        self.client.models.list.side_effect = self.status_error(401)
        with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
            wait_for_model(self.config)
        self.client.models.list.assert_called_once()
        self.sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
