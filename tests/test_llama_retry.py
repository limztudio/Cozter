"""Tests for the llama backend's transient-failure retry/backoff.

A completion is safe to retry because tool side effects only run after
_stream_completion returns, so dropped connections / read timeouts / HTTP
429 / 5xx are retried with backoff. These tests stub the single-attempt
request so no real server is contacted.
"""

import asyncio
import unittest

from Cozter import config
from Cozter.backends_agent import llama


class LlamaBackoffTests(unittest.TestCase):
    def test_parse_retry_after(self) -> None:
        self.assertEqual(llama._parse_retry_after("5"), 5.0)
        self.assertIsNone(llama._parse_retry_after(None))
        self.assertIsNone(llama._parse_retry_after("soon"))  # HTTP-date form

    def test_backoff_honors_retry_after_and_caps(self) -> None:
        self.assertEqual(llama._backoff_delay(1, 3.0), 3.0)
        self.assertEqual(llama._backoff_delay(9, 100.0), 10.0)  # capped
        # Without Retry-After: grows with attempt, never exceeds cap+jitter.
        self.assertGreaterEqual(llama._backoff_delay(3), llama._backoff_delay(1))
        self.assertLessEqual(llama._backoff_delay(20), 10.0 * 1.25)


class LlamaRetryLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_once = llama._stream_once
        self._orig_delay = llama._backoff_delay
        self._orig_retries = config.get_llama_max_retries

        def _no_delay(*args, **kwargs) -> float:
            return 0.0

        def _two_retries() -> int:
            return 2

        llama._backoff_delay = _no_delay
        config.get_llama_max_retries = _two_retries

    def tearDown(self) -> None:
        llama._stream_once = self._orig_once
        llama._backoff_delay = self._orig_delay
        config.get_llama_max_retries = self._orig_retries

    def test_retries_then_succeeds(self) -> None:
        calls = {"n": 0}

        async def fake_once(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise llama._RetryableError("boom")
            return ("ok", [])

        llama._stream_once = fake_once
        result = asyncio.run(llama._stream_completion("http://x", {}))
        self.assertEqual(result, ("ok", []))
        self.assertEqual(calls["n"], 3)  # 2 failures, then success

    def test_gives_up_after_max_retries(self) -> None:
        calls = {"n": 0}

        async def fake_once(*args, **kwargs):
            calls["n"] += 1
            raise llama._RetryableError("always fails")

        llama._stream_once = fake_once
        with self.assertRaises(RuntimeError):
            asyncio.run(llama._stream_completion("http://x", {}))
        self.assertEqual(calls["n"], 3)  # initial attempt + 2 retries


if __name__ == "__main__":
    unittest.main()
