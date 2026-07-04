"""Tests for the shared OpenAI-compatible loop's transient-failure retry.

The streaming/retry client lives in backends_agent._openai_agent and is
used by both the llama and zai backends. A completion is safe to retry
because tool side effects only run after _stream_completion returns, so
dropped connections / read timeouts / HTTP 429 / 5xx are retried with
backoff. These tests stub the single-attempt request so no real server is
contacted.
"""

import asyncio
import unittest

from Cozter.backends_agent import _openai_agent as oa


class OpenAIBackoffTests(unittest.TestCase):
    def test_parse_retry_after(self) -> None:
        self.assertEqual(oa._parse_retry_after("5"), 5.0)
        self.assertIsNone(oa._parse_retry_after(None))
        self.assertIsNone(oa._parse_retry_after("soon"))  # HTTP-date form

    def test_backoff_honors_retry_after_and_caps(self) -> None:
        self.assertEqual(oa._backoff_delay(1, 3.0), 3.0)
        self.assertEqual(oa._backoff_delay(9, 100.0), 10.0)  # capped
        self.assertGreaterEqual(oa._backoff_delay(3), oa._backoff_delay(1))
        self.assertLessEqual(oa._backoff_delay(20), 10.0 * 1.25)


class OpenAIRetryLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_once = oa._stream_once
        self._orig_delay = oa._backoff_delay

        def _no_delay(*args, **kwargs) -> float:
            return 0.0

        oa._backoff_delay = _no_delay

    def tearDown(self) -> None:
        oa._stream_once = self._orig_once
        oa._backoff_delay = self._orig_delay

    def _run(self, once) -> tuple:
        oa._stream_once = once
        # max_retries=2 -> initial attempt + 2 retries.
        return asyncio.run(oa._stream_completion(
            "http://x/chat/completions", {}, {}, 300, 2, "test",
        ))

    def test_retries_then_succeeds(self) -> None:
        calls = {"n": 0}

        async def once(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise oa._RetryableError("boom")
            return ("ok", [])

        self.assertEqual(self._run(once), ("ok", []))
        self.assertEqual(calls["n"], 3)

    def test_gives_up_after_max_retries(self) -> None:
        calls = {"n": 0}

        async def once(*args, **kwargs):
            calls["n"] += 1
            raise oa._RetryableError("always fails")

        with self.assertRaises(RuntimeError):
            self._run(once)
        self.assertEqual(calls["n"], 3)  # initial + 2 retries


if __name__ == "__main__":
    unittest.main()
