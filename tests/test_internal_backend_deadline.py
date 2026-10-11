"""Internal helper deadlines include startup and preserve cleanup."""

import asyncio
import logging
import sys
import unittest
from unittest import mock

from Cozter import utils
from Cozter.backends_agent import base


class InternalBackendDeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, backend, timeout):
        return await utils.run_internal_backend(
            backend, "/work", "private fixture prompt", "fixture-model",
            timeout=timeout, label="fixture helper",
            log=logging.getLogger("Cozter.tests.internal-deadline"),
            missing_executable_message="%s missing",
        )

    async def test_stalled_startup_is_cancelled_within_helper_deadline(self):
        cancelled = asyncio.Event()

        class Backend:
            executable = "fixture-cli"

            async def launch(self, *_args, **_kwargs):
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        with self.assertLogs("Cozter.tests.internal-deadline", level="WARNING") as logs:
            result = await asyncio.wait_for(self._run(Backend(), 0.02), timeout=1)
        self.assertEqual(result, "")
        self.assertTrue(cancelled.is_set())
        self.assertIn("including startup", "\n".join(logs.output))
        self.assertNotIn("private fixture prompt", "\n".join(logs.output))

    async def test_response_uses_remaining_budget_after_startup(self):
        entered_drain = asyncio.Event()
        cancelled_drain = asyncio.Event()

        class Backend:
            executable = "fixture-cli"

            async def launch(self, *_args, **_kwargs):
                await asyncio.sleep(0.06)
                return object()

        async def drain(*_args, **_kwargs):
            entered_drain.set()
            try:
                # This response fits a fresh 0.1s budget, but not the
                # remaining 0.04s after startup. The overall call must expire.
                await asyncio.sleep(0.06)
                return "would exceed the total deadline"
            finally:
                cancelled_drain.set()

        with mock.patch.object(utils, "drain_llm_subprocess", new=drain):
            with self.assertLogs("Cozter.tests.internal-deadline", level="WARNING"):
                result = await asyncio.wait_for(self._run(Backend(), 0.1), timeout=1)
        self.assertEqual(result, "")
        self.assertTrue(entered_drain.is_set())
        self.assertTrue(cancelled_drain.is_set())

    async def test_expired_response_discards_partial_text_and_reaps_child(self):
        class Backend:
            executable = sys.executable
            proc = None
            cleaned = False

            async def launch(self, *_args, **_kwargs):
                self.proc = await base.create_prompt_subprocess(
                    [sys.executable, "-c",
                     "import json,time; print(json.dumps({'text':'partial'}),flush=True); time.sleep(60)"],
                    "fixture",
                )
                return self.proc

            @staticmethod
            def extract_agent_text(event):
                return event.get("text")

            async def cleanup_process(self, _proc):
                self.cleaned = True

        backend = Backend()
        with self.assertLogs("Cozter.tests.internal-deadline", level="WARNING"):
            result = await asyncio.wait_for(self._run(backend, 0.3), timeout=5)
        self.assertEqual(result, "")
        self.assertIsNotNone(backend.proc)
        self.assertIsNotNone(backend.proc.returncode)
        self.assertTrue(backend.cleaned)

    async def test_absent_or_nonpositive_deadline_preserves_response(self):
        class Backend:
            executable = "fixture-cli"

            async def launch(self, *_args, **_kwargs):
                return object()

        async def drain(*_args, **_kwargs):
            await asyncio.sleep(0.01)
            return "complete"

        with mock.patch.object(utils, "drain_llm_subprocess", new=drain):
            for timeout in (None, 0, -1):
                with self.subTest(timeout=timeout):
                    self.assertEqual(await self._run(Backend(), timeout), "complete")

    async def test_user_cancellation_propagates(self):
        entered = asyncio.Event()

        class Backend:
            executable = "fixture-cli"

            async def launch(self, *_args, **_kwargs):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(self._run(Backend(), 1))
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
