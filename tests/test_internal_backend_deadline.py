"""Internal helpers run to completion unless explicitly bounded or cancelled."""

import asyncio
import logging
import sys
import unittest
from unittest import mock

from Cozter import process_launch
from Cozter import utils


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
                self.proc = await process_launch.create_prompt_subprocess(
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

    async def test_default_call_survives_two_hours_in_startup_and_drain_until_cancelled(self):
        entered_startup, release_startup, saw_output = (
            asyncio.Event(), asyncio.Event(), asyncio.Event(),
        )

        class Backend:
            executable = sys.executable
            proc = None
            cleaned = False

            async def launch(self, *_args, **_kwargs):
                entered_startup.set()
                await release_startup.wait()
                self.proc = await process_launch.create_prompt_subprocess(
                    [sys.executable, "-c",
                     "import json,time; print(json.dumps({'text':'partial'}),flush=True); time.sleep(60)"],
                    "fixture",
                )
                return self.proc

            @staticmethod
            def extract_agent_text(event):
                text = event.get("text")
                if text:
                    saw_output.set()
                return text

            async def cleanup_process(self, _proc):
                self.cleaned = True

        backend = Backend()
        real_timeout = asyncio.timeout
        logical_elapsed = 0.0
        timers = []
        owner = None

        def tracked_timeout(delay):
            timer = real_timeout(delay)
            if asyncio.current_task() is owner:
                timers.append((timer, None if delay is None else logical_elapsed + delay))
            return timer

        async def advance_two_hours():
            nonlocal logical_elapsed
            logical_elapsed += 7_201
            # Advance only this call's real Timeout contexts, leaving the
            # event-loop clock and the test's short guard deadlines untouched.
            for timer, due in timers:
                if due is not None and due <= logical_elapsed and not timer.expired():
                    timer.reschedule(asyncio.get_running_loop().time() - 1)
            for _ in range(3):
                await asyncio.sleep(0)

        with mock.patch.object(asyncio, "timeout", side_effect=tracked_timeout):
            owner = asyncio.create_task(utils.run_internal_backend(
                backend, "/work", "private fixture prompt", "fixture-model",
                label="fixture helper",
                log=logging.getLogger("Cozter.tests.internal-deadline"),
                missing_executable_message="%s missing",
            ))
            try:
                await asyncio.wait_for(entered_startup.wait(), timeout=5)
                await advance_two_hours()
                self.assertFalse(owner.done(), "default startup acquired a wall-clock deadline")
                self.assertIsNone(backend.proc)

                release_startup.set()
                await asyncio.wait_for(saw_output.wait(), timeout=5)
                await advance_two_hours()
                self.assertFalse(owner.done(), "default response draining acquired a wall-clock deadline")
                self.assertIsNone(backend.proc.returncode)
                self.assertTrue(timers, "the real response drain was not exercised")
                self.assertTrue(all(due is None for _timer, due in timers))

                owner.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(owner, timeout=5)
                self.assertIsNotNone(backend.proc.returncode)
                self.assertTrue(backend.cleaned)
            finally:
                release_startup.set()
                if not owner.done():
                    owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
                if backend.proc is not None and backend.proc.returncode is None:
                    await process_launch.reap_failed_subprocess(backend.proc)

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
