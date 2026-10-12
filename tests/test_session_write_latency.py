"""Durable session writes stay off-loop and retain ownership after /stop."""

import asyncio
import contextvars
import tempfile
import threading
import unittest
from unittest import mock

from Cozter import agent, flexible, session, workspace
from Cozter.backends_agent.base import AgentResult, ChatEvent


class _WriteBackend:
    name = "session-write-test"
    default_summary_model = "summary-default"
    supports_typed_plugins = True


class SessionWriteLatencyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = temporary.name
        self.session_id = session.create_session(self.path, name="Work")["id"]
        self.release = threading.Event()
        self.owners: list[asyncio.Task] = []
        self.calls: list[list[dict]] = []
        self.writer_threads: list[int] = []
        self.writer_contexts: list[str] = []
        self.context = contextvars.ContextVar("session_write_test", default="missing")
        self.fail_first_write = False
        original_background = agent.create_background_task

        def capture(coro, *, name, log=None):
            task = original_background(coro, name=name, log=log)
            self.owners.append(task)
            return task

        async def drive(*_args, **_kwargs):
            return AgentResult(text="finished"), False

        self.enterContext(mock.patch.object(
            agent, "create_background_task", side_effect=capture,
        ))
        self.enterContext(mock.patch.object(
            agent.backends_agent, "get_backend", return_value=_WriteBackend(),
        ))
        self.enterContext(mock.patch.object(agent, "_drive_backend", side_effect=drive))
        self.enterContext(mock.patch.object(
            agent, "_judge_draft", new=mock.AsyncMock(return_value=(
                flexible.JudgeVerdict(should_continue=False), False,
            )),
        ))
        self.enterContext(mock.patch.object(
            agent, "_run_post_turn_maintenance", new=mock.AsyncMock(),
        ))

    async def asyncSetUp(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.assertTrue(self.loop.get_debug())
        self.assertEqual(self.loop.slow_callback_duration, 0.1)
        self.started = asyncio.Event()

    async def asyncTearDown(self) -> None:
        self.release.set()
        await asyncio.gather(*self.owners, return_exceptions=True)

    def _block_appends(self) -> None:
        append = session.append_messages

        def blocked(path, session_id, messages):
            self.calls.append([dict(message) for message in messages])
            call_no = len(self.calls)
            self.writer_threads.append(threading.get_ident())
            self.writer_contexts.append(self.context.get())
            self.loop.call_soon_threadsafe(self.started.set)
            if not self.release.wait(timeout=5):
                raise TimeoutError("test did not release the session write")
            if self.fail_first_write and call_no == 1:
                raise OSError("session write fixture failure")
            return append(path, session_id, messages)

        self.enterContext(mock.patch.object(session, "append_messages", side_effect=blocked))

    def _turn(self, prompt: str = "do the work") -> asyncio.Task:
        return asyncio.create_task(agent.run(
            prompt, self.path, 1,
            backend_name=_WriteBackend.name, session_id=self.session_id,
        ))

    async def _assert_resuming_is_blocked(
        self, resuming: asyncio.Task, prompts: list[str],
    ) -> None:
        await asyncio.sleep(0)
        self.assertFalse(resuming.done())
        self.assertEqual(len(prompts), 1)

    async def _wait_for_blocked_write(self) -> None:
        await asyncio.wait_for(self.started.wait(), timeout=1)
        self.assertNotEqual(self.writer_threads[0], threading.get_ident())
        self.assertTrue(workspace.get_lock(self.path).locked())
        # A scheduled callback must run while the real writer is blocked.
        heartbeat = asyncio.Event()
        self.loop.call_soon(heartbeat.set)
        await asyncio.wait_for(heartbeat.wait(), timeout=0.5)

    async def _finish_owners(self) -> None:
        await asyncio.wait_for(asyncio.gather(*self.owners), timeout=2)

    def _messages(self) -> list[dict]:
        data = session.load_session(self.path, self.session_id)
        assert data is not None
        return data["messages"]

    async def test_normal_write_keeps_loop_responsive_and_preserves_context(self) -> None:
        self._block_appends()
        self.context.set("request-context")
        turning = self._turn()
        try:
            await self._wait_for_blocked_write()
            self.assertFalse(turning.done())
        finally:
            self.release.set()
            await turning
        await self._finish_owners()
        self.assertEqual(self.writer_contexts, ["request-context"])
        self.assertEqual(self._messages(), [
            {"role": "user", "content": "do the work"},
            {"role": "assistant", "content": "finished"},
        ])

    async def test_repeated_cancellation_retains_lock_and_avoids_duplicate_pair(self) -> None:
        self._block_appends()
        turning = self._turn()
        waiter = None
        try:
            await self._wait_for_blocked_write()
            turning.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(turning, timeout=0.5)
            turning.cancel()
            owner = next(task for task in self.owners if task.get_name().startswith("log-session-turn:"))
            owner.cancel()
            await asyncio.sleep(0)
            owner.cancel()
            await asyncio.sleep(0)
            waiter = asyncio.create_task(workspace.get_lock(self.path).acquire())
            await asyncio.sleep(0)
            self.assertFalse(owner.done())
            self.assertFalse(waiter.done())
            self.assertEqual(len(self.calls), 1)
        finally:
            self.release.set()
            await asyncio.gather(turning, return_exceptions=True)
            await self._finish_owners()
            if waiter is not None:
                await waiter
                workspace.get_lock(self.path).release()
        self.assertEqual(len(self.calls), 1)
        messages = self._messages()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["content"], "finished")

    async def test_failed_normal_write_after_cancel_persists_one_interrupted_pair(self) -> None:
        self.fail_first_write = True
        self._block_appends()
        with self.assertLogs(agent.logger, level="ERROR") as output:
            turning = self._turn()
            try:
                await self._wait_for_blocked_write()
                turning.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(turning, timeout=0.5)
                self.assertEqual(len(self.calls), 1)
            finally:
                self.release.set()
                await asyncio.gather(turning, return_exceptions=True)
                await self._finish_owners()
        self.assertTrue(any("Failed to log session" in line for line in output.output))
        self.assertEqual(len(self.calls), 2)
        messages = self._messages()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["content"], "do the work")
        self.assertIn("[Interrupted turn", messages[1]["content"])

    async def test_next_turn_reads_the_cancelled_normal_write_after_it_commits(self) -> None:
        self._block_appends()
        prompts: list[str] = []

        async def drive(_backend, _ws, prompt, *_args, **_kwargs):
            prompts.append(prompt)
            return AgentResult(text="finished" if len(prompts) == 1 else "resumed"), False

        resuming = None
        with mock.patch.object(agent, "_drive_backend", side_effect=drive):
            turning = self._turn()
            try:
                await self._wait_for_blocked_write()
                turning.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(turning, timeout=0.5)
                with (
                    mock.patch.object(session, "load_session", wraps=session.load_session) as load,
                    mock.patch.object(agent.colony, "get_items", wraps=agent.colony.get_items) as memory,
                ):
                    resuming = self._turn("resume the work")
                    await self._assert_resuming_is_blocked(resuming, prompts)
                    load.assert_not_called()
                    memory.assert_not_called()
                    self.release.set()
                    await asyncio.wait_for(resuming, timeout=2)
            finally:
                self.release.set()
                await asyncio.gather(*[task for task in (turning, resuming) if task], return_exceptions=True)
        await self._finish_owners()
        self.assertEqual(len(prompts), 2)
        self.assertIn("do the work", prompts[1])
        self.assertIn("finished", prompts[1])
        self.assertEqual(len(self._messages()), 4)

    async def test_next_turn_reads_the_interrupted_write_after_it_commits(self) -> None:
        self._block_appends()
        entered = asyncio.Event()
        prompts: list[str] = []

        async def drive(_backend, _ws, prompt, *_args, on_event, **_kwargs):
            prompts.append(prompt)
            if len(prompts) == 1:
                await on_event(ChatEvent(kind="text", content="partial work"))
                entered.set()
                await asyncio.Event().wait()
            return AgentResult(text="resumed"), False

        resuming = None
        with mock.patch.object(agent, "_drive_backend", side_effect=drive):
            turning = self._turn()
            try:
                await asyncio.wait_for(entered.wait(), timeout=1)
                turning.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(turning, timeout=0.5)
                await self._wait_for_blocked_write()
                resuming = self._turn("resume the work")
                await self._assert_resuming_is_blocked(resuming, prompts)
                self.release.set()
                await asyncio.wait_for(resuming, timeout=2)
            finally:
                self.release.set()
                await asyncio.gather(*[task for task in (turning, resuming) if task], return_exceptions=True)
        await self._finish_owners()
        self.assertEqual(len(prompts), 2)
        self.assertIn("partial work", prompts[1])
        self.assertIn("[Interrupted turn", prompts[1])
        self.assertEqual(len(self._messages()), 4)

    async def test_cancelling_the_next_turn_does_not_cancel_the_pending_writer(self) -> None:
        self._block_appends()
        turning = self._turn()
        waiting = None
        try:
            await self._wait_for_blocked_write()
            turning.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(turning, timeout=0.5)
            waiting = self._turn()
            await asyncio.sleep(0)
            waiting.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(waiting, timeout=0.5)
            self.assertTrue(workspace.get_lock(self.path).locked())
            self.assertEqual(len(self.calls), 1)
            self.assertFalse(next(task for task in self.owners
                                  if task.get_name().startswith("log-session-turn:")).done())
        finally:
            self.release.set()
            await asyncio.gather(*[task for task in (turning, waiting) if task], return_exceptions=True)
        await self._finish_owners()
        self.assertEqual(len(self._messages()), 2)

    async def test_detached_launch_reads_the_pending_normal_write_after_it_commits(self) -> None:
        self._block_appends()
        session.set_last_session(self.path, 1, self.session_id)
        backend = _WriteBackend()
        backend.supports_detached_tasks = True
        backend.launch_detached = mock.AsyncMock(return_value="provider-task")
        launching = None
        with mock.patch.object(agent.backends_agent, "get_backend", return_value=backend):
            turning = self._turn()
            try:
                await self._wait_for_blocked_write()
                turning.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(turning, timeout=0.5)
                launching = asyncio.create_task(agent.launch_detached(
                    "continue in the background", self.path, 1,
                    backend_name=_WriteBackend.name,
                ))
                await asyncio.sleep(0)
                self.assertFalse(launching.done())
                backend.launch_detached.assert_not_awaited()
                self.release.set()
                launch = await asyncio.wait_for(launching, timeout=2)
            finally:
                self.release.set()
                await asyncio.gather(*[task for task in (turning, launching) if task], return_exceptions=True)
        await self._finish_owners()
        self.assertEqual(launch.session_id, self.session_id)
        self.assertIn("finished", backend.launch_detached.await_args.args[1])
        self.assertEqual(len(self._messages()), 4)

    async def test_interrupted_write_survives_owner_cancellation(self) -> None:
        entered = asyncio.Event()

        async def interrupted(*_args, on_event, **_kwargs):
            await on_event(ChatEvent(kind="text", content="partial work"))
            entered.set()
            await asyncio.Event().wait()

        self._block_appends()
        with mock.patch.object(agent, "_drive_backend", side_effect=interrupted):
            turning = self._turn()
            try:
                await asyncio.wait_for(entered.wait(), timeout=1)
                turning.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(turning, timeout=0.5)
                await self._wait_for_blocked_write()
                self.owners[0].cancel()
                await asyncio.sleep(0)
                self.assertFalse(self.owners[0].done())
                self.assertTrue(workspace.get_lock(self.path).locked())
            finally:
                self.release.set()
                await asyncio.gather(turning, return_exceptions=True)
                await self._finish_owners()
        messages = self._messages()
        self.assertEqual(len(messages), 2)
        self.assertIn("partial work", messages[1]["content"])
        self.assertIn("[Interrupted turn", messages[1]["content"])

if __name__ == "__main__":
    unittest.main()
