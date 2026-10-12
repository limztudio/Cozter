"""Durable bot commits release the event loop while retaining cancellation order."""

import asyncio
import contextvars
import os
import tempfile
import threading
import unittest
from unittest import mock

from Cozter import session, workspace
from Cozter.backends_agent.base import AgentResult, ChatEvent, DetachedTaskStatus
from Cozter.backends_bot import base as bot_base
from Cozter.tests.helpers import TestBot, temporary_config


class _PersistenceBot(TestBot):
    def __init__(self, path: str) -> None:
        super().__init__(["u1", "u2"])
        self.path = path
        self.watcher_threads: list[int] = []
        self.turns: list[str] = []

    def start_detached_task_watcher(self) -> None:
        self.watcher_threads.append(threading.get_ident())

    async def _current_workspace_for_turn(self, _uid: str, _chat: str) -> str:
        return self.path

    async def _run_turn(self, _uid: str, _chat: str, text: str, **_kwargs) -> None:
        self.turns.append(text)


class QueuePersistenceLatencyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = temporary.name
        self.enterContext(temporary_config({}))
        self.enterContext(mock.patch.object(workspace, "CONFIG_DIR", self.path))
        self.bot = _PersistenceBot(self.path)
        self.release = threading.Event()
        self.finished = threading.Event()
        self.tasks: list[asyncio.Task] = []
        self.worker_threads: list[int] = []
        self.worker_contexts: list[str] = []
        self.context = contextvars.ContextVar("queue_commit_test", default="missing")

    async def asyncSetUp(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.entered = asyncio.Event()
        self.assertTrue(self.loop.get_debug())
        self.assertEqual(self.loop.slow_callback_duration, 0.1)

    async def asyncTearDown(self) -> None:
        self.release.set()
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def _spawn(self, coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task

    def _block(self, owner, name: str, *, error: Exception | None = None) -> None:
        operation = getattr(owner, name)

        def blocking(*args, **kwargs):
            self.worker_threads.append(threading.get_ident())
            self.worker_contexts.append(self.context.get())
            first = len(self.worker_threads) == 1
            if first:
                self.loop.call_soon_threadsafe(self.entered.set)
                if not self.release.wait(timeout=5):
                    raise TimeoutError("test did not release durable operation")
            try:
                if error is not None:
                    raise error
                return operation(*args, **kwargs)
            finally:
                if first:
                    self.finished.set()

        self.enterContext(mock.patch.object(owner, name, side_effect=blocking))

    async def _wait_blocked(self, lock: asyncio.Lock) -> None:
        await asyncio.wait_for(self.entered.wait(), timeout=1)
        self.assertTrue(lock.locked())
        self.assertNotEqual(self.worker_threads[0], threading.get_ident())
        heartbeat = asyncio.Event()
        self.loop.call_soon(heartbeat.set)
        await asyncio.wait_for(heartbeat.wait(), timeout=0.5)

    async def _cancel_twice(self, task: asyncio.Task, lock: asyncio.Lock) -> None:
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertTrue(lock.locked())
            self.assertFalse(self.finished.is_set())

    async def _finish_cancelled(self, task: asyncio.Task) -> None:
        self.release.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        # A returned cancellation cannot leave its executor mutation live.
        self.assertTrue(self.finished.is_set())

    async def _stage(self, uid: str = "u1", entry_id: str = "entry") -> dict:
        return await self.bot._stage_reply_delivery(
            uid, entry_id, "chat", self.path,
            AgentResult(events=[ChatEvent(kind="text", content="durable reply [[await]]")]),
            allow_await=True,
        )

    async def _register_detached(self, task_id: str) -> bool:
        return await self.bot._register_detached_task(
            uid="u1", chat_id="chat", workspace_path=self.path,
            session_id=None, backend_name="claude_code", task_id=task_id,
            validate=False,
        )

    async def test_detached_registration_preserves_fifo_context_and_watcher_thread(self) -> None:
        self._block(os, "fsync")
        self.context.set("request-context")
        first = self._spawn(self._register_detached("first-task"))
        try:
            await self._wait_blocked(self.bot._detached_task_file_lock)
            second = self._spawn(self._register_detached("second-task"))
            await asyncio.sleep(0)
            self.assertFalse(second.done())
            self.release.set()
            self.assertEqual(await asyncio.wait_for(asyncio.gather(first, second), timeout=2), [True, True])
        finally:
            self.release.set()
        records = await self.bot._list_detached_task_records()
        self.assertEqual([record["task_id"] for record in records], ["first-task", "second-task"])
        self.assertEqual(self.worker_contexts, ["request-context", "request-context"])
        self.assertTrue(all(thread != threading.get_ident() for thread in self.worker_threads))
        self.assertEqual(self.bot.watcher_threads, [threading.get_ident()] * 2)
        with mock.patch.object(self.bot, "_write_detached_tasks_file") as write:
            self.assertTrue(await self._register_detached("first-task"))
            write.assert_not_called()
        self.assertEqual(len(self.bot.watcher_threads), 2)

    async def test_cancelled_detached_registration_commits_before_watcher_on_loop(self) -> None:
        self._block(os, "fsync")
        registering = self._spawn(self._register_detached("provider-task"))
        await self._wait_blocked(self.bot._detached_task_file_lock)
        await self._cancel_twice(registering, self.bot._detached_task_file_lock)
        await self._finish_cancelled(registering)
        records = await self.bot._list_detached_task_records()
        self.assertEqual([record["task_id"] for record in records], ["provider-task"])
        self.assertEqual(self.bot.watcher_threads, [threading.get_ident()])

    async def test_cancelled_detached_update_finishes_before_removal(self) -> None:
        await self._register_detached("first-task")
        await self._register_detached("other-task")
        record = (await self.bot._list_detached_task_records())[0]
        self._block(os, "fsync")
        updating = self._spawn(self.bot._update_detached_task_record(
            record["id"], delivery_text="committed callback",
        ))
        await self._wait_blocked(self.bot._detached_task_file_lock)
        await self._cancel_twice(updating, self.bot._detached_task_file_lock)
        removing = self._spawn(self.bot._remove_detached_task_record(record["id"]))
        await asyncio.sleep(0)
        self.assertFalse(removing.done())
        await self._finish_cancelled(updating)
        await asyncio.wait_for(removing, timeout=2)
        self.assertIsNone(await self.bot._get_detached_task_record(record["id"]))
        self.assertEqual([record["task_id"] for record in await self.bot._list_detached_task_records()],
                         ["other-task"])
        self.assertFalse(self.bot._detached_task_file_lock.locked())

    async def test_queue_commits_keep_loop_responsive_and_preserve_fifo_context(self) -> None:
        self._block(os, "fsync")
        self.context.set("request-context")
        first = self._spawn(self.bot._persist_enqueue("u1", "first", "chat"))
        await self._wait_blocked(self.bot._queue_file_lock)
        second = self._spawn(self.bot._persist_enqueue("u1", "second", "chat"))
        third = self._spawn(self.bot._persist_enqueue("u1", "third", "chat", ephemeral=True))
        await asyncio.sleep(0)
        self.assertFalse(second.done())
        self.assertFalse(third.done())
        self.release.set()
        entry_ids = await asyncio.wait_for(asyncio.gather(first, second, third), timeout=2)
        rows = self.bot._read_queue_file()["u1"]
        self.assertEqual([row["text"] for row in rows], ["first", "second", "third"])
        self.assertEqual([row["id"] for row in rows], entry_ids)
        self.assertTrue(rows[2]["ephemeral"])
        self.assertEqual(self.worker_contexts, ["request-context"] * 3)
        self.assertFalse(self.bot._queue_file_lock.locked())

    async def test_cancelled_admission_keeps_outer_lock_until_stop_can_clear(self) -> None:
        self._block(os, "fsync")
        admission = self._spawn(self.bot._persist_admitted_dispatch_entry(
            "u1", "cancelled prompt", "chat", 0,
        ))
        await self._wait_blocked(self.bot._queue_file_lock)
        admission_lock = self.bot._ensure_dispatch_admission_lock("u1")
        await self._cancel_twice(admission, admission_lock)
        stopping = self._spawn(self.bot._cancel_user_work("u1"))
        await asyncio.sleep(0)
        self.assertFalse(stopping.done())
        self.assertTrue(admission_lock.locked())
        await self._finish_cancelled(admission)
        await asyncio.wait_for(stopping, timeout=2)
        self.assertEqual(self.bot._read_queue_file(), {})
        self.assertFalse(admission_lock.locked())
        self.assertFalse(self.bot._queue_file_lock.locked())

    async def _check_detached_poll_during_cancelled_queue_commit(self, state: str) -> None:
        await asyncio.to_thread(
            self.bot._write_detached_tasks_file,
            {"tasks": [{"id": "detached-entry", "user_id": "u1", "chat_id": "chat",
                        "workspace_path": self.path, "session_id": None,
                        "backend_name": "claude_code", "task_id": "provider-task"}]},
        )
        await self.bot._persist_enqueue("u1", "queued prompt", "chat")
        output_started = asyncio.Event()
        release_output = asyncio.Event()

        async def output(_ws, _task_id):
            output_started.set()
            await release_output.wait()
            return "provider output completed during stop"

        backend = mock.Mock()
        backend.get_detached_task_status = mock.AsyncMock(return_value=DetachedTaskStatus(state))
        backend.get_detached_task_output = mock.AsyncMock(side_effect=output)
        backend.stop_detached_task = mock.AsyncMock(return_value=True)
        stopping = None
        with (
            mock.patch.object(bot_base.backends_agent, "get_backend", return_value=backend),
            mock.patch.object(self.bot, "send_text", new=mock.AsyncMock()) as send,
        ):
            polling = self._spawn(self.bot._check_detached_tasks())
            try:
                await asyncio.wait_for(output_started.wait(), timeout=1)
                self._block(os, "fsync")
                stopping = self._spawn(self.bot._cancel_user_work("u1"))
                await self._wait_blocked(self.bot._queue_file_lock)
                release_output.set()
                await asyncio.wait_for(polling, timeout=2)
                self.assertFalse(stopping.done())
                send.assert_not_awaited()
            finally:
                release_output.set()
                self.release.set()
                await asyncio.gather(*[task for task in (polling, stopping) if task], return_exceptions=True)
            self.assertEqual(stopping.result(), (False, True))
            self.assertEqual(await self.bot._list_detached_task_records(), [])
            send.assert_not_awaited()

    async def test_detached_completion_during_stop_queue_commit_cannot_begin_a_send(self) -> None:
        await self._check_detached_poll_during_cancelled_queue_commit("done")

    async def test_detached_blocked_report_during_stop_queue_commit_cannot_begin_a_send(self) -> None:
        await self._check_detached_poll_during_cancelled_queue_commit("blocked")

    async def test_overlapping_cancels_hold_outbound_gate_until_both_finish(self) -> None:
        entered = [asyncio.Event(), asyncio.Event()]
        releases = [asyncio.Event(), asyncio.Event()]
        callbacks: list[str] = []
        cleanup_count = 0

        async def delayed_cleanup(_uid):
            nonlocal cleanup_count
            index = cleanup_count
            cleanup_count += 1
            entered[index].set()
            await releases[index].wait()
            return 0

        async def delivery():
            callbacks.append("accepted")

        with mock.patch.object(self.bot, "_cancel_detached_tasks", side_effect=delayed_cleanup):
            first = self._spawn(self.bot._cancel_user_work("u1"))
            await asyncio.wait_for(entered[0].wait(), timeout=1)
            second = self._spawn(self.bot._cancel_user_work("u1"))
            try:
                await asyncio.sleep(0)
                releases[0].set()
                await asyncio.wait_for(first, timeout=1)
                await asyncio.wait_for(entered[1].wait(), timeout=1)
                await self.bot._run_detached_delivery("u1", delivery())
                self.assertEqual(callbacks, [])
                releases[1].set()
                await asyncio.wait_for(second, timeout=1)
                await self.bot._run_detached_delivery("u1", delivery())
                self.assertEqual(callbacks, ["accepted"])
                self.assertEqual(first.result(), (False, False))
                self.assertEqual(second.result(), (False, False))
            finally:
                for event in releases:
                    event.set()
                await asyncio.gather(first, second, return_exceptions=True)

    async def test_cancelled_stop_retires_outbound_gate(self) -> None:
        callbacks: list[str] = []

        async def delivery():
            callbacks.append("accepted")

        await self.bot._queue_file_lock.acquire()
        stopping = self._spawn(self.bot._cancel_user_work("u1"))
        try:
            await asyncio.sleep(0)
            await self.bot._run_detached_delivery("u1", delivery())
            self.assertEqual(callbacks, [])
            stopping.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(stopping, timeout=1)
        finally:
            self.bot._queue_file_lock.release()
        await self.bot._run_detached_delivery("u1", delivery())
        self.assertEqual(callbacks, ["accepted"])

    async def test_new_admission_waits_until_prior_reply_cancellation_finishes(self) -> None:
        await self._stage()
        self._block(os, "fsync")
        stopping = self._spawn(self.bot._cancel_user_work("u1"))
        admitting = None
        enqueue_started = asyncio.Event()
        enqueue = self.bot._persist_enqueue

        async def observe_enqueue(*args, **kwargs):
            enqueue_started.set()
            return await enqueue(*args, **kwargs)

        try:
            await self._wait_blocked(self.bot._reply_delivery_file_lock)
            generation = self.bot._cancel_generations.get("u1", 0)
            with mock.patch.object(self.bot, "_persist_enqueue", side_effect=observe_enqueue):
                admitting = self._spawn(self.bot._persist_admitted_dispatch_entry(
                    "u1", "new prompt after stop", "chat", generation,
                ))
                with self.assertRaises(TimeoutError):
                    await asyncio.wait_for(enqueue_started.wait(), timeout=0.05)
                self.assertFalse(admitting.done())
                self.assertEqual(self.bot._read_queue_file(), {})
                self.release.set()
                self.assertEqual(await asyncio.wait_for(stopping, timeout=2), (False, True))
                entry_id = await asyncio.wait_for(admitting, timeout=2)
                self.assertTrue(enqueue_started.is_set())
        finally:
            self.release.set()
            await asyncio.gather(*[task for task in (stopping, admitting) if task], return_exceptions=True)
        self.assertEqual(self.bot._read_queue_file()["u1"], [
            {"id": entry_id, "text": "new prompt after stop", "chat_id": "chat"},
        ])
        self.assertNotIn("u1", self.bot._pending_reply_delivery_users)

    async def test_durable_cancellation_cleanup_counts_as_active_until_finished(self) -> None:
        await self.bot._persist_enqueue("u1", "queued prompt", "chat")
        self._block(os, "fsync")
        stopping = self._spawn(self.bot._cancel_user_work("u1"))
        try:
            await self._wait_blocked(self.bot._queue_file_lock)
            self.assertTrue(self.bot.has_active_turns())
            self.release.set()
            await asyncio.wait_for(stopping, timeout=2)
        finally:
            self.release.set()
            await asyncio.gather(stopping, return_exceptions=True)
        self.assertFalse(self.bot.has_active_turns())

    async def test_new_admission_waits_for_upload_cancellation_cleanup(self) -> None:
        upload_started = asyncio.Event()
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        enqueue_started = asyncio.Event()

        async def upload():
            upload_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_started.set()
                await release_cleanup.wait()
                raise

        uploading = self._spawn(upload())
        self.bot._attachment_tasks["u1"] = {uploading}
        await asyncio.wait_for(upload_started.wait(), timeout=1)
        stopping = self._spawn(self.bot._cancel_user_work("u1"))
        admitting = None
        enqueue = self.bot._persist_enqueue

        async def observe_enqueue(*args, **kwargs):
            enqueue_started.set()
            return await enqueue(*args, **kwargs)

        try:
            await asyncio.wait_for(cleanup_started.wait(), timeout=1)
            with mock.patch.object(self.bot, "_persist_enqueue", side_effect=observe_enqueue):
                admitting = self._spawn(self.bot._persist_admitted_dispatch_entry(
                    "u1", "new prompt after stop", "chat",
                    self.bot._cancel_generations.get("u1", 0),
                ))
                with self.assertRaises(TimeoutError):
                    await asyncio.wait_for(enqueue_started.wait(), timeout=0.05)
                self.assertFalse(admitting.done())
                release_cleanup.set()
                self.assertEqual(await asyncio.wait_for(stopping, timeout=2), (False, True))
                entry_id = await asyncio.wait_for(admitting, timeout=2)
        finally:
            release_cleanup.set()
            await asyncio.gather(*[task for task in (uploading, stopping, admitting) if task], return_exceptions=True)
        self.assertEqual(self.bot._read_queue_file()["u1"][0]["id"], entry_id)
        self.assertFalse(self.bot.has_active_turns())

    async def test_cancelled_completion_finishes_before_next_enqueue(self) -> None:
        first_id = await self.bot._persist_enqueue("u1", "first", "chat")
        second_id = await self.bot._persist_enqueue("u1", "second", "chat")
        self._block(os, "fsync")
        completing = self._spawn(self.bot._persist_complete("u1", first_id))
        await self._wait_blocked(self.bot._queue_file_lock)
        await self._cancel_twice(completing, self.bot._queue_file_lock)
        third = self._spawn(self.bot._persist_enqueue("u1", "third", "chat"))
        await asyncio.sleep(0)
        self.assertFalse(third.done())
        await self._finish_cancelled(completing)
        third_id = await asyncio.wait_for(third, timeout=2)
        rows = self.bot._read_queue_file()["u1"]
        self.assertEqual([row["id"] for row in rows], [second_id, third_id])

    async def test_cancelled_stage_keeps_durable_row_and_updates_pending_on_loop(self) -> None:
        self._block(os, "fsync")
        staging = self._spawn(self._stage())
        await self._wait_blocked(self.bot._reply_delivery_file_lock)
        await self._cancel_twice(staging, self.bot._reply_delivery_file_lock)
        await self._finish_cancelled(staging)
        self.assertIn("u1", self.bot._pending_reply_delivery_users)
        self.assertEqual(self.bot.watcher_threads, [threading.get_ident()])
        rows = await self.bot._list_reply_delivery_records()
        self.assertEqual(rows, [{"id": "entry", "user_id": "u1", "chat_id": "chat",
                                 "messages": ["durable reply"], "awaiting": True}])

    async def test_cancelled_reply_clear_keeps_delivery_lock_and_clears_pending(self) -> None:
        await self._stage()
        self._block(os, "fsync")
        clearing = self._spawn(self.bot._clear_reply_deliveries("u1"))
        await self._wait_blocked(self.bot._reply_delivery_file_lock)
        delivery_lock = self.bot._reply_delivery_lock("u1")
        await self._cancel_twice(clearing, delivery_lock)
        competing = self._spawn(delivery_lock.acquire())
        await asyncio.sleep(0)
        self.assertFalse(competing.done())
        await self._finish_cancelled(clearing)
        await asyncio.wait_for(competing, timeout=1)
        delivery_lock.release()
        self.assertNotIn("u1", self.bot._pending_reply_delivery_users)
        self.assertEqual(await self.bot._list_reply_delivery_records(), [])

    async def test_cancelled_reply_removal_updates_only_its_user_pending_flag(self) -> None:
        await self._stage()
        await self._stage("u2", "other-entry")
        self._block(os, "fsync")
        removing = self._spawn(self.bot._remove_reply_delivery_record("entry"))
        await self._wait_blocked(self.bot._reply_delivery_file_lock)
        await self._cancel_twice(removing, self.bot._reply_delivery_file_lock)
        await self._finish_cancelled(removing)
        self.assertEqual(self.bot._pending_reply_delivery_users, {"u2"})
        self.assertEqual([row["id"] for row in await self.bot._list_reply_delivery_records()],
                         ["other-entry"])

    async def test_cancelled_ephemeral_creation_cleans_session_without_running_turn(self) -> None:
        self._block(os, "fsync")
        turning = self._spawn(self.bot._run_ephemeral_turn("u1", "chat", "scheduled"))
        file_lock = workspace.get_lock(self.path)
        await self._wait_blocked(file_lock)
        await self._cancel_twice(turning, file_lock)
        await self._finish_cancelled(turning)
        self.assertEqual(self.bot.turns, [])
        self.assertEqual(os.listdir(os.path.join(self.path, ".cozter", "sessions")), [])
        self.assertFalse(file_lock.locked())

    async def test_cancelled_ephemeral_deletion_settles_before_return(self) -> None:
        self._block(session, "delete_session")
        turning = self._spawn(self.bot._run_ephemeral_turn("u1", "chat", "scheduled"))
        file_lock = workspace.get_lock(self.path)
        await self._wait_blocked(file_lock)
        await self._cancel_twice(turning, file_lock)
        await self._finish_cancelled(turning)
        self.assertEqual(self.bot.turns, ["scheduled"])
        self.assertEqual(os.listdir(os.path.join(self.path, ".cozter", "sessions")), [])
        self.assertFalse(file_lock.locked())

    async def test_repeated_cancel_while_ephemeral_cleanup_waits_still_deletes_session(self) -> None:
        file_lock = workspace.get_lock(self.path)
        lock_held = asyncio.Event()

        async def retain_file_lock(*_args, **_kwargs):
            await file_lock.acquire()
            lock_held.set()

        with (
            mock.patch.object(self.bot, "_run_turn", side_effect=retain_file_lock),
            mock.patch.object(session, "delete_session", wraps=session.delete_session) as delete,
        ):
            turning = self._spawn(self.bot._run_ephemeral_turn("u1", "chat", "scheduled"))
            await asyncio.wait_for(lock_held.wait(), timeout=1)
            try:
                await asyncio.sleep(0)
                for _ in range(2):
                    turning.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(turning.done())
                    self.assertTrue(file_lock.locked())
                    delete.assert_not_called()
                self.assertEqual(len(os.listdir(os.path.join(self.path, ".cozter", "sessions"))), 1)
            finally:
                file_lock.release()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(turning, timeout=2)
            delete.assert_called_once()
        self.assertEqual(os.listdir(os.path.join(self.path, ".cozter", "sessions")), [])
        self.assertFalse(file_lock.locked())

    async def test_cancel_before_file_lock_does_not_start_a_write(self) -> None:
        await self.bot._queue_file_lock.acquire()
        try:
            with mock.patch.object(self.bot, "_write_queue_file") as write:
                waiting = self._spawn(self.bot._persist_enqueue("u1", "prompt", "chat"))
                await asyncio.sleep(0)
                waiting.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await waiting
                write.assert_not_called()
        finally:
            self.bot._queue_file_lock.release()
        self.assertEqual(self.bot._read_queue_file(), {})

    async def test_fsync_failure_preserves_previous_rows_and_releases_lock(self) -> None:
        entry_id = await self.bot._persist_enqueue("u1", "previous", "chat")
        with mock.patch.object(os, "fsync", side_effect=OSError("fixture disk failure")):
            with self.assertRaisesRegex(OSError, "fixture disk failure"):
                await self.bot._persist_enqueue("u1", "uncommitted", "chat")
        self.assertEqual([row["id"] for row in self.bot._read_queue_file()["u1"]], [entry_id])
        self.assertFalse(self.bot._queue_file_lock.locked())

    async def test_failed_commit_during_cancellation_logs_error_and_replays_cancel(self) -> None:
        self._block(os, "fsync", error=OSError("fixture cancelled disk failure"))
        writing = self._spawn(self.bot._persist_enqueue("u1", "uncommitted", "chat"))
        await self._wait_blocked(self.bot._queue_file_lock)
        await self._cancel_twice(writing, self.bot._queue_file_lock)
        with self.assertLogs("Cozter.backends_bot.base", level="ERROR") as captured:
            await self._finish_cancelled(writing)
        self.assertEqual(len(captured.output), 1)
        self.assertIn("Durable file operation failed during cancellation", captured.output[0])
        self.assertIn("fixture cancelled disk failure", captured.output[0])
        self.assertEqual(self.bot._read_queue_file(), {})
        self.assertFalse(self.bot._queue_file_lock.locked())


if __name__ == "__main__":
    unittest.main()
