"""Scheduled-job admission and start notifications reflect actual queue state."""

import asyncio
import os
import tempfile
import unittest
from unittest import mock

from Cozter import workspace
from Cozter.queue_store import ScheduledOccurrence
from Cozter.tests.helpers import OfflineBot


class _ScheduleBot(OfflineBot):
    def __init__(self, workspace_path: str) -> None:
        super().__init__(["u1"])
        self.workspace_path = workspace_path
        self.notices: list[tuple[str, list[dict], int, bool]] = []
        self.ran: list[tuple[str, str]] = []
        self.foreground_started = asyncio.Event()
        self.foreground_release = asyncio.Event()
        self.notice_started = asyncio.Event()
        self.slow_notice = ""
        self.notice_cancelled = False
        self.drains: list[asyncio.Task] = []

    async def send_text(self, _chat_id: str, text: str, *, rich: bool = False):
        msg_queue = self._message_queues.get("u1")
        self.notices.append((
            text,
            self._read_queue_file().get("u1", []),
            msg_queue.qsize() if msg_queue is not None else 0,
            self._ensure_task_lock("u1").locked(),
        ))
        if self.slow_notice and text.startswith(self.slow_notice):
            self.notice_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.notice_cancelled = True
                raise

    async def _current_workspace_for_turn(self, _uid: str, _chat_id: str):
        return self.workspace_path

    async def _run_turn(
        self, _uid: str, _chat_id: str, text: str, *,
        session_id: str | None = None, queue_entry_id: str | None = None,
        is_scheduled: bool = False,
    ) -> None:
        self.ran.append(("scheduled" if is_scheduled else "normal", text))
        if text == "active":
            self.foreground_started.set()
            await self.foreground_release.wait()

    def _start_queue_drain(self, uid: str) -> asyncio.Task[None]:
        task = asyncio.create_task(self._drain_message_queue(uid))
        self.drains.append(task)
        return task


class ScheduleExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.enterContext(mock.patch.object(workspace, "CONFIG_DIR", self.tmp.name))
        self.bot = _ScheduleBot(self.tmp.name)
        self.schedule = {"command": "scheduled work", "chat_id": "chat",
                         "platform_id": self.bot.platform_id}

    async def asyncTearDown(self) -> None:
        self.bot.foreground_release.set()
        for task in self.bot.drains:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.bot.drains, return_exceptions=True)

    async def _finish_drains(self) -> None:
        await asyncio.wait_for(asyncio.gather(*self.bot.drains), timeout=1)

    async def test_direct_admission_requires_explicit_matching_platform(self) -> None:
        with (
            mock.patch.object(self.bot, "_admit_queue_entry", new=mock.AsyncMock()) as persist,
            mock.patch.object(self.bot, "_current_workspace_for_turn", new=mock.AsyncMock()) as workspace_lookup,
            mock.patch.object(self.bot, "authorized") as authorized,
        ):
            for platform in (None, "", "other-platform", True, 1):
                with self.subTest(platform=platform):
                    schedule = {key: value for key, value in self.schedule.items() if key != "platform_id"}
                    if platform is not None:
                        schedule["platform_id"] = platform
                    await self.bot._fire_schedule("u1", schedule, workspace_path=self.tmp.name)
            persist.assert_not_awaited()
            workspace_lookup.assert_not_awaited()
            authorized.assert_not_called()
        self.assertEqual(self.bot.notices, [])
        self.assertEqual(self.bot.drains, [])
        self.assertEqual(self.bot._read_queue_file(), {})

    async def test_queue_notice_follows_durable_and_in_memory_admission(self) -> None:
        await self.bot._fire_schedule("u1", self.schedule, workspace_path=self.tmp.name)
        queued = self.bot.notices[0]
        self.assertEqual(queued[0], "⏰ Queued scheduled task: scheduled work")
        self.assertEqual(len(queued[1]), 1)
        self.assertEqual(queued[1][0]["text"], "scheduled work")
        self.assertIs(queued[1][0]["ephemeral"], True)
        self.assertEqual(queued[2], 1)

        await self._finish_drains()
        started = self.bot.notices[1]
        self.assertEqual(started[0], "⏰ Starting scheduled task: scheduled work")
        self.assertTrue(started[3])
        self.assertEqual(self.bot.ran, [("scheduled", "scheduled work")])
        self.assertEqual(self.bot._read_queue_file(), {})
        sessions_dir = os.path.join(self.tmp.name, ".cozter", "sessions")
        self.assertEqual(os.listdir(sessions_dir), [])

    async def test_schedule_starts_after_foreground_and_older_pending_work(self) -> None:
        active = asyncio.create_task(self.bot._dispatch_ai(
            self.bot.make_context("u1", "chat"), "active",
        ))
        try:
            await asyncio.wait_for(self.bot.foreground_started.wait(), timeout=1)
            await self.bot._dispatch_ai(
                self.bot.make_context("u1", "chat"), "older pending work",
            )
            await self.bot._fire_schedule("u1", self.schedule, workspace_path=self.tmp.name)
            await asyncio.sleep(0)
            self.assertEqual(self.bot.ran, [("normal", "active")])
            self.assertFalse(any(
                text.startswith("⏰ Starting") for text, *_ in self.bot.notices
            ))
            self.assertEqual(len(self.bot._read_queue_file()["u1"]), 3)
            self.assertEqual(self.bot._message_queues["u1"].qsize(), 2)

            self.bot.foreground_release.set()
            await asyncio.wait_for(active, timeout=1)
            await self._finish_drains()
            self.assertEqual(self.bot.ran, [
                ("normal", "active"), ("normal", "older pending work"),
                ("scheduled", "scheduled work"),
            ])
            self.assertEqual(self.bot._read_queue_file(), {})
        finally:
            if not active.done():
                active.cancel()
                await asyncio.gather(active, return_exceptions=True)

    async def test_simultaneous_idle_dispatches_follow_the_committed_fifo_order(self) -> None:
        # Both observe an idle turn lock while the first durable admission is
        # waiting; execution ownership must still follow admission order.
        await self.bot._queue_file_lock.acquire()
        first = asyncio.create_task(self.bot._dispatch_ai(
            self.bot.make_context("u1", "chat"), "active",
        ))
        second = None
        try:
            await asyncio.sleep(0)
            self.assertTrue(self.bot._ensure_dispatch_admission_lock("u1").locked())
            second = asyncio.create_task(self.bot._dispatch_ai(
                self.bot.make_context("u1", "chat"), "second",
            ))
            await asyncio.sleep(0)
            self.assertFalse(self.bot._ensure_task_lock("u1").locked())
            self.bot._queue_file_lock.release()
            await asyncio.wait_for(self.bot.foreground_started.wait(), timeout=1)
            await asyncio.wait_for(second, timeout=1)
            self.assertEqual(self.bot.ran, [("normal", "active")])
            self.assertEqual([row["text"] for row in self.bot._read_queue_file()["u1"]],
                             ["active", "second"])
            self.bot.foreground_release.set()
            await asyncio.wait_for(first, timeout=1)
            await self._finish_drains()
            self.assertEqual(self.bot.ran, [("normal", "active"), ("normal", "second")])
            self.assertEqual(self.bot._read_queue_file(), {})
        finally:
            if self.bot._queue_file_lock.locked():
                self.bot._queue_file_lock.release()
            self.bot.foreground_release.set()
            for task in (first, second):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(*[task for task in (first, second) if task], return_exceptions=True)

    async def test_slow_queue_notice_does_not_prevent_job_start(self) -> None:
        self.bot.slow_notice = "⏰ Queued"
        with mock.patch("Cozter.backends_bot.base._STATUS_OPERATION_TIMEOUT_SEC", 0.01):
            await asyncio.wait_for(self.bot._fire_schedule("u1", self.schedule, workspace_path=self.tmp.name), timeout=1)
            await self._finish_drains()
        self.assertTrue(self.bot.notice_cancelled)
        self.assertEqual(self.bot.ran, [("scheduled", "scheduled work")])
        self.assertEqual(self.bot._read_queue_file(), {})

    async def test_slow_start_notice_does_not_prevent_agent_execution(self) -> None:
        self.bot.slow_notice = "⏰ Starting"
        with mock.patch("Cozter.backends_bot.base._STATUS_OPERATION_TIMEOUT_SEC", 0.01):
            await self.bot._fire_schedule("u1", self.schedule, workspace_path=self.tmp.name)
            await self._finish_drains()
        self.assertTrue(self.bot.notice_cancelled)
        self.assertEqual(self.bot.ran, [("scheduled", "scheduled work")])
        self.assertEqual(self.bot._read_queue_file(), {})

    async def test_cancelled_queue_notice_still_wakes_admitted_work(self) -> None:
        self.bot.slow_notice = "⏰ Queued"
        firing = asyncio.create_task(self.bot._fire_schedule("u1", self.schedule, workspace_path=self.tmp.name))
        await asyncio.wait_for(self.bot.notice_started.wait(), timeout=1)
        firing.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await firing
        await self._finish_drains()
        self.assertTrue(self.bot.notice_cancelled)
        self.assertEqual(self.bot.ran, [("scheduled", "scheduled work")])
        self.assertEqual(self.bot._read_queue_file(), {})

    async def test_stop_during_queue_notice_discards_work_without_starting(self) -> None:
        self.bot.slow_notice = "⏰ Queued"
        firing = asyncio.create_task(self.bot._fire_schedule("u1", self.schedule, workspace_path=self.tmp.name))
        await asyncio.wait_for(self.bot.notice_started.wait(), timeout=1)
        task_running, cancelled_work = await self.bot._cancel_user_work("u1")
        firing.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await firing
        self.assertFalse(task_running)
        self.assertTrue(cancelled_work)
        self.assertEqual(self.bot.drains, [])
        self.assertEqual(self.bot.ran, [])
        self.assertEqual(self.bot._read_queue_file(), {})
        self.assertTrue(self.bot._message_queues["u1"].empty())

    async def test_full_queue_defers_without_repeated_chat_notices(self) -> None:
        self.bot.max_queue_size = 1
        msg_queue = self.bot._ensure_message_queue("u1")
        msg_queue.put_nowait(("older work", "chat", "older-id", False))
        for _ in range(2):
            await asyncio.wait_for(self.bot._fire_schedule("u1", self.schedule, workspace_path=self.tmp.name), timeout=1)
        self.assertEqual(self.bot.notices, [])
        self.assertEqual(self.bot.drains, [])
        self.assertEqual(self.bot.ran, [])
        self.assertEqual(self.bot._read_queue_file(), {})
        self.assertEqual(msg_queue.qsize(), 1)

    async def test_active_occurrence_is_not_republished_by_repeated_admission_or_restore(self) -> None:
        occurrence = ScheduledOccurrence(self.tmp.name, "u1", "calendar", "2026-10-16T09:00:00")
        started = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def run(_uid, _chat, text, **_kwargs):
            calls.append(text)
            started.set()
            await release.wait()

        first = await self.bot._admit_queue_entry(
            "u1", "scheduled work", "chat", 0, ephemeral=True,
            workspace_path=self.tmp.name, occurrence=occurrence,
        )
        self.assertTrue(first.created)
        with mock.patch.object(self.bot, "_run_turn", side_effect=run):
            drain = asyncio.create_task(self.bot._drain_message_queue("u1"))
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                repeated = await asyncio.gather(*[
                    self.bot._admit_queue_entry(
                        "u1", "scheduled work", "chat", 0, ephemeral=True,
                        workspace_path=self.tmp.name, occurrence=occurrence,
                    ) for _ in range(2)
                ])
                self.assertTrue(all(not result.created for result in repeated))
                await self.bot.restore_queues()
                self.assertTrue(self.bot._message_queues["u1"].empty())
                release.set()
                await asyncio.wait_for(drain, timeout=1)
            finally:
                release.set()
                if not drain.done():
                    drain.cancel()
                    await asyncio.gather(drain, return_exceptions=True)
        await self._finish_drains()
        self.assertEqual(calls, ["scheduled work"])
        self.assertEqual(self.bot._read_queue_file(), {})
        self.assertTrue(self.bot._message_queues["u1"].empty())
        self.assertNotIn("u1", self.bot._paused_jobs)
        self.assertNotIn("u1", self.bot._active_queue_entries)
        consumed = await self.bot._admit_queue_entry(
            "u1", "scheduled work", "chat", 0, ephemeral=True,
            workspace_path=self.tmp.name, occurrence=occurrence,
        )
        self.assertTrue(consumed.already_consumed)
        self.assertIsNone(consumed.entry)

    async def test_ephemeral_write_rejects_unavailable_or_empty_origin(self) -> None:
        with mock.patch.object(self.bot, "_current_workspace_for_turn", new=mock.AsyncMock(return_value=None)):
            for origin in (None, "", "  ", False, 1):
                with self.subTest(origin=origin), self.assertRaisesRegex(ValueError, "originating workspace"):
                    await self.bot._admit_queue_entry(
                        "u1", "scheduled work", "chat", 0, ephemeral=True, workspace_path=origin,
                    )
        self.assertEqual(self.bot._read_queue_file(), {})

    async def test_direct_schedule_requires_source_without_current_workspace_fallback(self) -> None:
        with mock.patch.object(self.bot, "_current_workspace_for_turn", new=mock.AsyncMock()) as current:
            with self.assertRaises(TypeError):
                await self.bot._fire_schedule("u1", self.schedule)
            for origin in (None, "", "  "):
                with self.subTest(origin=origin), self.assertRaisesRegex(ValueError, "source workspace"):
                    await self.bot._fire_schedule("u1", self.schedule, workspace_path=origin)
        current.assert_not_awaited()
        self.assertEqual(self.bot._read_queue_file(), {})


if __name__ == "__main__":
    unittest.main()
