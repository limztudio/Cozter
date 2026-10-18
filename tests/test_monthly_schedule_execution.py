"""Monthly and yearly rules through real scheduler claims and queue admission."""

import asyncio
from datetime import datetime
import os
import tempfile
import threading
import unittest
from unittest import mock

from Cozter import schedules, workspace
from Cozter.tests.helpers import OfflineBot
from Cozter.backends_bot.base import QueueCapacityError
from Cozter.queue_store import ScheduledOccurrence


class MonthlyScheduleExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.object(workspace, "CONFIG_DIR", self.ws))
        self.enterContext(mock.patch.object(
            workspace, "WORKSPACE_STATE_PATH", os.path.join(self.ws, "workspaces.json"),
        ))
        self.enterContext(mock.patch.dict(schedules._SCHEDULES_CACHE, clear=True))
        self.bot = OfflineBot(["owner"])
        self.store = self.bot._queue_store()
        self.enterContext(mock.patch.object(self.bot, "_queue_store", return_value=self.store))
        self.enterContext(mock.patch.object(self.bot, "_start_queue_drain"))
        workspace.select_workspace("owner", self.ws, self.bot.platform_id)

    def _add(self, days: list[str | int], **changes: object) -> dict:
        record = {
            "id": "calendar", "days": days, "time": "09:00",
            "created": "2026-10-01T00:00:00", "command": "calendar work",
            "chat_id": "room", "platform_id": self.bot.platform_id,
        }
        record.update(changes)
        schedules.add_schedule(self.ws, "owner", record)
        return record

    async def _tick(self, now: datetime, bot: OfflineBot | None = None) -> None:
        with mock.patch("Cozter.backends_bot.base.datetime") as clock:
            clock.now.return_value = now
            await (bot or self.bot)._scheduler_tick()

    def _entries(self) -> list[dict]:
        return self.bot._read_queue_file().get("owner", [])

    def _saved(self) -> dict:
        return schedules.list_schedules(self.ws, "owner")[0]

    async def test_full_queue_keeps_slot_due_until_admission_succeeds(self) -> None:
        self._add([16])
        self.bot.max_queue_size = 1
        queue = self.bot._ensure_message_queue("owner")
        queue.put_nowait(("older work", "room", "older-id", False))
        with mock.patch.object(self.bot, "send_text", new=mock.AsyncMock()) as notice:
            await self._tick(datetime(2026, 10, 16, 9, 0))
            await self._tick(datetime(2026, 10, 16, 9, 1))
            notice.assert_not_awaited()
        self.assertNotIn("last_fired", self._saved())
        self.assertEqual(self._entries(), [])
        queue.get_nowait()
        await self._tick(datetime(2026, 10, 16, 9, 2))
        await self._tick(datetime(2026, 10, 16, 9, 3))
        self.assertEqual(len(self._entries()), 1)
        self.assertEqual(self._saved()["last_fired"], "2026-10-16T09:00:00")

    async def test_rejected_auth_and_cancelled_admission_do_not_consume_slot(self) -> None:
        self._add([16])
        with mock.patch.object(self.bot, "authorized", return_value=False):
            await self._tick(datetime(2026, 10, 16, 9, 0))
        self.assertNotIn("last_fired", self._saved())
        persist = self.bot._admit_queue_entry

        async def cancelled(*args, **kwargs):
            self.bot._cancel_generations["owner"] = 1
            return await persist(*args, **kwargs)

        with mock.patch.object(self.bot, "_admit_queue_entry", side_effect=cancelled):
            await self._tick(datetime(2026, 10, 16, 9, 1))
        self.assertNotIn("last_fired", self._saved())
        self.assertEqual(self._entries(), [])
        await self._tick(datetime(2026, 10, 16, 9, 2))
        self.assertEqual(len(self._entries()), 1)

    async def test_failed_queue_write_does_not_advance_schedule(self) -> None:
        self._add([16])
        with mock.patch.object(self.store, "admit", side_effect=OSError("disk unavailable")):
            await self._tick(datetime(2026, 10, 16, 9, 0))
        self.assertNotIn("last_fired", self._saved())
        self.assertEqual(self._entries(), [])
        await self._tick(datetime(2026, 10, 16, 9, 1))
        self.assertEqual(len(self._entries()), 1)

    async def test_queue_receipt_recovers_failed_claim_without_duplicate_entry_or_notice(self) -> None:
        self._add([16])
        with mock.patch.object(schedules, "update_schedule_fired", side_effect=OSError("claim failed")):
            await self._tick(datetime(2026, 10, 16, 9, 0))
        self.assertNotIn("last_fired", self._saved())
        first = self._entries()[0]
        self.assertEqual(first["job"]["schedule_id"], "calendar")
        self.assertEqual(first["job"]["schedule_slot"], "2026-10-16T09:00:00")
        self.assertEqual(self.bot._message_queues["owner"].qsize(), 1)
        with mock.patch.object(self.bot, "send_text", new=mock.AsyncMock()) as notice:
            await self._tick(datetime(2026, 10, 16, 9, 1))
            await self._tick(datetime(2026, 10, 16, 9, 2))
            notice.assert_not_awaited()
        self.assertEqual([entry["id"] for entry in self._entries()], [first["id"]])
        self.assertEqual(self.bot._message_queues["owner"].qsize(), 1)
        self.assertEqual(self._saved()["last_fired"], "2026-10-16T09:00:00")

    async def test_restart_runs_work_even_while_schedule_projection_is_unavailable(self) -> None:
        self._add([16])
        with mock.patch.object(schedules, "update_schedule_fired", side_effect=OSError("read-only definitions")):
            await self._tick(datetime(2026, 10, 16, 9, 0))
            entry_id = self._entries()[0]["id"]
            restarted = OfflineBot(["owner"])
            starts = []

            def start(uid):
                starts.append(uid)
                self.assertNotIn("last_fired", self._saved())

            with mock.patch.object(restarted, "_start_queue_drain", side_effect=start):
                await restarted.restore_queues()
                await self._tick(datetime(2026, 10, 16, 9, 1), restarted)
            self.assertEqual(starts, ["owner"])
            self.assertEqual([entry["id"] for entry in self._entries()], [entry_id])
            self.assertEqual(restarted._message_queues["owner"].qsize(), 1)
        await self._tick(datetime(2026, 10, 16, 9, 2), restarted)
        self.assertEqual(self._saved()["last_fired"], "2026-10-16T09:00:00")

    async def test_completed_occurrence_cannot_reappear_after_failed_projection(self) -> None:
        self._add([16])
        with mock.patch.object(schedules, "update_schedule_fired", side_effect=OSError("read-only definitions")):
            await self._tick(datetime(2026, 10, 16, 9, 0))
            entry_id = self._entries()[0]["id"]
            self.bot._message_queues["owner"].get_nowait()
            await self.bot._persist_complete("owner", entry_id)
            restarted = OfflineBot(["owner"])
            with mock.patch.object(restarted, "_start_queue_drain") as wake:
                await restarted.restore_queues()
                await self._tick(datetime(2026, 10, 16, 9, 1), restarted)
                wake.assert_not_called()
        self.assertEqual(self._entries(), [])
        await self._tick(datetime(2026, 10, 16, 9, 2), restarted)
        self.assertEqual(self._saved()["last_fired"], "2026-10-16T09:00:00")
        self.assertEqual(self._entries(), [])

    async def test_failed_projection_does_not_starve_other_schedule_owners(self) -> None:
        self._add([16])
        workspace.select_workspace("other", self.ws, self.bot.platform_id)
        schedules.add_schedule(self.ws, "other", {
            "id": "other-calendar", "days": [16], "time": "09:00",
            "created": "2026-10-01T00:00:00", "command": "other work",
            "chat_id": "other room", "platform_id": self.bot.platform_id,
        })
        self.bot.notify_targets.append("other")
        update = schedules.update_schedule_fired

        def project(ws, uid, schedule_id, slot):
            if uid == "owner":
                raise OSError("owner projection unavailable")
            return update(ws, uid, schedule_id, slot)

        with mock.patch.object(schedules, "update_schedule_fired", side_effect=project):
            await self._tick(datetime(2026, 10, 16, 9, 0))
        rows = self.bot._read_queue_file()
        self.assertEqual(len(rows["owner"]), 1)
        self.assertEqual(len(rows["other"]), 1)
        self.assertEqual(self.bot._message_queues["owner"].qsize(), 1)
        self.assertEqual(self.bot._message_queues["other"].qsize(), 1)

    async def test_queue_ack_runs_after_workspace_lock_is_released_and_claim_commits(self) -> None:
        self._add([16])
        lock = workspace.get_lock(self.ws)

        async def notice(_chat, _text, **_kwargs):
            self.assertFalse(lock.locked())
            self.assertEqual(self._saved()["last_fired"], "2026-10-16T09:00:00")
            self.assertEqual(len(self._entries()), 1)

        with mock.patch.object(self.bot, "send_text", side_effect=notice) as send:
            await self._tick(datetime(2026, 10, 16, 9, 0))
        send.assert_awaited_once()

    async def test_source_schedule_workspace_is_pinned_when_current_workspace_changes(self) -> None:
        self._add([16])
        other = self.enterContext(tempfile.TemporaryDirectory())
        with mock.patch.object(self.bot, "_current_workspace_for_turn", new=mock.AsyncMock(return_value=other)) as current:
            await self._tick(datetime(2026, 10, 16, 9, 0))
        current.assert_not_awaited()
        self.assertEqual(self._entries()[0]["job"]["workspace_path"], workspace.canonicalize_workspace_path(self.ws))

    async def test_scheduled_and_ordinary_admission_share_one_capacity_barrier(self) -> None:
        self._add([16])
        self.bot.max_queue_size = 1
        entered = asyncio.Event()
        release = threading.Event()
        admit = self.store.admit
        loop = asyncio.get_running_loop()

        def blocked(*args, **kwargs):
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(2):
                raise RuntimeError("Test did not release transaction")
            return admit(*args, **kwargs)

        with mock.patch.object(self.store, "admit", side_effect=blocked):
            scheduled = asyncio.create_task(self._tick(datetime(2026, 10, 16, 9, 0)))
            ordinary = None
            try:
                await asyncio.wait_for(entered.wait(), timeout=1)
                ordinary = asyncio.create_task(self.bot._admit_queue_entry(
                    "owner", "ordinary work", "room", 0,
                ))
                await asyncio.sleep(0)
                self.assertFalse(ordinary.done())
                release.set()
                await asyncio.wait_for(scheduled, timeout=1)
                with self.assertRaises(QueueCapacityError):
                    await asyncio.wait_for(ordinary, timeout=1)
            finally:
                release.set()
                await asyncio.gather(*[task for task in (scheduled, ordinary) if task], return_exceptions=True)
        self.assertEqual(len(self._entries()), 1)
        self.assertEqual(self.bot._message_queues["owner"].qsize(), 1)
        self.assertEqual(self.bot._message_queues["owner"].maxsize, 0)
        self.assertFalse(workspace.get_lock(self.ws).locked())

    async def test_cancellation_during_owned_commit_publishes_and_wakes_after_claim(self) -> None:
        self._add([16])
        started = asyncio.Event()
        release = threading.Event()
        write = self.store.admit
        loop = asyncio.get_running_loop()

        def slow_write(*args, **kwargs):
            loop.call_soon_threadsafe(started.set)
            if not release.wait(2):
                raise RuntimeError("Test did not release queue commit")
            return write(*args, **kwargs)

        with (
            mock.patch.object(self.store, "admit", side_effect=slow_write),
            mock.patch.object(self.bot, "_start_queue_drain") as wake,
        ):
            tick = asyncio.create_task(self._tick(datetime(2026, 10, 16, 9, 0)))
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                tick.cancel()
                await asyncio.sleep(0)
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(tick, timeout=1)
            finally:
                release.set()
                if not tick.done():
                    tick.cancel()
                    await asyncio.gather(tick, return_exceptions=True)
            wake.assert_called_once_with("owner")
        self.assertEqual(len(self._entries()), 1)
        self.assertEqual(self.bot._message_queues["owner"].qsize(), 1)
        occurrence = ScheduledOccurrence(self.ws, "owner", "calendar", "2026-10-16T09:00:00")
        self.assertEqual(self.store.schedule_watermarks()[occurrence.key()], occurrence.slot)
        self.assertFalse(workspace.get_lock(self.ws).locked())

    async def test_cancellation_before_announcement_entry_still_wakes_durable_work(self) -> None:
        self._add([16])
        with (
            mock.patch.object(self.bot, "_announce_schedule", side_effect=asyncio.CancelledError),
            mock.patch.object(self.bot, "_start_queue_drain") as wake,
        ):
            with self.assertRaises(asyncio.CancelledError):
                await self._tick(datetime(2026, 10, 16, 9, 0))
            wake.assert_called_once_with("owner")
        self.assertEqual(len(self._entries()), 1)
        self.assertEqual(self.bot._message_queues["owner"].qsize(), 1)
        self.assertEqual(self._saved()["last_fired"], "2026-10-16T09:00:00")

    async def test_number_runs_on_monthly_date_and_keeps_origin_in_durable_queue(self) -> None:
        self._add([1])

        await self._tick(datetime(2026, 10, 1, 8, 59))
        self.assertEqual(self._entries(), [])
        await self._tick(datetime(2026, 10, 1, 9, 0))
        await self._tick(datetime(2026, 10, 1, 23, 59))
        # Numeric 1 previously meant Monday; it must not run on this Monday.
        await self._tick(datetime(2026, 10, 5, 9, 0))
        self.assertEqual(len(self._entries()), 1)
        self.assertEqual(self._saved()["last_fired"], "2026-10-01T09:00:00")

        await self._tick(datetime(2026, 11, 1, 9, 0))
        entries = self._entries()
        self.assertEqual(len(entries), 2)
        self.assertEqual(len({entry["id"] for entry in entries}), 2)
        for entry in entries:
            self.assertTrue(entry["ephemeral"])
            self.assertEqual(entry["text"], "calendar work")
            self.assertEqual(entry["chat_id"], "room")
            self.assertEqual(
                entry["job"]["workspace_path"], workspace.canonicalize_workspace_path(self.ws),
            )
        self.assertEqual(self._saved()["last_fired"], "2026-11-01T09:00:00")

    async def test_31st_skips_short_month_without_moving_to_last_day(self) -> None:
        self._add(
            [31], created="2026-01-01T00:00:00", last_fired="2026-01-31T09:00:00",
        )
        for now in (datetime(2026, 2, 28, 23, 59), datetime(2026, 3, 1, 9, 0)):
            await self._tick(now)
        self.assertEqual(self._entries(), [])
        self.assertEqual(self._saved()["last_fired"], "2026-01-31T09:00:00")

        await self._tick(datetime(2026, 3, 31, 9, 0))
        self.assertEqual(len(self._entries()), 1)
        self.assertEqual(self._saved()["last_fired"], "2026-03-31T09:00:00")

    async def test_missing_month_catches_up_latest_real_slot_once(self) -> None:
        self._add([31], created="2026-01-01T00:00:00")

        await self._tick(datetime(2026, 2, 15, 9, 0))
        self.assertEqual(self._saved()["last_fired"], "2026-01-31T09:00:00")
        await self._tick(datetime(2026, 3, 1, 9, 0))
        self.assertEqual(len(self._entries()), 1)

        await self._tick(datetime(2026, 4, 30, 23, 59))
        self.assertEqual(len(self._entries()), 2)
        self.assertEqual(self._saved()["last_fired"], "2026-03-31T09:00:00")

    async def test_leap_day_is_due_but_nonleap_february_does_not_invent_it(self) -> None:
        self._add([29], created="2028-02-01T00:00:00")
        await self._tick(datetime(2028, 2, 29, 8, 59))
        self.assertEqual(self._entries(), [])
        await self._tick(datetime(2028, 2, 29, 9, 0))
        self.assertEqual(len(self._entries()), 1)
        self.assertEqual(self._saved()["last_fired"], "2028-02-29T09:00:00")

        current = schedules._load_all(self.ws)
        current["owner"][0].update(created="2029-02-01T00:00:00")
        current["owner"][0].pop("last_fired")
        schedules._save_all(self.ws, current)
        await self._tick(datetime(2029, 2, 28, 23, 59))
        await self._tick(datetime(2029, 3, 1, 9, 0))
        self.assertEqual(len(self._entries()), 1)
        self.assertEqual(self._saved()["last_fired"], "2028-02-29T09:00:00")
        await self._tick(datetime(2029, 3, 29, 9, 0))
        self.assertEqual(len(self._entries()), 2)
        self.assertEqual(self._saved()["last_fired"], "2029-03-29T09:00:00")

    async def test_mixed_rules_run_union_and_overlap_is_claimed_once(self) -> None:
        self._add(["mon", 15], created="2027-03-09T00:00:00")
        # March 15 is both Monday and the monthly date.
        await self._tick(datetime(2027, 3, 15, 9, 0))
        await self._tick(datetime(2027, 3, 15, 18, 0))
        self.assertEqual(len(self._entries()), 1)
        await self._tick(datetime(2027, 3, 22, 9, 0))
        self.assertEqual(len(self._entries()), 2)
        # April 15 is Thursday, so this run belongs only to the monthly rule.
        await self._tick(datetime(2027, 4, 15, 9, 0))
        self.assertEqual(len(self._entries()), 3)
        self.assertEqual(self._saved()["last_fired"], "2027-04-15T09:00:00")

    async def test_creation_at_due_slot_prevents_retroactive_fire(self) -> None:
        self._add([16], created="2026-10-16T09:00:00")
        await self._tick(datetime(2026, 10, 16, 18, 0))
        self.assertEqual(self._entries(), [])
        self.assertNotIn("last_fired", self._saved())
        await self._tick(datetime(2026, 11, 16, 9, 0))
        self.assertEqual(len(self._entries()), 1)

    async def test_persisted_monthly_claim_survives_bot_restart(self) -> None:
        self._add([16])
        await self._tick(datetime(2026, 10, 16, 9, 0))
        restarted = OfflineBot(["owner"])
        with mock.patch.object(restarted, "_start_queue_drain"):
            await self._tick(datetime(2026, 10, 16, 23, 59), restarted)
            await self._tick(datetime(2026, 10, 17, 9, 0), restarted)
        self.assertEqual(len(self._entries()), 1)
        self.assertEqual(self._saved()["last_fired"], "2026-10-16T09:00:00")

    async def test_monthly_days_are_rechecked_after_waiting_for_workspace_lock(self) -> None:
        for original_days, updated_days, expected_slot in (
            (["fri"], [17], None),
            ([16], [17], None),
            (["fri"], [16], "2026-10-16T09:00:00"),
        ):
            with self.subTest(original_days=original_days, updated_days=updated_days):
                schedules._save_all(self.ws, {})
                self._add(original_days, created="2026-10-15T00:00:00")
                lock = asyncio.Lock()
                await lock.acquire()
                with (
                    mock.patch.object(workspace, "get_lock", return_value=lock),
                    mock.patch.object(self.bot, "_admit_schedule", new=mock.AsyncMock(return_value=None)) as fire,
                ):
                    task = asyncio.create_task(self._tick(datetime(2026, 10, 16, 9, 0)))
                    try:
                        await asyncio.sleep(0)
                        self.assertFalse(task.done())
                        current = schedules._load_all(self.ws)
                        current["owner"][0].update(days=updated_days, command="edited work")
                        schedules._save_all(self.ws, current)
                        lock.release()
                        await asyncio.wait_for(task, timeout=1)
                    finally:
                        if lock.locked():
                            lock.release()
                        if not task.done():
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                    if expected_slot is None:
                        fire.assert_not_awaited()
                        self.assertNotIn("last_fired", self._saved())
                    else:
                        fire.assert_awaited_once()
                        self.assertEqual(fire.await_args.kwargs["schedule_slot"], expected_slot)
                        self.assertEqual(fire.await_args.args[1]["command"], "edited work")

    async def test_concurrent_ticks_admit_overlapping_monthly_slot_only_once(self) -> None:
        self._add(["fri", 16])
        lock = asyncio.Lock()
        await lock.acquire()
        with (
            mock.patch.object(workspace, "get_lock", return_value=lock),
            mock.patch("Cozter.backends_bot.base.datetime") as clock,
        ):
            clock.now.return_value = datetime(2026, 10, 16, 9, 0)
            tasks = [asyncio.create_task(self.bot._scheduler_tick()) for _ in range(2)]
            try:
                await asyncio.sleep(0)
                self.assertTrue(all(not task.done() for task in tasks))
                lock.release()
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
            finally:
                if lock.locked():
                    lock.release()
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(len(self._entries()), 1)
        self.assertEqual(self.bot._message_queues["owner"].qsize(), 1)
        self.assertEqual(self._saved()["last_fired"], "2026-10-16T09:00:00")

    async def test_monthly_schedules_still_require_matching_platform_ownership(self) -> None:
        for schedule_id, platform_id in (
            ("own", self.bot.platform_id), ("foreign", "different:bot"),
            ("empty", ""), ("unscoped", None),
        ):
            record = self._add([16], id=schedule_id, platform_id=platform_id)
            if platform_id is None:
                current = schedules._load_all(self.ws)
                current["owner"][-1].pop("platform_id")
                schedules._save_all(self.ws, current)
            self.assertEqual(record["id"], schedule_id)
        await self._tick(datetime(2026, 10, 16, 9, 0))
        self.assertEqual(len(self._entries()), 1)
        claimed = {
            record["id"] for record in schedules.list_schedules(self.ws, "owner")
            if "last_fired" in record
        }
        self.assertEqual(claimed, {"own"})

    async def test_yearly_date_runs_at_time_and_repeats_only_next_year(self) -> None:
        self._add(["10/16"])
        await self._tick(datetime(2026, 10, 16, 8, 59))
        self.assertEqual(self._entries(), [])
        await self._tick(datetime(2026, 10, 16, 9, 0))
        await self._tick(datetime(2026, 10, 16, 23, 59))
        await self._tick(datetime(2026, 11, 16, 9, 0))
        await self._tick(datetime(2027, 1, 1, 9, 0))
        await self._tick(datetime(2027, 10, 16, 8, 59))
        self.assertEqual(len(self._entries()), 1)
        self.assertEqual(self._saved()["last_fired"], "2026-10-16T09:00:00")

        await self._tick(datetime(2027, 10, 16, 9, 0))
        self.assertEqual(len(self._entries()), 2)
        self.assertEqual(self._saved()["last_fired"], "2027-10-16T09:00:00")

    async def test_yearly_creation_guard_does_not_run_at_or_before_creation(self) -> None:
        self._add(["10/16"], created="2026-10-16T09:00:00")
        await self._tick(datetime(2026, 10, 16, 18, 0))
        await self._tick(datetime(2027, 10, 16, 8, 59))
        self.assertEqual(self._entries(), [])
        self.assertNotIn("last_fired", self._saved())
        await self._tick(datetime(2027, 10, 16, 9, 0))
        self.assertEqual(len(self._entries()), 1)

    async def test_yearly_catchup_claims_latest_year_once_even_after_restart(self) -> None:
        self._add(["10/16"], created="2020-01-01T00:00:00")
        await self._tick(datetime(2026, 12, 1, 9, 0))
        self.assertEqual(len(self._entries()), 1)
        self.assertEqual(self._saved()["last_fired"], "2026-10-16T09:00:00")
        restarted = OfflineBot(["owner"])
        with mock.patch.object(restarted, "_start_queue_drain"):
            await self._tick(datetime(2026, 12, 2, 9, 0), restarted)
            await self._tick(datetime(2027, 10, 16, 8, 59), restarted)
            self.assertEqual(len(self._entries()), 1)
            await self._tick(datetime(2027, 12, 1, 9, 0), restarted)
        self.assertEqual(len(self._entries()), 2)
        self.assertEqual(self._saved()["last_fired"], "2027-10-16T09:00:00")

    async def test_yearly_leap_day_skips_century_gap_and_catches_next_real_slot(self) -> None:
        self._add(["02/29"], created="2096-02-01T00:00:00")
        await self._tick(datetime(2096, 2, 29, 8, 59))
        self.assertEqual(self._entries(), [])
        await self._tick(datetime(2096, 2, 29, 9, 0))
        self.assertEqual(self._saved()["last_fired"], "2096-02-29T09:00:00")
        # Neither other months' 29ths nor nonleap February create a slot.
        for now in (
            datetime(2096, 3, 29, 9, 0), datetime(2097, 3, 1, 9, 0),
            datetime(2100, 3, 1, 9, 0), datetime(2103, 3, 1, 9, 0),
        ):
            await self._tick(now)
        self.assertEqual(len(self._entries()), 1)
        self.assertEqual(self._saved()["last_fired"], "2096-02-29T09:00:00")

        # A restart after the next leap day catches up that one valid slot.
        restarted = OfflineBot(["owner"])
        with mock.patch.object(restarted, "_start_queue_drain"):
            await self._tick(datetime(2104, 3, 1, 9, 0), restarted)
            await self._tick(datetime(2104, 3, 2, 9, 0), restarted)
        self.assertEqual(len(self._entries()), 2)
        self.assertEqual(self._saved()["last_fired"], "2104-02-29T09:00:00")

    async def test_weekly_monthly_and_yearly_overlap_admits_one_job(self) -> None:
        self._add(["fri", 16, "10/16"])
        # October 16 is Friday and matches both calendar rules.
        await self._tick(datetime(2026, 10, 16, 9, 0))
        await self._tick(datetime(2026, 10, 16, 23, 59))
        self.assertEqual(len(self._entries()), 1)
        self.assertEqual(self.bot._message_queues["owner"].qsize(), 1)
        self.assertEqual(self._saved()["last_fired"], "2026-10-16T09:00:00")
        await self._tick(datetime(2026, 10, 23, 9, 0))
        self.assertEqual(len(self._entries()), 2)
        # November 16 is Monday, so only the monthly rule matches.
        await self._tick(datetime(2026, 11, 16, 9, 0))
        self.assertEqual(len(self._entries()), 3)
        self.assertEqual(self._saved()["last_fired"], "2026-11-16T09:00:00")

    async def test_yearly_date_is_rechecked_after_waiting_for_workspace_lock(self) -> None:
        for updated_days, expected_slot in (
            (["10/17"], None),
            (["10/15"], "2026-10-15T09:00:00"),
        ):
            with self.subTest(updated_days=updated_days):
                schedules._save_all(self.ws, {})
                self._add(["10/16"], created="2026-10-01T00:00:00")
                lock = asyncio.Lock()
                await lock.acquire()
                with (
                    mock.patch.object(workspace, "get_lock", return_value=lock),
                    mock.patch.object(self.bot, "_admit_schedule", new=mock.AsyncMock(return_value=None)) as fire,
                ):
                    task = asyncio.create_task(self._tick(datetime(2026, 10, 16, 9, 0)))
                    try:
                        await asyncio.sleep(0)
                        self.assertFalse(task.done())
                        current = schedules._load_all(self.ws)
                        current["owner"][0].update(days=updated_days, command="edited yearly work")
                        schedules._save_all(self.ws, current)
                        lock.release()
                        await asyncio.wait_for(task, timeout=1)
                    finally:
                        if lock.locked():
                            lock.release()
                        if not task.done():
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                    if expected_slot is None:
                        fire.assert_not_awaited()
                        self.assertNotIn("last_fired", self._saved())
                    else:
                        fire.assert_awaited_once()
                        self.assertEqual(fire.await_args.kwargs["schedule_slot"], expected_slot)
                        self.assertEqual(fire.await_args.args[1]["command"], "edited yearly work")
