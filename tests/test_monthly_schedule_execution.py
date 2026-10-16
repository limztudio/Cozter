"""Monthly and yearly rules through real scheduler claims and queue admission."""

import asyncio
from datetime import datetime
import os
import tempfile
import unittest
from unittest import mock

from Cozter import schedules, workspace
from Cozter.tests.helpers import TestBot


class MonthlyScheduleExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.object(workspace, "CONFIG_DIR", self.ws))
        self.enterContext(mock.patch.object(
            workspace, "WORKSPACE_STATE_PATH", os.path.join(self.ws, "workspaces.json"),
        ))
        self.enterContext(mock.patch.dict(schedules._SCHEDULES_CACHE, clear=True))
        self.bot = TestBot(["owner"])
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

    async def _tick(self, now: datetime, bot: TestBot | None = None) -> None:
        with mock.patch("Cozter.backends_bot.base.datetime") as clock:
            clock.now.return_value = now
            await (bot or self.bot)._scheduler_tick()

    def _entries(self) -> list[dict]:
        return self.bot._read_queue_file().get("owner", [])

    def _saved(self) -> dict:
        return schedules.list_schedules(self.ws, "owner")[0]

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
        self.assertNotIn("last_fired", self._saved())
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
        restarted = TestBot(["owner"])
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
                    mock.patch.object(self.bot, "_fire_schedule", new=mock.AsyncMock()) as fire,
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
                        self.assertEqual(fire.await_args.args[1]["last_fired"], expected_slot)
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
        restarted = TestBot(["owner"])
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
        restarted = TestBot(["owner"])
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
                    mock.patch.object(self.bot, "_fire_schedule", new=mock.AsyncMock()) as fire,
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
                        self.assertEqual(fire.await_args.args[1]["last_fired"], expected_slot)
                        self.assertEqual(fire.await_args.args[1]["command"], "edited yearly work")
