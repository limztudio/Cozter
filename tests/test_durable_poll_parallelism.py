"""Offline proofs that a slow recovery owner cannot stall unrelated work."""

import asyncio
import unittest
from unittest import mock

from Cozter.backends_bot import base as bot_base
from Cozter.tests.helpers import OfflineBot


class DurablePollParallelismTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bot = OfflineBot([])

    async def test_stalled_reply_does_not_postpone_retry_or_provider_poll_cycles(self):
        reply_started = asyncio.Event()
        retry_twice = asyncio.Event()
        provider_twice = asyncio.Event()
        reply_cancelled = asyncio.Event()
        retry_count = provider_count = 0

        async def retry():
            nonlocal retry_count
            retry_count += 1
            if retry_count >= 2:
                retry_twice.set()

        async def reply():
            reply_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                reply_cancelled.set()

        async def provider():
            nonlocal provider_count
            provider_count += 1
            if provider_count >= 2:
                provider_twice.set()

        with (mock.patch.object(self.bot, "_check_job_retries", retry),
              mock.patch.object(self.bot, "_check_reply_deliveries", reply),
              mock.patch.object(self.bot, "_check_detached_tasks", provider),
              mock.patch.object(bot_base, "_DETACHED_TASK_POLL_INTERVAL_SEC", .001)):
            watcher = asyncio.create_task(self.bot._detached_task_loop())
            try:
                await asyncio.wait_for(reply_started.wait(), 1)
                await asyncio.wait_for(asyncio.gather(
                    retry_twice.wait(), provider_twice.wait(),
                ), 1)
            finally:
                watcher.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await watcher
        self.assertTrue(reply_cancelled.is_set())

    async def test_slow_reply_owner_does_not_hold_other_owners_and_failure_keeps_fifo(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        other_done = asyncio.Event()
        calls = []
        records = [{"id": key, "user_id": owner} for key, owner in (
            ("a1", "a"), ("a2", "a"), ("b1", "b"), ("b2", "b"),
        )]

        async def deliver(record):
            calls.append(record["id"])
            if record["id"] == "a1":
                entered.set()
                await release.wait()
                return False
            if record["id"] == "b2":
                other_done.set()
            return True

        with (mock.patch.object(self.bot, "_list_reply_delivery_records",
                                new=mock.AsyncMock(return_value=records)),
              mock.patch.object(self.bot, "_deliver_staged_reply", deliver)):
            polling = asyncio.create_task(self.bot._check_reply_deliveries())
            try:
                await asyncio.wait_for(entered.wait(), 1)
                await asyncio.wait_for(other_done.wait(), 1)
                self.assertEqual(calls, ["a1", "b1", "b2"])
                release.set()
                await asyncio.wait_for(polling, 1)
            finally:
                release.set()
                await polling
        self.assertNotIn("a2", calls)

    async def test_detached_polls_are_bounded_and_preserve_each_owner_order(self):
        release = asyncio.Event()
        saturated = asyncio.Event()
        active = peak = 0
        calls: dict[str, list[int]] = {}
        limit = bot_base._DURABLE_POLL_CONCURRENCY
        records = [{"id": f"{owner}:{index}", "user_id": str(owner), "index": index}
                   for index in range(2) for owner in range(limit + 3)]

        async def check(record):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            calls.setdefault(record["user_id"], []).append(record["index"])
            if active == limit:
                saturated.set()
            try:
                await release.wait()
            finally:
                active -= 1

        with (mock.patch.object(self.bot, "_list_detached_task_records",
                                new=mock.AsyncMock(return_value=records)),
              mock.patch.object(self.bot, "_check_detached_task", check)):
            polling = asyncio.create_task(self.bot._check_detached_tasks())
            try:
                await asyncio.wait_for(saturated.wait(), 1)
                self.assertEqual(len(calls), limit)
                release.set()
                await asyncio.wait_for(polling, 1)
            finally:
                release.set()
                await polling
        self.assertEqual(peak, limit)
        self.assertEqual(active, 0)
        self.assertEqual(calls, {str(owner): [0, 1] for owner in range(limit + 3)})

    async def test_recovery_cancellation_settles_all_owned_poll_workers(self):
        limit = bot_base._DURABLE_POLL_CONCURRENCY
        entered = asyncio.Event()
        active = 0
        cancelled = []

        async def check(record):
            nonlocal active
            active += 1
            if active == limit:
                entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(record["id"])
                active -= 1

        records = [{"id": str(index), "user_id": str(index)} for index in range(limit + 3)]
        polling = asyncio.create_task(self.bot._poll_records_by_owner(records, check, label="test"))
        await asyncio.wait_for(entered.wait(), 1)
        polling.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(polling, 1)
        self.assertEqual(active, 0)
        self.assertEqual(len(cancelled), limit)

    async def test_one_owner_failure_keeps_other_owners_running(self):
        records = [{"id": key, "user_id": owner} for key, owner in (
            ("a1", "a"), ("a2", "a"), ("b1", "b"),
        )]
        calls = []

        async def check(record):
            calls.append(record["id"])
            if record["id"] == "a1":
                raise OSError("one unavailable owner")

        with self.assertLogs(bot_base.logger, level="WARNING"):
            await self.bot._poll_records_by_owner(records, check, label="test")
        self.assertEqual(calls, ["a1", "b1"])

    async def test_same_owner_recovery_classes_do_not_interleave_reply_chunks(self):
        first_sent = asyncio.Event()
        release = asyncio.Event()
        sent = []
        reply = {"id": "reply", "user_id": "a", "chat_id": "chat",
                 "messages": ["first", "second"], "awaiting": False}
        detached = {"id": "detached", "user_id": "a", "chat_id": "chat",
                    "workspace_path": "unused", "session_id": None}
        self.bot._pending_reply_delivery_users.add("a")

        async def send(_chat, text, **_kwargs):
            sent.append(text)
            if text == "first":
                first_sent.set()
                await release.wait()

        async def remove_reply(_id):
            self.bot._pending_reply_delivery_users.discard("a")

        async def callback(_chat, _ws, _result, _uid):
            sent.append("callback")

        with (mock.patch.object(self.bot, "authorized", return_value=True),
              mock.patch.object(self.bot, "_get_reply_delivery_record", new=mock.AsyncMock(return_value=reply)),
              mock.patch.object(self.bot, "_complete_reply_input", new=mock.AsyncMock(return_value=True)),
              mock.patch.object(self.bot, "_remove_reply_delivery_record", remove_reply),
              mock.patch.object(self.bot, "_get_detached_task_record", new=mock.AsyncMock(return_value=detached)),
              mock.patch.object(self.bot, "_remove_detached_task_record", new=mock.AsyncMock()),
              mock.patch.object(self.bot, "_start_queue_drain"),
              mock.patch.object(self.bot, "send_text", send),
              mock.patch.object(self.bot, "_send_detached_result", callback),
              mock.patch.object(bot_base.agent, "log_detached_task_completion", new=mock.AsyncMock())):
            sending = asyncio.create_task(self.bot._deliver_current_staged_reply(reply))
            notifying = None
            try:
                await asyncio.wait_for(first_sent.wait(), 1)
                notifying = asyncio.create_task(self.bot._deliver_current_detached_task(detached, "done"))
                await asyncio.sleep(0)
                self.assertFalse(notifying.done())
                self.assertEqual(sent, ["first"])
                release.set()
                await asyncio.wait_for(asyncio.gather(sending, notifying), 1)
            finally:
                release.set()
                await asyncio.gather(*[task for task in (sending, notifying) if task], return_exceptions=True)
        self.assertEqual(sent, ["first", "second", "callback"])

    async def test_detached_notification_waits_for_owed_reply_without_clearing_ledger(self):
        record = {"id": "detached", "user_id": "a", "chat_id": "chat"}
        self.bot._pending_reply_delivery_users.add("a")
        with (mock.patch.object(self.bot, "_get_detached_task_record", new=mock.AsyncMock()) as read,
              mock.patch.object(self.bot, "_remove_detached_task_record", new=mock.AsyncMock()) as remove,
              mock.patch.object(self.bot, "send_text", new=mock.AsyncMock()) as send):
            await self.bot._deliver_current_detached_task(record, "done")
            await self.bot._report_live_blocked_detached_task(record, "needs input")
        read.assert_not_awaited()
        remove.assert_not_awaited()
        send.assert_not_awaited()

    async def test_next_owner_pass_waits_for_existing_provider_operation_deadline(self):
        """Owner waves share a pass boundary; recovery classes do not."""
        fast_first = asyncio.Event()
        fast_second = asyncio.Event()
        slow_deadline = asyncio.Event()
        fire_deadline = asyncio.Event()
        calls: dict[str, int] = {}
        records = [{"id": "slow", "user_id": "slow"}, {"id": "fast", "user_id": "fast"}]

        async def check(record):
            owner = record["user_id"]
            calls[owner] = calls.get(owner, 0) + 1
            if owner == "slow" and calls[owner] == 1:
                try:
                    # Model the existing bounded provider control operation.
                    await fire_deadline.wait()
                    raise TimeoutError
                except TimeoutError:
                    slow_deadline.set()
            elif owner == "fast":
                (fast_first if calls[owner] == 1 else fast_second).set()

        with (mock.patch.object(self.bot, "_list_detached_task_records",
                                new=mock.AsyncMock(return_value=records)),
              mock.patch.object(self.bot, "_check_detached_task", check),
              mock.patch.object(self.bot, "_check_job_retries", new=mock.AsyncMock()),
              mock.patch.object(self.bot, "_check_reply_deliveries", new=mock.AsyncMock()),
              mock.patch.object(bot_base, "_DETACHED_TASK_POLL_INTERVAL_SEC", .001)):
            watcher = asyncio.create_task(self.bot._detached_task_loop())
            try:
                await asyncio.wait_for(fast_first.wait(), 1)
                self.assertFalse(fast_second.is_set())
                self.assertFalse(slow_deadline.is_set())
                fire_deadline.set()
                await asyncio.wait_for(fast_second.wait(), 1)
                self.assertTrue(slow_deadline.is_set())
            finally:
                fire_deadline.set()
                watcher.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await watcher

    async def test_repeated_shutdown_cancel_keeps_provider_cleanup_owned(self):
        limit = bot_base._DURABLE_POLL_CONCURRENCY
        entered = asyncio.Event()
        cleanup_entered = asyncio.Event()
        cleanup_release = asyncio.Event()
        active = 0
        cleaning = 0
        completed = []

        async def check(record):
            nonlocal active, cleaning
            active += 1
            if active == limit:
                entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning += 1
                if cleaning == limit:
                    cleanup_entered.set()
                await cleanup_release.wait()
                completed.append(record["id"])

        records = [{"id": str(index), "user_id": str(index)} for index in range(limit)]
        polling = asyncio.create_task(self.bot._poll_records_by_owner(records, check, label="test"))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            polling.cancel()
            await asyncio.wait_for(cleanup_entered.wait(), 1)
            polling.cancel()
            await asyncio.sleep(0)
            self.assertFalse(polling.done())
            cleanup_release.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(polling, 1)
        finally:
            cleanup_release.set()
            await asyncio.gather(polling, return_exceptions=True)
        self.assertEqual(len(completed), limit)


if __name__ == "__main__":
    unittest.main()
