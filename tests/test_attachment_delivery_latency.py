"""Best-effort attachments cannot indefinitely hold a later completed reply."""

import asyncio
import tempfile
import unittest
from unittest import mock

from Cozter import workspace
from Cozter.backends_agent.base import AgentResult, ChatEvent
from Cozter.backends_bot import base
from Cozter.tests.helpers import persist_queue_entry, TestBot


class _UploadProbe(TestBot):
    def __init__(self):
        super().__init__(["u1"])
        self.sent = []
        self.files = []
        self.active = 0
        self.peak = 0
        self.started = asyncio.Event()
        self.three_started = asyncio.Event()
        self.release = asyncio.Event()

    async def send_text(self, _chat, text, *, rich=False):
        self.sent.append(text)

    async def send_file(self, _chat, path):
        self.files.append(path)
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started.set()
        if self.active == 3:
            self.three_started.set()
        try:
            await self.release.wait()
            await asyncio.sleep(0)
        finally:
            self.active -= 1


class AttachmentDeliveryLatencyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.object(workspace, "CONFIG_DIR", self.ws))
        self.bot = _UploadProbe()
        self.enterContext(mock.patch.object(self.bot, "start_detached_task_watcher"))

    async def asyncTearDown(self):
        for owner in list(self.bot._attachment_tasks):
            await self.bot._cancel_attachment_uploads(owner)
        await asyncio.sleep(0)

    def reply(self, text="Later completed answer"):
        return AgentResult(text=text, events=[ChatEvent(kind="text", content=text)])

    async def test_stuck_older_upload_times_out_without_cancelling_it_or_losing_text(self):
        tail = self.bot._track_attachment_upload("u1", "chat", ["old-proof.png"])
        await asyncio.wait_for(self.bot.started.wait(), 1)
        with (mock.patch.object(base, "_ATTACHMENT_TAIL_WAIT_SEC", .03),
              self.assertLogs("Cozter.performance", "INFO") as logs):
            await asyncio.wait_for(self.bot._send_result("chat", self.ws, self.reply(), uid="u1"), 1)
        self.assertEqual(self.bot.sent, ["Later completed answer"])
        self.assertFalse(tail.done())
        self.assertEqual(self.bot.active, 1)
        self.assertTrue(any("stage=attachment_tail_wait outcome=timeout" in line for line in logs.output))
        self.bot.release.set()
        await tail
        self.assertFalse(tail.cancelled())

    async def test_cancelling_later_text_wait_propagates_without_cancelling_earlier_upload(self):
        tail = self.bot._track_attachment_upload("u1", "chat", ["old.png"])
        await asyncio.wait_for(self.bot.started.wait(), 1)
        reply = asyncio.create_task(self.bot._send_result("chat", self.ws, self.reply(), uid="u1"))
        await asyncio.sleep(0)
        reply.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await reply
        self.assertFalse(tail.done())
        self.assertEqual(self.bot.sent, [])
        self.bot.release.set()
        await tail

    async def test_tail_timeout_completes_durable_text_delivery_without_agent_reexecution(self):
        entry_id = await persist_queue_entry(self.bot, "u1", "Completed request", "chat")
        tail = self.bot._track_attachment_upload("u1", "chat", ["older-proof.png"])
        await asyncio.wait_for(self.bot.started.wait(), 1)
        result = self.reply()
        record = await self.bot._stage_reply_delivery(
            "u1", entry_id, "chat", self.ws, result, allow_await=False,
        )
        with (mock.patch.object(base, "_ATTACHMENT_TAIL_WAIT_SEC", .01),
              mock.patch.object(base.agent, "run", mock.AsyncMock()) as run):
            delivered = await asyncio.wait_for(self.bot._deliver_staged_reply(
                record, ws=self.ws, result=result, awaiting_uid="u1",
            ), 1)
        self.assertTrue(delivered)
        self.assertEqual(self.bot.sent, ["Later completed answer"])
        self.assertEqual(await self.bot._list_reply_delivery_records(), [])
        self.assertEqual(self.bot._read_queue_file(), {})
        run.assert_not_awaited()
        self.assertFalse(tail.done())
        self.bot.release.set()
        await tail

    async def test_cancelled_older_upload_does_not_fail_the_later_reply(self):
        tail = self.bot._track_attachment_upload("u1", "chat", ["old.png"])
        await asyncio.wait_for(self.bot.started.wait(), 1)
        reply = asyncio.create_task(self.bot._send_result("chat", self.ws, self.reply(), uid="u1"))
        await asyncio.sleep(0)
        tail.cancel()
        await asyncio.gather(tail, return_exceptions=True)
        await asyncio.wait_for(reply, 1)
        self.assertEqual(self.bot.sent, ["Later completed answer"])

    async def test_thousand_explicit_files_use_three_workers_and_all_are_attempted(self):
        paths = [f"proof-{index}.png" for index in range(1000)]
        batch = self.bot._track_attachment_upload("u1", "chat", paths)
        await asyncio.wait_for(self.bot.three_started.wait(), 1)
        self.assertEqual(self.bot.peak, 3)
        self.assertEqual(self.bot.files, paths[:3])
        workers = [task for task in asyncio.all_tasks()
                   if task.get_name().startswith(f"{self.bot.platform_id}:upload-file:")]
        self.assertEqual(len(workers), 3)
        self.bot.release.set()
        await asyncio.wait_for(batch, 2)
        self.assertEqual(self.bot.files, paths)
        self.assertEqual(self.bot.peak, 3)
        self.assertEqual(self.bot.active, 0)

    async def test_text_timeout_does_not_reorder_attachment_batches(self):
        first = self.bot._track_attachment_upload("u1", "chat", ["A.png", "B.png"])
        await asyncio.wait_for(self.bot.started.wait(), 1)
        second = self.bot._track_attachment_upload("u1", "chat", ["C.png", "D.png"])
        with mock.patch.object(base, "_ATTACHMENT_TAIL_WAIT_SEC", .01):
            await self.bot._send_result("chat", self.ws, self.reply(), uid="u1")
        self.assertEqual(self.bot.files, ["A.png", "B.png"])
        self.assertEqual(self.bot.sent, ["Later completed answer"])
        self.bot.release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 1)
        self.assertEqual(self.bot.files, ["A.png", "B.png", "C.png", "D.png"])

    async def test_cancelling_later_batch_preserves_another_owners_earlier_batch(self):
        first = self.bot._track_attachment_upload("earlier", "chat", ["A.png"])
        await asyncio.wait_for(self.bot.started.wait(), 1)
        second = self.bot._track_attachment_upload("later", "chat", ["B.png"])
        await asyncio.sleep(0)
        self.assertEqual(await self.bot._cancel_attachment_uploads("later"), 1)
        self.assertTrue(second.cancelled())
        self.assertFalse(first.done())
        self.bot.release.set()
        await first
        self.assertEqual(self.bot.files, ["A.png"])

    async def test_cancelling_earlier_batch_does_not_discard_another_owners_later_batch(self):
        first = self.bot._track_attachment_upload("earlier", "chat", ["A.png"])
        await asyncio.wait_for(self.bot.started.wait(), 1)
        second = self.bot._track_attachment_upload("later", "chat", ["B.png"])
        await asyncio.sleep(0)
        self.bot.started.clear()
        self.assertEqual(await self.bot._cancel_attachment_uploads("earlier"), 1)
        await asyncio.wait_for(self.bot.started.wait(), 1)
        self.assertTrue(first.cancelled())
        self.assertFalse(second.done())
        self.assertEqual(self.bot.files, ["A.png", "B.png"])
        self.bot.release.set()
        await asyncio.wait_for(second, 1)
        self.assertFalse(second.cancelled())

    async def test_stop_cancels_only_started_pool_and_leaves_no_worker_or_upload_owner(self):
        batch = self.bot._track_attachment_upload("u1", "chat", [str(index) for index in range(1000)])
        await asyncio.wait_for(self.bot.three_started.wait(), 1)
        self.assertTrue(self.bot.has_active_turns())
        await asyncio.wait_for(self.bot._cancel_user_work("u1"), 1)
        self.assertTrue(batch.cancelled())
        self.assertEqual(len(self.bot.files), 3)
        self.assertEqual(self.bot.active, 0)
        self.assertFalse(self.bot.has_active_turns())
        self.assertEqual(self.bot._attachment_tasks, {})
        self.assertEqual(self.bot._attachment_tails, {})
        self.assertFalse(any(task.get_name().startswith(f"{self.bot.platform_id}:upload-file:")
                             for task in asyncio.all_tasks()))

    async def test_one_upload_failure_does_not_abandon_remaining_explicit_files(self):
        async def send(_chat, path):
            self.bot.files.append(path)
            if path == "broken.png":
                raise OSError("fixture upload failure")
            await asyncio.sleep(0)

        paths = ["broken.png", "valid-1.png", "valid-2.png", "valid-3.png"]
        with (mock.patch.object(self.bot, "send_file", send),
              self.assertLogs(base.logger, "WARNING")):
            batch = self.bot._track_attachment_upload("u1", "chat", paths)
            await batch
        self.assertEqual(self.bot.files, paths)
        self.assertTrue(any("Failed to attach broken.png" in text for text in self.bot.sent))

