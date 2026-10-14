"""Prerequisite recovery waits durably, retries automatically, and keeps FIFO."""

import asyncio
import os
import tempfile
import threading
import unittest
from unittest import mock

from Cozter import session, workspace
from Cozter.backends_agent.base import AgentResult, ChatEvent
from Cozter.backends_bot import base
from Cozter.tests.test_reply_delivery import _ReplyDeliveryBot


class JobPrerequisiteRetryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.object(workspace, "CONFIG_DIR", self.ws))
        self.bot = _ReplyDeliveryBot(self.ws, fail_final=False)
        self.sid = session.create_session(self.ws, name="Preserved job")["id"]
        self.original = "Repair every requested target, verify it, and commit the result"

    async def enqueue(self, text, *, bot=None):
        bot = bot or self.bot
        entry_id = await bot._persist_enqueue("u1", text, "chat")
        bot._ensure_message_queue("u1").put_nowait((text, "chat", entry_id, False))
        return entry_id

    def blocked(self, number=1):
        return AgentResult(
            text="Verified source repairs are saved; desktop checks remain unfinished",
            session_id=self.sid, completion_verified=False,
            continuation_reason="prerequisite",
            blocked_reason=f"A visible desktop is unavailable (probe {number})",
            continue_instruction="Recover the desktop using authorized tools, then finish remaining checks",
            continuation_checkpoint={
                "pending": ["Exercise the requested visible desktop"],
                "reports": [{"instruction": "Repair sources", "report": "Repairs and tests passed"}],
            },
            events=[ChatEvent(kind="attachment", content=os.path.join(self.ws, "proof.txt"))],
            usage={"input_tokens": 7},
        )

    async def waiting(self, *, retry_at=1000, bot=None):
        bot = bot or self.bot
        entry_id = await self.enqueue(self.original, bot=bot)
        await bot._save_job_checkpoint("u1", entry_id, {
            "state": "paused", "request": self.original,
            "workspace_path": self.ws, "session_id": self.sid,
            "next_retry_at": retry_at, "retry_count": 1,
            "retry_reason": "Visible desktop unavailable",
        })
        bot._paused_jobs.add("u1")
        return entry_id

    async def test_prerequisite_retains_work_and_queue_without_an_approval_prompt(self):
        entry = await self.enqueue(self.original)
        later = await self.enqueue("Later independent maintenance")
        result = self.blocked()
        with (mock.patch.object(self.bot, "_run_turn_with_budget", mock.AsyncMock(return_value=result)) as run,
              mock.patch.object(base.time, "time", return_value=1000)):
            await self.bot._drain_message_queue("u1")
        run.assert_awaited_once()
        saved = await self.bot._load_job_checkpoint("u1", entry)
        self.assertEqual(saved["state"], "paused")
        self.assertEqual(saved["next_retry_at"], 1030)
        self.assertEqual(saved["retry_count"], 1)
        self.assertEqual(saved["request"], self.original)
        self.assertEqual(saved["session_id"], self.sid)
        self.assertIn("Repairs and tests passed", str(saved["checkpoint"]))
        self.assertEqual(saved["output"]["usage"], {"input_tokens": 7})
        self.assertEqual(saved["output"]["attachments"], [os.path.join(self.ws, "proof.txt")])
        self.assertFalse(result.completion_verified)
        self.assertTrue(self.bot._job_result_is_incomplete(result))
        self.assertFalse(self.bot.has_active_turns())
        self.assertEqual([row["id"] for row in self.bot._read_queue_file()["u1"]], [entry, later])
        self.assertIn("automatically", self.bot.sent[-1])
        self.assertNotIn("guidance", self.bot.sent[-1])
        self.assertNotIn("permission", self.bot.sent[-1])
        self.assertNotIn("approval", self.bot.sent[-1])

    async def test_backoff_is_bounded_and_does_not_reset_when_probe_numbers_change(self):
        entry = await self.enqueue(self.original)
        delays = []
        for number in range(1, 8):
            job = await self.bot._load_job_checkpoint("u1", entry)
            job["state"] = "running"
            await self.bot._save_job_checkpoint("u1", entry, job)
            self.bot._paused_jobs.discard("u1")
            with (mock.patch.object(self.bot, "_run_turn_with_budget", mock.AsyncMock(return_value=self.blocked(number))),
                  mock.patch.object(base.time, "time", return_value=1000)):
                self.assertFalse(await self.bot._run_turn("u1", "chat", self.original, queue_entry_id=entry))
            saved = await self.bot._load_job_checkpoint("u1", entry)
            delays.append(saved["next_retry_at"] - 1000)
            self.assertEqual(saved["retry_count"], number)
        self.assertEqual(delays, [30, 60, 120, 240, 300, 300, 300])

    async def test_future_retry_survives_restart_and_due_head_wakes_once(self):
        entry = await self.waiting(retry_at=1200)
        later = await self.enqueue("Later work")
        restarted = _ReplyDeliveryBot(self.ws, fail_final=False)
        with mock.patch.object(restarted, "_drain_message_queue", mock.AsyncMock()) as drain:
            await restarted.restore_queues()
            await asyncio.sleep(0)
            drain.assert_not_awaited()
        with (mock.patch.object(base.time, "time", return_value=1199),
              mock.patch.object(restarted, "_start_queue_drain") as wake):
            await restarted._check_job_retries()
            wake.assert_not_called()
        with (mock.patch.object(base.time, "time", return_value=1200),
              mock.patch.object(restarted, "_start_queue_drain") as wake):
            await asyncio.gather(restarted._check_job_retries(), restarted._check_job_retries())
            wake.assert_called_once_with("u1")
        saved = await restarted._load_job_checkpoint("u1", entry)
        self.assertEqual(saved["state"], "running")
        self.assertNotIn("next_retry_at", saved)
        self.assertEqual(saved["retry_count"], 1)
        self.assertEqual(saved["session_id"], self.sid)
        self.assertNotIn("u1", restarted._paused_jobs)
        self.assertEqual([row[2] for row in restarted._message_queues["u1"]._queue], [entry, later])

    async def test_due_retry_finishes_original_before_next_queued_job(self):
        entry = await self.waiting(retry_at=1000)
        await self.enqueue("Later work")
        with (mock.patch.object(base.time, "time", return_value=1000),
              mock.patch.object(self.bot, "_start_queue_drain")):
            await self.bot._check_job_retries()
        calls = []

        async def complete(_uid, _chat, prompt, **kwargs):
            calls.append((prompt, kwargs.get("session_id"), kwargs.get("queue_entry_id")))
            return None  # The completed result has already been delivered.

        with mock.patch.object(self.bot, "_run_turn_with_budget", complete):
            await self.bot._drain_message_queue("u1")
        self.assertEqual(calls[0], (self.original, self.sid, entry))
        self.assertEqual(calls[1][0], "Later work")
        self.assertEqual(self.bot._read_queue_file(), {})

    async def test_manual_guidance_resumes_immediately_and_clears_retry_metadata(self):
        entry = await self.waiting(retry_at=5000)
        with mock.patch.object(self.bot, "_start_queue_drain") as wake:
            self.assertTrue(await self.bot._resume_paused_job("u1", "chat", "Use the available desktop", 0))
            wake.assert_called_once_with("u1")
        saved = await self.bot._load_job_checkpoint("u1", entry)
        self.assertIn(self.original, saved["request"])
        self.assertIn("Use the available desktop", saved["request"])
        self.assertEqual(saved["state"], "running")
        for key in ("next_retry_at", "retry_count", "retry_reason"):
            self.assertNotIn(key, saved)

    async def test_explicit_input_wait_and_non_head_retries_do_not_autostart(self):
        first = await self.enqueue(self.original)
        await self.bot._save_job_checkpoint("u1", first, {"state": "paused", "instruction": "Which target? [[await]]"})
        await self.waiting(retry_at=0)
        await self.bot._persist_promote("u1", first)
        with mock.patch.object(self.bot, "_start_queue_drain") as wake:
            await self.bot._check_job_retries()
            wake.assert_not_called()

    async def test_update_and_active_turn_prevent_due_claim_without_losing_retry(self):
        entry = await self.waiting(retry_at=0)
        with mock.patch.object(self.bot, "_start_queue_drain") as wake:
            self.bot._update_restart_pending = True
            await self.bot._check_job_retries()
            self.bot._update_restart_pending = False
            lock = self.bot._ensure_task_lock("u1")
            await lock.acquire()
            try:
                await self.bot._check_job_retries()
            finally:
                lock.release()
            wake.assert_not_called()
            saved = await self.bot._load_job_checkpoint("u1", entry)
            self.assertEqual(saved["state"], "paused")
            self.assertEqual(saved["next_retry_at"], 0)
            await self.bot._check_job_retries()
            wake.assert_called_once_with("u1")

    async def test_stop_wins_over_due_claim_waiting_for_admission(self):
        await self.waiting(retry_at=0)
        admission = self.bot._ensure_dispatch_admission_lock("u1")
        await admission.acquire()
        with mock.patch.object(self.bot, "_start_queue_drain") as wake:
            poll = asyncio.create_task(self.bot._check_job_retries())
            await asyncio.sleep(.02)
            stopping = asyncio.create_task(self.bot._cancel_user_work("u1"))
            await asyncio.sleep(0)
            admission.release()
            await asyncio.gather(poll, stopping)
            wake.assert_not_called()
        self.assertEqual(self.bot._read_queue_file(), {})
        self.assertTrue(self.bot._message_queues["u1"].empty())
        self.assertNotIn("u1", self.bot._paused_jobs)

    async def test_stop_clears_an_entry_retained_after_the_due_claim(self):
        await self.waiting(retry_at=0)
        self.bot._message_queues["u1"].get_nowait()
        entered, release = asyncio.Event(), asyncio.Event()
        retain = self.bot._retain_job_entry

        async def delayed_retain(*args):
            entered.set()
            await release.wait()
            await retain(*args)

        with (mock.patch.object(self.bot, "_retain_job_entry", delayed_retain),
              mock.patch.object(self.bot, "_start_queue_drain") as wake):
            poll = asyncio.create_task(self.bot._check_job_retries())
            await asyncio.wait_for(entered.wait(), 1)
            stopping = asyncio.create_task(self.bot._cancel_user_work("u1"))
            await asyncio.sleep(0)
            release.set()
            await asyncio.gather(poll, stopping)
            wake.assert_not_called()
        self.assertEqual(self.bot._read_queue_file(), {})
        self.assertTrue(self.bot._message_queues["u1"].empty())

    async def test_completion_watcher_checks_retries_before_provider_polling(self):
        calls = []

        async def retry():
            calls.append("retry")

        async def reply():
            calls.append("reply")
            raise asyncio.CancelledError

        with (mock.patch.object(self.bot, "_check_job_retries", retry),
              mock.patch.object(self.bot, "_check_reply_deliveries", reply)):
            with self.assertRaises(asyncio.CancelledError):
                await self.bot._detached_task_loop()
        self.assertEqual(calls, ["retry", "reply"])

    async def test_cancelled_claim_settles_disk_write_and_restores_as_runnable(self):
        entry = await self.waiting(retry_at=0)
        entered, release = threading.Event(), threading.Event()
        write = self.bot._write_queue_file

        def blocking_claim(data):
            if data.get("u1", [{}])[0].get("job", {}).get("state") == "running":
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("test claim was not released")
            write(data)

        with (mock.patch.object(self.bot, "_write_queue_file", side_effect=blocking_claim),
              mock.patch.object(self.bot, "_start_queue_drain") as wake):
            poll = asyncio.create_task(self.bot._check_job_retries())
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                poll.cancel()
                await asyncio.sleep(.01)
                self.assertFalse(poll.done())
            finally:
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await poll
            wake.assert_not_called()
        restarted = _ReplyDeliveryBot(self.ws, fail_final=False)
        with mock.patch.object(restarted, "_start_queue_drain") as wake:
            await restarted.restore_queues()
            wake.assert_called_once_with("u1")
        saved = await restarted._load_job_checkpoint("u1", entry)
        self.assertEqual(saved["state"], "running")
        self.assertNotIn("u1", restarted._paused_jobs)

    def test_invalid_retry_deadlines_never_create_a_busy_loop(self):
        for value in (None, True, False, "1", -1, float("inf"), float("nan"), 10 ** 1000):
            with self.subTest(value=repr(value)[:30]):
                self.assertFalse(self.bot._job_retry_due({
                    "job": {"state": "paused", "next_retry_at": value},
                }, 1000))

