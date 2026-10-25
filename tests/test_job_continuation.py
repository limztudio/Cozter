"""An unfinished request keeps its place until its full scope is complete."""

import asyncio
import os
import tempfile
import unittest
from unittest import mock

from Cozter.tests.helpers import persist_queue_entry

from Cozter import agent, session, workspace
from Cozter.backends_agent.base import AgentResult, ChatEvent
from Cozter.tests.test_reply_delivery import ReplyDeliveryBot


class JobContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.object(workspace, "CONFIG_DIR", self.tmp))
        self.enterContext(mock.patch.object(
            workspace, "get_run_config",
            return_value=("flexible", "", "", "full", "meta"),
        ))
        self.enterContext(mock.patch.object(agent, "_run_post_turn_maintenance", mock.AsyncMock()))
        self.bot = ReplyDeliveryBot(self.tmp, fail_final=False)
        self.sid = session.create_session(self.tmp, name="Original job")["id"]

    async def asyncTearDown(self):
        await agent._wait_for_pending_session_writes(self.tmp)
        await asyncio.sleep(0)

    async def enqueue(self, text, *, ephemeral=False, bot=None):
        bot = bot or self.bot
        entry_id = await persist_queue_entry(bot, "u1", text, "chat", ephemeral=ephemeral)
        bot._ensure_message_queue("u1").put_nowait((text, "chat", entry_id, ephemeral))
        return entry_id

    def final(self, text="Full original request completed", *, sid=None):
        return AgentResult(
            text=text, session_id=sid or self.sid,
            events=[ChatEvent(kind="text", content=text)],
        )

    def partial(self, *, reason="worker_budget", sid=None):
        return AgentResult(
            text="PARTIAL: repair completed; release remains unfinished",
            error="Worker allowance exhausted.",
            continue_instruction="Verify all targets, then commit and push the completed work",
            continuation_reason=reason,
            continuation_checkpoint={
                "request": "Repair every source and verify, commit, and push all work",
                "pending": ["Verify every source", "Commit and push the completed work"],
                "reports": [{"instruction": "Repair every source", "report": "All repairs are retained"}],
                "latest_draft": "Repair completed; release remains unfinished",
                "truncated": False,
            },
            session_id=sid or self.sid,
        )

    async def test_worker_exhaustion_gets_fresh_attempt_before_next_job(self):
        """Exercise worker exhaustion with a real saved continuation before later work."""
        original = "Repair every source and verify, commit, and push all work"
        later = "Start an unrelated scheduled optimization"
        await self.enqueue(original)
        await self.enqueue(later, ephemeral=True)
        calls = []
        budgets = []

        async def implementation(turn, prompt, ws, *_args, **kwargs):
            calls.append((prompt, ws, kwargs.get("session_id")))
            budgets.append(agent._TURN_BUDGET.get())
            turn.session_id = kwargs.get("session_id") or self.sid
            if len(calls) == 1:
                budget = agent._TURN_BUDGET.get()
                budget.pending = ["Verify every source", "Commit and push the completed work"]
                budget.reports = [("Repair every source", "All repairs are retained")]
                raise agent._TurnBudgetExhaustedError("worker allowance reached")
            if prompt == later:
                return self.final("Later task completed", sid=turn.session_id)
            return self.final(sid=turn.session_id)

        with mock.patch.object(agent, "_run_turn_impl", implementation):
            await asyncio.wait_for(self.bot._drain_message_queue("u1"), timeout=2)

        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][0], original)
        self.assertIn(original, calls[1][0])
        self.assertIn("Verify every source", calls[1][0])
        self.assertIn("Commit and push the completed work", calls[1][0])
        self.assertIn("All repairs are retained", calls[1][0])
        self.assertEqual(calls[1][2], self.sid)
        self.assertEqual(calls[2][0], later)
        self.assertIsNot(budgets[0], budgets[1])
        self.assertFalse(hasattr(budgets[1], "deadline"))
        self.assertLess(self.bot.sent.index("Full original request completed"),
                        self.bot.sent.index("Later task completed"))
        self.assertEqual(self.bot._read_queue_file(), {})

    async def test_judge_continuations_finish_before_next_job(self):
        await self.enqueue("Finish the entire original request")
        await self.enqueue("Later request")
        attempts = 5
        calls = []
        budgets = []

        async def run(prompt, _ws, **_kwargs):
            calls.append(prompt)
            budgets.append(agent._TURN_BUDGET.get())
            if len(calls) <= attempts:
                return AgentResult(text="Completed one more target", session_id=self.sid,
                                   continue_instruction="Finish the targets still remaining")
            return self.final("Later completed" if prompt == "Later request" else "Entire scope completed")

        with mock.patch.object(agent, "run", run):
            await asyncio.wait_for(self.bot._drain_message_queue("u1"), timeout=2)

        self.assertEqual(len(calls), attempts + 2)
        self.assertTrue(all("Finish the entire original request" in prompt for prompt in calls[:-1]))
        self.assertEqual(calls[-1], "Later request")
        self.assertEqual(len({id(budget) for budget in budgets}), len(budgets))
        self.assertIn("Entire scope completed", self.bot.sent)
        self.assertEqual(self.bot._read_queue_file(), {})

    async def test_update_checkpoint_survives_restart_with_scheduled_session_and_workspace(self):
        original = "Repair every source and verify, commit, and push all work"
        entry = await self.enqueue(original, ephemeral=True)
        await self.enqueue("Later request")
        first_calls = []
        artifact = os.path.join(self.tmp, "retained.txt")
        with open(artifact, "w", encoding="utf-8") as stream:
            stream.write("Retained repair evidence")

        async def pause_for_update(prompt, ws, **kwargs):
            first_calls.append((prompt, ws, kwargs["session_id"]))
            self.bot._update_restart_pending = True
            result = self.partial(sid=kwargs["session_id"])
            result.events.append(ChatEvent(kind="attachment", content=artifact))
            result.usage = {"input_tokens": 3}
            return result

        with mock.patch.object(agent, "run", pause_for_update):
            await asyncio.wait_for(self.bot._drain_message_queue("u1"), timeout=2)

        self.assertEqual(len(first_calls), 1)
        scheduled_sid = first_calls[0][2]
        saved = await self.bot._load_job_checkpoint("u1", entry)
        self.assertEqual(saved["workspace_path"], self.tmp)
        self.assertEqual(saved["session_id"], scheduled_sid)
        self.assertEqual(saved["request"], original)
        self.assertEqual(saved["attempts"], 1)
        self.assertIsNotNone(session.load_session(self.tmp, scheduled_sid))
        self.assertEqual([item["text"] for item in self.bot._read_queue_file()["u1"]],
                         [original, "Later request"])

        switched_ws = self.enterContext(tempfile.TemporaryDirectory())
        restarted = ReplyDeliveryBot(switched_ws, fail_final=False)
        resumed_calls = []

        async def finish_after_restart(prompt, ws, **kwargs):
            resumed_calls.append((prompt, ws, kwargs.get("session_id")))
            result = self.final("Later completed" if prompt == "Later request" else "Original completed",
                                sid=kwargs.get("session_id") or self.sid)
            result.usage = {"input_tokens": 7}
            return result

        with (mock.patch.object(restarted, "_start_queue_drain"),
              mock.patch.object(agent, "run", finish_after_restart),
              mock.patch.object(restarted, "_stage_reply_delivery",
                                wraps=restarted._stage_reply_delivery) as stage):
            await restarted.restore_queues()
            await asyncio.wait_for(restarted._drain_message_queue("u1"), timeout=2)

        self.assertEqual(resumed_calls[0][1:], (self.tmp, scheduled_sid))
        self.assertIn(original, resumed_calls[0][0])
        self.assertIn("All repairs are retained", resumed_calls[0][0])
        self.assertIn("Commit and push the completed work", resumed_calls[0][0])
        self.assertEqual(resumed_calls[1][0:2], ("Later request", switched_ws))
        completed = stage.await_args_list[0].args[4]
        self.assertEqual(completed.usage, {"input_tokens": 10})
        self.assertEqual([event.content for event in completed.events if event.kind == "attachment"],
                         [artifact])
        self.assertIsNone(session.load_session(self.tmp, scheduled_sid))
        self.assertEqual(restarted._read_queue_file(), {})

    async def test_shutdown_keeps_scheduled_session_and_completed_checkpoint(self):
        original = "Repair every source and verify, commit, and push all work"
        entry = await self.enqueue(original, ephemeral=True)
        await self.enqueue("Later request")
        first_calls = []

        async def interrupted(prompt, ws, **kwargs):
            first_calls.append((prompt, ws, kwargs["session_id"]))
            if len(first_calls) == 1:
                return self.partial(reason="worker_budget", sid=kwargs["session_id"])
            raise asyncio.CancelledError

        with mock.patch.object(agent, "run", interrupted):
            await asyncio.wait_for(self.bot._drain_message_queue("u1"), timeout=2)

        self.assertEqual(len(first_calls), 2)
        scheduled_sid = first_calls[0][2]
        saved = await self.bot._load_job_checkpoint("u1", entry)
        self.assertEqual(saved["session_id"], scheduled_sid)
        self.assertEqual(saved["checkpoint"]["pending"],
                         ["Verify every source", "Commit and push the completed work"])
        self.assertIsNotNone(session.load_session(self.tmp, scheduled_sid))
        self.assertEqual(len(self.bot._read_queue_file()["u1"]), 2)

        restarted = ReplyDeliveryBot(self.tmp, fail_final=False)
        calls = []

        async def resumed(prompt, ws, **kwargs):
            calls.append((prompt, ws, kwargs.get("session_id")))
            return self.final(sid=kwargs.get("session_id") or self.sid)

        with (mock.patch.object(restarted, "_start_queue_drain"),
              mock.patch.object(agent, "run", resumed)):
            await restarted.restore_queues()
            await asyncio.wait_for(restarted._drain_message_queue("u1"), timeout=2)

        self.assertEqual(calls[0][1:], (self.tmp, scheduled_sid))
        self.assertIn("All repairs are retained", calls[0][0])
        self.assertEqual(calls[1][0], "Later request")
        self.assertIsNone(session.load_session(self.tmp, scheduled_sid))
        self.assertEqual(restarted._read_queue_file(), {})

    async def test_unexpected_provider_exception_pauses_current_job(self):
        original = "Complete the full repair and verification"
        entry = await self.enqueue(original)
        await self.enqueue("Later scheduled task", ephemeral=True)
        with mock.patch.object(agent, "run", mock.AsyncMock(side_effect=OSError("connection reset"))) as run:
            with self.assertLogs("Cozter.backends_bot.base", level="ERROR"):
                await asyncio.wait_for(self.bot._drain_message_queue("u1"), timeout=2)
            run.assert_awaited_once()

        saved = await self.bot._load_job_checkpoint("u1", entry)
        self.assertEqual(saved["state"], "paused")
        self.assertEqual(saved["request"], original)
        self.assertEqual([item["text"] for item in self.bot._read_queue_file()["u1"]],
                         [original, "Later scheduled task"])

    async def test_provider_error_blocks_later_jobs_and_plain_reply_resumes_original(self):
        original = "Finish all requested repairs and release verification"
        entry = await self.enqueue(original)
        await self.enqueue("Later scheduled task", ephemeral=True)
        with mock.patch.object(agent, "run", mock.AsyncMock(return_value=AgentResult(
            text="Authentication is unavailable", error="invalid API key", session_id=self.sid,
        ))) as first:
            await asyncio.wait_for(self.bot._drain_message_queue("u1"), timeout=2)
            first.assert_awaited_once()

        saved = await self.bot._load_job_checkpoint("u1", entry)
        self.assertEqual(saved["state"], "paused")
        self.assertEqual(saved["request"], original)
        self.assertEqual([item["text"] for item in self.bot._read_queue_file()["u1"]],
                         [original, "Later scheduled task"])

        restarted = ReplyDeliveryBot(self.tmp, fail_final=False)
        with (mock.patch.object(restarted, "_start_queue_drain"),
              mock.patch.object(agent, "run", mock.AsyncMock()) as run):
            await restarted.restore_queues()
            await restarted._drain_message_queue("u1")
            run.assert_not_awaited()

        calls = []

        async def resume(prompt, _ws, **kwargs):
            calls.append((prompt, kwargs.get("session_id")))
            return self.final("Later completed" if prompt == "Later scheduled task" else "Original completed",
                              sid=kwargs.get("session_id") or self.sid)

        with (mock.patch.object(agent, "run", resume),
              mock.patch.object(restarted, "_start_queue_drain")):
            await asyncio.wait_for(restarted._dispatch_ai(
                restarted.make_context("u1", "chat"), "Credentials repaired; continue the original request",
            ), timeout=2)
            await asyncio.wait_for(restarted._drain_message_queue("u1"), timeout=2)

        self.assertEqual(len(calls), 2)
        self.assertIn(original, calls[0][0])
        self.assertIn("Credentials repaired", calls[0][0])
        self.assertEqual(calls[0][1], self.sid)
        self.assertEqual(calls[1][0], "Later scheduled task")
        self.assertEqual(restarted._read_queue_file(), {})

    async def test_await_keeps_current_job_ahead_of_scheduled_backlog(self):
        original = "Finish release after choosing its target"
        entry = await self.enqueue(original)
        await self.enqueue("Later scheduled task", ephemeral=True)
        question = AgentResult(
            text="Which target should I release? [[await]]", session_id=self.sid,
            events=[ChatEvent(kind="text", content="Which target should I release? [[await]]")],
        )
        with mock.patch.object(agent, "run", mock.AsyncMock(return_value=question)) as run:
            await asyncio.wait_for(self.bot._drain_message_queue("u1"), timeout=2)
            run.assert_awaited_once()

        saved = await self.bot._load_job_checkpoint("u1", entry)
        self.assertEqual(saved["state"], "paused")
        self.assertEqual([item["text"] for item in self.bot._read_queue_file()["u1"]],
                         [original, "Later scheduled task"])
        calls = []

        async def finish(prompt, _ws, **kwargs):
            calls.append(prompt)
            return self.final(sid=kwargs.get("session_id") or self.sid)

        with (mock.patch.object(agent, "run", finish),
              mock.patch.object(self.bot, "_start_queue_drain")):
            await asyncio.wait_for(self.bot._dispatch_ai(
                self.bot.make_context("u1", "chat"), "Use the staging target",
            ), timeout=2)
            await self.bot._drain_message_queue("u1")

        self.assertEqual(len(calls), 2)
        self.assertIn(original, calls[0])
        self.assertIn("Use the staging target", calls[0])
        self.assertEqual(calls[1], "Later scheduled task")
        self.assertEqual(self.bot._read_queue_file(), {})

    async def test_stop_discards_paused_job_and_checkpoint_cannot_resurrect_it(self):
        entry = await self.enqueue("Original unfinished job")
        await self.enqueue("Later job")
        with mock.patch.object(agent, "run", mock.AsyncMock(return_value=AgentResult(
            text="Provider unavailable", error="service unavailable", session_id=self.sid,
        ))):
            await self.bot._drain_message_queue("u1")
        saved = await self.bot._load_job_checkpoint("u1", entry)
        self.assertEqual(saved["state"], "paused")

        await self.bot.cmd_stop(self.bot.make_context("u1", "chat", command="stop"))
        self.assertEqual(self.bot._read_queue_file(), {})
        self.assertTrue(self.bot._message_queues["u1"].empty())
        self.assertNotIn("u1", self.bot._awaiting_answer)
        self.assertIn("Cancelled.", self.bot.sent)
        self.assertFalse(await self.bot._save_job_checkpoint("u1", entry, saved))
        self.assertEqual(self.bot._read_queue_file(), {})

    async def test_checkpoint_update_never_changes_original_queue_request(self):
        original = "Repair all targets, then verify and push"
        entry = await self.enqueue(original)
        saved = {
            "workspace_path": self.tmp, "session_id": self.sid,
            "request": original, "instruction": "Verify and push retained repairs",
            "checkpoint": {"pending": ["Verify and push"], "reports": []},
            "state": "running", "attempts": 1, "output": {},
        }
        self.assertTrue(await self.bot._save_job_checkpoint("u1", entry, saved))
        self.assertEqual((await self.bot._load_job_checkpoint("u1", entry))["instruction"],
                         "Verify and push retained repairs")
        self.assertEqual(self.bot._read_queue_file()["u1"][0]["text"], original)
        await self.bot._persist_complete("u1", entry)
        self.assertFalse(await self.bot._save_job_checkpoint("u1", entry, saved))
        self.assertEqual(self.bot._read_queue_file(), {})

    async def test_stop_cleans_up_retained_paused_scheduled_session(self):
        entry = await self.enqueue("Unfinished scheduled repair", ephemeral=True)
        await self.enqueue("Later request")

        async def unavailable(_prompt, _ws, **kwargs):
            return AgentResult(text="Provider unavailable", error="service unavailable",
                               session_id=kwargs["session_id"])

        with mock.patch.object(agent, "run", unavailable):
            await self.bot._drain_message_queue("u1")

        saved = await self.bot._load_job_checkpoint("u1", entry)
        scheduled_sid = saved["session_id"]
        self.assertEqual(saved["state"], "paused")
        self.assertIsNotNone(session.load_session(self.tmp, scheduled_sid))
        await self.bot.cmd_stop(self.bot.make_context("u1", "chat", command="stop"))
        self.assertEqual(self.bot._read_queue_file(), {})
        self.assertIsNone(session.load_session(self.tmp, scheduled_sid))

    async def test_unverified_done_claim_keeps_current_job_and_never_stages_final_reply(self):
        original = "Repair all sources, verify all targets, then release"
        entry = await self.enqueue(original)
        await self.enqueue("Later scheduled task", ephemeral=True)
        result = self.final("DONE: every requested obligation is complete")
        result.completion_verified = False

        with (mock.patch.object(agent, "run", mock.AsyncMock(return_value=result)) as run,
              mock.patch.object(self.bot, "_stage_reply_delivery",
                                wraps=self.bot._stage_reply_delivery) as stage):
            await asyncio.wait_for(self.bot._drain_message_queue("u1"), timeout=2)

        run.assert_awaited_once()
        stage.assert_not_awaited()
        saved = await self.bot._load_job_checkpoint("u1", entry)
        self.assertEqual(saved["state"], "paused")
        self.assertEqual(saved["request"], original)
        self.assertEqual([item["text"] for item in self.bot._read_queue_file()["u1"]],
                         [original, "Later scheduled task"])
        self.assertEqual(await self.bot._list_reply_delivery_records(), [])
        self.assertTrue(any("Completion could not be verified" in text for text in self.bot.sent))

    async def test_different_chat_cannot_bypass_the_users_paused_job(self):
        original = "Finish the original repair and full verification"
        entry = await self.enqueue(original)
        with mock.patch.object(agent, "run", mock.AsyncMock(return_value=AgentResult(
            text="Provider unavailable", error="service unavailable", session_id=self.sid,
        ))):
            await self.bot._drain_message_queue("u1")

        self.assertTrue(self.bot.authorized("u1", "other-chat"))
        incoming = "Start a different job in this other chat"
        with mock.patch.object(agent, "run", mock.AsyncMock(return_value=self.final())) as run:
            await asyncio.wait_for(self.bot._dispatch_ai(
                self.bot.make_context("u1", "other-chat"), incoming,
            ), timeout=2)
            await asyncio.sleep(0)

        run.assert_not_awaited()
        saved = await self.bot._load_job_checkpoint("u1", entry)
        self.assertEqual(saved["state"], "paused")
        self.assertEqual(saved["request"], original)
        entries = self.bot._read_queue_file()["u1"]
        self.assertEqual([item["text"] for item in entries], [original, incoming])
        self.assertEqual(entries[0]["chat_id"], "chat")
        self.assertEqual(entries[1]["chat_id"], "other-chat")

    async def test_guidance_during_blocked_pause_notice_resumes_without_another_message(self):
        await self._assert_guidance_during_blocked_notice_resumes()

    async def test_provider_exception_guidance_during_blocked_notice_resumes_without_another_message(self):
        with self.assertLogs("Cozter.backends_bot.base", level="ERROR"):
            await self._assert_guidance_during_blocked_notice_resumes(provider_exception=True)

    async def _assert_guidance_during_blocked_notice_resumes(self, *, provider_exception=False):
        """A wake-up while the old drain owns the lock must not strand the job."""
        original = "Finish the original repairs and release verification"
        await self.enqueue(original)
        await self.enqueue("Later scheduled task", ephemeral=True)
        notice_started = asyncio.Event()
        notice_release = asyncio.Event()
        kicked_drain_finished = asyncio.Event()
        real_send = self.bot.send_text
        real_drain = self.bot._drain_message_queue
        calls = []
        drain_calls = 0

        async def blocked_notice(chat_id, text, *, rich=False):
            if ("The current job is still unfinished" in text
                    or "Current job remains unfinished" in text):
                notice_started.set()
                await notice_release.wait()
            await real_send(chat_id, text, rich=rich)

        async def observe_drain(uid):
            nonlocal drain_calls
            drain_calls += 1
            number = drain_calls
            try:
                await real_drain(uid)
            finally:
                if number == 2:
                    kicked_drain_finished.set()

        async def run(prompt, _ws, **kwargs):
            calls.append(prompt)
            if len(calls) == 1:
                if provider_exception:
                    raise RuntimeError("provider disconnected")
                return AgentResult(text="Provider unavailable", error="service unavailable",
                                   session_id=self.sid)
            return self.final(sid=kwargs.get("session_id") or self.sid)

        with (mock.patch.object(agent, "run", run),
              mock.patch.object(self.bot, "send_text", blocked_notice),
              mock.patch.object(self.bot, "_drain_message_queue", observe_drain)):
            original_drain = asyncio.create_task(self.bot._drain_message_queue("u1"))
            try:
                await asyncio.wait_for(notice_started.wait(), timeout=2)
                self.assertTrue(self.bot._ensure_task_lock("u1").locked())
                await asyncio.wait_for(self.bot._dispatch_ai(
                    self.bot.make_context("u1", "chat"), "Provider repaired; continue the full original job",
                ), timeout=2)
                # The automatically kicked drain returned because the old
                # attempt still owns the lock. No manual second drain follows.
                await asyncio.wait_for(kicked_drain_finished.wait(), timeout=2)
                self.assertEqual(calls, [original])
                self.assertFalse(original_drain.done())
                notice_release.set()
                await asyncio.wait_for(original_drain, timeout=2)
            finally:
                notice_release.set()
                if not original_drain.done():
                    original_drain.cancel()
                await asyncio.gather(original_drain, return_exceptions=True)

        self.assertEqual(len(calls), 3)
        self.assertIn(original, calls[1])
        self.assertIn("Provider repaired", calls[1])
        self.assertEqual(calls[2], "Later scheduled task")
        self.assertEqual(self.bot._read_queue_file(), {})

    async def test_worker_evidence_accumulates_across_attempts_without_nested_resume_requests(self):
        original = "Repair every target and verify both artifacts, then commit and push"
        entry = await self.enqueue(original)
        await self.enqueue("Later scheduled task", ephemeral=True)
        first_evidence = "First artifact evidence: first.json contains all repaired target IDs"
        second_evidence = "Second artifact evidence: second.json records every successful verification"
        calls = []
        snapshots = []

        async def run(prompt, _ws, **_kwargs):
            calls.append(prompt)
            if len(calls) <= 2:
                result = self.partial(reason="worker_budget")
                result.continuation_checkpoint.update({
                    # The agent receives the generated resume prompt on the
                    # second attempt; saving it recursively would multiply
                    # scope and evidence on every attempt.
                    "request": prompt,
                    "reports": [{
                        "instruction": "Repair first artifact" if len(calls) == 1 else "Verify second artifact",
                        "report": first_evidence if len(calls) == 1 else second_evidence,
                    }],
                    "pending": ["Commit and push both verified artifacts"],
                })
                return result
            if len(calls) == 3:
                snapshots.append(await self.bot._load_job_checkpoint("u1", entry))
            return self.final("Later completed" if prompt == "Later scheduled task" else "Original completed")

        with mock.patch.object(agent, "run", run):
            await asyncio.wait_for(self.bot._drain_message_queue("u1"), timeout=2)

        self.assertEqual(len(calls), 4)
        self.assertIn(first_evidence, calls[1])
        self.assertIn(first_evidence, calls[2])
        self.assertIn(second_evidence, calls[2])
        self.assertEqual(calls[2].count(original), 1)
        self.assertEqual(calls[2].count("[Resume the SAME unfinished job"), 1)
        self.assertEqual(snapshots[0]["request"], original)
        self.assertNotIn("request", snapshots[0]["checkpoint"])
        self.assertEqual([row["report"] for row in snapshots[0]["checkpoint"]["reports"]],
                         [first_evidence, second_evidence])
        self.assertEqual(calls[3], "Later scheduled task")
        self.assertEqual(self.bot._read_queue_file(), {})

    async def test_interactive_session_is_checkpointed_before_worker_actions_and_recovers_after_cancel(self):
        original = "Finish the original repair and full verification"
        entry = await self.enqueue(original)
        await self.enqueue("Later request")
        snapshots = []
        worker_actions = []

        async def interrupted(_prompt, _ws, **kwargs):
            await kwargs["on_session"](self.sid)
            snapshots.append((await self.bot._load_job_checkpoint("u1", entry), list(worker_actions)))
            worker_actions.append("worker began applying retained repairs")
            raise asyncio.CancelledError

        with mock.patch.object(agent, "run", interrupted):
            await asyncio.wait_for(self.bot._drain_message_queue("u1"), timeout=2)

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0][0]["session_id"], self.sid)
        self.assertEqual(snapshots[0][0]["workspace_path"], self.tmp)
        self.assertEqual(snapshots[0][1], [])
        saved = await self.bot._load_job_checkpoint("u1", entry)
        self.assertEqual(saved["session_id"], self.sid)
        self.assertEqual(saved["request"], original)
        self.assertEqual(saved["state"], "running")
        self.assertEqual([row["text"] for row in self.bot._read_queue_file()["u1"]],
                         [original, "Later request"])

        switched_ws = self.enterContext(tempfile.TemporaryDirectory())
        restarted = ReplyDeliveryBot(switched_ws, fail_final=False)
        resumed = []

        async def finish(prompt, ws, **kwargs):
            resumed.append((prompt, ws, kwargs.get("session_id")))
            return self.final(sid=kwargs.get("session_id") or self.sid)

        with (mock.patch.object(restarted, "_start_queue_drain"),
              mock.patch.object(agent, "run", finish)):
            await restarted.restore_queues()
            await asyncio.wait_for(restarted._drain_message_queue("u1"), timeout=2)

        self.assertEqual(resumed[0], (original, self.tmp, self.sid))
        self.assertEqual(resumed[1][0:2], ("Later request", switched_ws))
        self.assertEqual(restarted._read_queue_file(), {})

    async def test_large_worker_checkpoints_keep_evidence_usable_in_next_attempt(self):
        original = "Repair every source and verify all retained artifacts, then release"
        entry = await self.enqueue(original)
        await self.enqueue("Later request")
        first_evidence = "First artifact evidence: repaired-targets.json lists all repaired source IDs"
        second_evidence = "Second artifact evidence: verified-targets.json records every successful check"
        calls = []
        snapshots = []

        async def run(prompt, _ws, **_kwargs):
            calls.append(prompt)
            if len(calls) <= 2:
                result = self.partial(reason="worker_budget")
                result.continuation_checkpoint.update({
                    "request": prompt,
                    "reports": [{
                        "instruction": "Repair sources" if len(calls) == 1 else "Verify sources",
                        "report": (first_evidence if len(calls) == 1 else second_evidence)
                                  + "\nDetailed provider verification evidence. " * 1550,
                    }],
                    "pending": ["Commit and push all verified work"],
                })
                return result
            if len(calls) == 3:
                snapshots.append(await self.bot._load_job_checkpoint("u1", entry))
            return self.final("Later completed" if prompt == "Later request" else "Original completed")

        with mock.patch.object(agent, "run", run):
            await asyncio.wait_for(self.bot._drain_message_queue("u1"), timeout=2)

        self.assertEqual(len(calls), 4)
        self.assertIn(first_evidence, calls[2])
        self.assertIn(second_evidence, calls[2])
        self.assertIn("Commit and push all verified work", calls[2])
        checkpoint = snapshots[0]["checkpoint"]
        self.assertTrue(checkpoint["truncated"])
        retained = "\n".join(row["report"] for row in checkpoint["reports"])
        self.assertIn(first_evidence, retained)
        self.assertIn(second_evidence, retained)
        self.assertIn("verify its artifacts", retained)
        self.assertEqual(calls[3], "Later request")
        self.assertEqual(self.bot._read_queue_file(), {})

    async def test_scheduled_missing_workspace_pauses_and_resumes_after_origin_is_restored(self):
        original = "Finish scheduled repairs and all release verification"
        entry = await self.enqueue(original, ephemeral=True)
        origin = os.path.join(self.tmp, "temporarily-unavailable-origin")
        await self.bot._save_job_checkpoint("u1", entry, {"workspace_path": origin})
        await self.enqueue("Later request")
        with (mock.patch.object(self.bot, "_current_workspace_for_turn", mock.AsyncMock(return_value=None)),
              mock.patch.object(agent, "run", mock.AsyncMock()) as run):
            await asyncio.wait_for(self.bot._drain_message_queue("u1"), timeout=2)

        run.assert_not_awaited()
        saved = await self.bot._load_job_checkpoint("u1", entry)
        self.assertEqual(saved["state"], "paused")
        self.assertEqual(saved["request"], original)
        self.assertEqual([row["text"] for row in self.bot._read_queue_file()["u1"]],
                         [original, "Later request"])

        reopened_ws = self.enterContext(tempfile.TemporaryDirectory())
        self.bot.workspace_path = reopened_ws
        os.mkdir(origin)
        calls = []

        async def resumed(prompt, ws, **kwargs):
            calls.append((prompt, ws, kwargs.get("session_id")))
            return self.final(sid=kwargs.get("session_id") or self.sid)

        with (mock.patch.object(agent, "run", resumed),
              mock.patch.object(self.bot, "_start_queue_drain")):
            await asyncio.wait_for(self.bot._dispatch_ai(
                self.bot.make_context("u1", "chat"), "Origin restored; continue the full scheduled job",
            ), timeout=2)
            await asyncio.wait_for(self.bot._drain_message_queue("u1"), timeout=2)

        self.assertEqual(len(calls), 2)
        self.assertIn(original, calls[0][0])
        self.assertIn("Origin restored", calls[0][0])
        self.assertEqual(calls[0][1], origin)
        self.assertIsNotNone(calls[0][2])
        self.assertEqual(calls[1][0:2], ("Later request", reopened_ws))
        self.assertIsNone(session.load_session(origin, calls[0][2]))
        self.assertEqual(self.bot._read_queue_file(), {})


if __name__ == "__main__":
    unittest.main()
