"""Durable flexible plans resume workers, preserve evidence, and bound fan-out."""

import asyncio
from collections import Counter, defaultdict
import os
import tempfile
import unittest
from unittest import mock

from Cozter.tests.helpers import persist_queue_entry

from Cozter import agent, execution_state, flexible, session, workspace
from Cozter.backends_agent.base import AgentResult, ChatEvent
from Cozter.tests.test_reply_delivery import ReplyDeliveryBot


class FlexibleContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.object(workspace, "CONFIG_DIR", self.ws))
        self.enterContext(mock.patch.object(agent, "_run_post_turn_maintenance", mock.AsyncMock()))
        for tier in flexible.TIERS:
            workspace.set_flexible_backend_name(self.ws, tier, "meta")
        self.sid = session.create_session(self.ws, name="Durable flexible test")["id"]
        self.plan = "[PLAN]\n1. [low] first\n2. [mid] second\n3. [high] third\n4. [high] fourth\n[/PLAN]"
        self.planner_calls = 0
        self.merge_prompts = []
        self.plan_records = []

    async def asyncTearDown(self):
        await agent._wait_for_pending_session_writes(self.ws)

    async def internal(self, _backend, _ws, prompt, _model, *, label, **_kwargs):
        if label == "Flexible planner":
            self.planner_calls += 1
            return self.plan
        self.merge_prompts.append(prompt)
        self.plan_records.append(await execution_state.record("flexible_plan"))
        return "final response"

    async def invoke(self, drive, *, job="job", canonical="Original requested scope", request="transport", **kwargs):
        async with execution_state.execution_scope(self.ws, job, canonical):
            with (mock.patch.object(agent, "run_internal_backend", self.internal),
                  mock.patch.object(agent, "_drive_backend", drive)):
                return await agent._run_flexible(
                    request, request, self.ws, approval=kwargs.pop("approval", "full"),
                    effort=kwargs.pop("effort", 100), collaborative=False,
                    summary_backend_name="meta", summary_model=kwargs.pop("summary_model", "summary"),
                    on_event=None, inject_queue=None, injected=kwargs.pop("injected", []), **kwargs,
                )

    async def test_real_durable_bot_crosses_two_allowances_with_one_plan_and_no_completed_replay(self):
        original = "Repair and verify the four requested parts, then commit."
        bot = ReplyDeliveryBot(self.ws, fail_final=False)
        bot.supports_transient_status = False
        entry = await persist_queue_entry(bot, "u1", original, "chat")
        bot._ensure_message_queue("u1").put_nowait((original, "chat", entry, False))
        await bot._save_job_checkpoint("u1", entry, {
            "state": "running", "workspace_path": self.ws, "request": original, "session_id": self.sid,
        })
        artifact = os.path.join(self.ws, "proof.txt")
        with open(artifact, "w", encoding="utf-8") as out:
            out.write("retained verification artifact")
        calls = Counter()
        roles = defaultdict(list)
        delivered = []
        prompts = []

        async def drive(_backend, _ws, prompt, _model, _approval, **_kwargs):
            index = int(prompt.split("[Sub-task ")[1].split("/")[0])
            calls[index] += 1
            prompts.append(prompt)
            probe = await execution_state.record("role-probe")
            roles[index].append(probe.key)
            if index in (2, 3) and calls[index] == 1:
                raise agent._TurnBudgetExhaustedError("worker allowance reached")
            return AgentResult(
                text=f"Verification evidence {index}", usage={"input_tokens": index},
                events=[ChatEvent(kind="attachment", content=artifact)] if index == 1 else [],
            ), False

        async def deliver(_chat, _ws, result, **_kwargs):
            delivered.append(result)

        with (mock.patch.object(workspace, "get_run_config", return_value=(
                    "flexible", None, "summary", "full", "meta")),
              mock.patch.object(agent, "run_internal_backend", self.internal),
              mock.patch.object(agent, "_drive_backend", drive),
              mock.patch.object(agent, "_judge_draft", mock.AsyncMock(return_value=(
                  flexible.JudgeVerdict(False), False))),
              mock.patch.object(bot, "_send_result", deliver)):
            async with asyncio.timeout(8):
                await bot._drain_message_queue("u1")
        self.assertEqual(self.planner_calls, 1)
        self.assertEqual(calls, {1: 1, 2: 2, 3: 2, 4: 1})
        self.assertEqual(roles[2][0], roles[2][1])
        self.assertEqual(roles[3][0], roles[3][1])
        self.assertTrue(any("[Resume the SAME unfinished job" in prompt for prompt in prompts))
        for index in range(1, 5):
            self.assertIn(f"Verification evidence {index}", self.merge_prompts[-1])
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0].usage, {"input_tokens": 10})
        self.assertTrue(any(event.kind == "attachment" for event in delivered[0].events))
        self.assertEqual(bot._read_queue_file(), {})
        # Final delivery acknowledges/removes the job's on-disk state. The
        # last merge still holds its record, which must already be inactive.
        saved = self.plan_records[-1]
        self.assertFalse(saved.get("active"))
        self.assertEqual(saved.get("cursor"), 4)
        self.assertEqual(saved.get("usage_processed_audit"), {"input_tokens": 10})

    async def test_completed_parallel_reader_survives_sibling_cancellation(self):
        self.plan = ("[PLAN]\n1. [low] [parallel-read] first\n"
                     "2. [mid] [parallel-read] second\n3. [high] third\n[/PLAN]")
        calls = Counter()
        completed = asyncio.Event()
        cancel = True
        marker = "[[attach: long-worker-proof.txt]]"
        with open(os.path.join(self.ws, "long-worker-proof.txt"), "w", encoding="utf-8") as out:
            out.write("worker proof")

        async def drive(_backend, _ws, prompt, *_args, **_kwargs):
            index = int(prompt.split("[Sub-task ")[1].split("/")[0])
            calls[index] += 1
            if index == 2 and cancel:
                await completed.wait()
                # Wait for reader 1's durable completion, not an arbitrary tick.
                async with asyncio.timeout(1):
                    while True:
                        saved = await execution_state.current_execution().record({
                            "record": "flexible_plan", "worker": "main",
                        })
                        if saved.get("worker_outputs", {}).get("0", {}).get("complete"):
                            stored = saved.get("worker_outputs")["0"]
                            self.assertNotIn(marker, stored["text"])
                            self.assertEqual(stored["attach_markers"], [marker])
                            break
                        await asyncio.sleep(0)
                raise asyncio.CancelledError
            completed.set()
            text = ("HEAD" * 2_000 + "\n" + marker + "\n" + "TAIL" * 2_000
                    if index == 1 else f"reader evidence {index}")
            return AgentResult(text=text), False

        with self.assertRaises(asyncio.CancelledError):
            await self.invoke(drive)
        cancel = False
        result, restarting = await self.invoke(drive, request="new resume wrapper")
        self.assertFalse(restarting)
        self.assertEqual(self.planner_calls, 1)
        self.assertEqual(calls, {1: 1, 2: 2, 3: 1})
        self.assertIn(marker, result.text)
        self.assertIn(flexible.REPORT_TRUNCATION_MARKER, self.merge_prompts[-1])

    async def test_interrupted_judge_followup_phase_restores_its_draft_on_normal_resume(self):
        self.plan = "[PLAN]\n1. [low] first\n2. [high] second\n[/PLAN]"
        calls = Counter()
        roles = []

        async def drive(_backend, _ws, prompt, *_args, **_kwargs):
            self.assertIn("Earlier verified repairs and artifacts", prompt)
            index = int(prompt.split("[Sub-task ")[1].split("/")[0])
            calls[index] += 1
            roles.append((index, (await execution_state.record("role-probe")).key))
            if index == 2 and calls[index] == 1:
                raise asyncio.CancelledError
            return AgentResult(text="new verification evidence"), False

        with self.assertRaises(asyncio.CancelledError):
            await self.invoke(drive, previous_draft="Earlier verified repairs and artifacts")
        await self.invoke(drive, request="normal allowance resume wrapper")
        self.assertEqual(self.planner_calls, 1)
        self.assertEqual(calls, {1: 1, 2: 2})
        self.assertEqual(roles[1][1], roles[2][1])

    async def test_completed_verification_can_rerun_after_repair_and_escalation_is_distinct(self):
        self.plan = "[PLAN]\n1. [low] Run test suite\n2. [high] Repair failures\n[/PLAN]"
        prompts = []

        async def drive(_backend, _ws, prompt, *_args, **_kwargs):
            prompts.append(prompt)
            if len(prompts) == 2:
                return AgentResult(text=(
                    "Repair complete\n[followup:low] Run test suite\n"
                    "[followup:low] Please Run test suite\n[followup:high] Run test suite"
                )), False
            return AgentResult(text="verified"), False

        await self.invoke(drive)
        self.assertEqual(len(prompts), 4)
        self.assertIn("[Sub-task 3/4: low]", prompts[2])
        self.assertIn("[Sub-task 4/4: high]", prompts[3])

    async def test_failure_keeps_cursor_but_finished_phase_cannot_be_reused(self):
        self.plan = "[PLAN]\n1. [low] first\n2. [high] second\n[/PLAN]"
        calls = Counter()

        async def drive(_backend, _ws, prompt, *_args, **_kwargs):
            index = int(prompt.split("[Sub-task ")[1].split("/")[0])
            calls[index] += 1
            if index == 2 and calls[index] == 1:
                raise RuntimeError("fixture backend interruption")
            return AgentResult(text=f"evidence {index}"), False

        with self.assertRaisesRegex(RuntimeError, "fixture backend"):
            await self.invoke(drive)
        await self.invoke(drive, request="another transport wrapper")
        self.assertEqual(calls, {1: 1, 2: 2})
        self.assertEqual(self.planner_calls, 1)
        # A new phase after completion, such as judge CONTINUE, needs a plan.
        await self.invoke(drive, previous_draft="judge requested another verification")
        self.assertEqual(self.planner_calls, 2)
        self.assertEqual(calls, {1: 2, 2: 3})

    async def test_injection_refreshes_plan_but_preserves_identical_worker_roles(self):
        self.plan = "[PLAN]\n1. [low] first\n2. [high] second\n[/PLAN]"
        injected = []
        calls = Counter()
        roles = []

        async def drive(_backend, _ws, prompt, *_args, **_kwargs):
            index = int(prompt.split("[Sub-task ")[1].split("/")[0])
            calls[index] += 1
            roles.append((index, (await execution_state.record("role-probe")).key))
            if index == 2 and calls[index] == 1:
                injected.append("Preserve the newly requested compatibility rule.")
                return AgentResult(), True
            return AgentResult(text=f"evidence {index}"), False

        _, restarted = await self.invoke(drive, injected=injected)
        self.assertTrue(restarted)
        await self.invoke(drive, injected=injected)
        self.assertEqual(self.planner_calls, 2)
        self.assertEqual(calls, {1: 2, 2: 2})
        self.assertEqual(roles[0][1], roles[2][1])
        self.assertEqual(roles[1][1], roles[3][1])

    async def test_request_model_permission_effort_and_job_changes_invalidate_pending_plan(self):
        self.plan = "[PLAN]\n1. [low] first\n2. [high] second\n[/PLAN]"
        for changed in ({"canonical": "Changed requested scope"}, {"approval": "confirm"},
                        {"effort": 80}, {"summary_model": "new-summary"}, {"job": "another-job"}):
            with self.subTest(changed=changed):
                job = "base-" + str(self.planner_calls)
                calls = Counter()

                async def drive(_backend, _ws, prompt, *_args, **_kwargs):
                    index = int(prompt.split("[Sub-task ")[1].split("/")[0])
                    calls[index] += 1
                    if index == 2 and calls[index] == 1:
                        raise asyncio.CancelledError
                    return AgentResult(text="evidence"), False

                before = self.planner_calls
                with self.assertRaises(asyncio.CancelledError):
                    await self.invoke(drive, job=job)
                options = {"job": job, **changed}
                await self.invoke(drive, **options)
                self.assertEqual(self.planner_calls, before + 2)
                self.assertEqual(calls[1], 2)

    async def test_many_same_tier_leftovers_run_as_one_cohesive_worker(self):
        self.plan = "[PLAN]\n1. [low] enumerate\n[/PLAN]"
        prompts = []

        async def drive(_backend, _ws, prompt, *_args, **_kwargs):
            prompts.append(prompt)
            if len(prompts) == 1:
                return AgentResult(text="enumerated\n" + "\n".join(
                    f"[followup:high] repair exact target {index}" for index in range(21))), False
            for index in range(21):
                self.assertIn(f"repair exact target {index}", prompt)
            return AgentResult(text="all repaired"), False

        await self.invoke(drive)
        self.assertEqual(len(prompts), 2)


class FollowupScopeTests(unittest.TestCase):
    def test_dedup_preserves_literal_whitespace_case_and_distinct_targets(self):
        known = {flexible.task_identity(flexible.Subtask("high", "repair target A"))}
        report = ("[followup:high] Please repair   target A\n"
                  "[followup:high] repair target a\n"
                  "[followup:high] preserve `a  b`\n"
                  "[followup:high] preserve `a b`")
        tasks, overflow, added = flexible.coalesce_followups(report, known)
        self.assertEqual(len(tasks), 1)
        self.assertFalse(overflow)
        self.assertEqual(sum(map(len, added.values())), 3)
        self.assertIn("repair target a", tasks[0].instruction)
        self.assertIn("`a  b`", tasks[0].instruction)
        self.assertIn("`a b`", tasks[0].instruction)

    def test_alternating_tiers_preserve_order_and_explicit_unstarted_scope(self):
        lines = [f"[followup:{flexible.TIERS[index % 3]}] remaining scope {index}" for index in range(8)]
        tasks, overflow, _ = flexible.coalesce_followups("\n".join(lines), set())
        self.assertEqual(len(tasks), flexible.MAX_FOLLOWUP_TASKS_PER_REPORT)
        self.assertEqual([task.instruction for task in tasks + overflow],
                         [f"remaining scope {index}" for index in range(8)])
        self.assertEqual([task.tier for task in tasks + overflow],
                         [flexible.TIERS[index % 3] for index in range(8)])
