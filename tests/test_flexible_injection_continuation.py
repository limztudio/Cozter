"""Accepted guidance keeps a durable plan when the bot folds it into the request."""

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


class FlexibleInjectionContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.object(workspace, "CONFIG_DIR", self.ws))
        self.enterContext(mock.patch.object(agent, "_run_post_turn_maintenance", mock.AsyncMock()))
        for tier in flexible.TIERS:
            workspace.set_flexible_backend_name(self.ws, tier, "meta")
        self.sid = session.create_session(self.ws, name="Injected continuation fixture")["id"]
        self.planner_calls = 0
        self.merge_prompts = []

    async def asyncTearDown(self):
        await agent._wait_for_pending_session_writes(self.ws)

    async def internal(self, _backend, _ws, prompt, _model, *, label, **_kwargs):
        if label == "Flexible planner":
            self.planner_calls += 1
            return "[PLAN]\n1. [low] first\n2. [mid] second\n3. [high] third\n[/PLAN]"
        self.merge_prompts.append(prompt)
        return "final verified response"

    async def invoke(self, drive, *, canonical="Original scope", injected=None, **kwargs):
        async with execution_state.execution_scope(self.ws, "fixture-job", canonical):
            with (mock.patch.object(agent, "run_internal_backend", self.internal),
                  mock.patch.object(agent, "_drive_backend", drive)):
                return await agent._run_flexible(
                    "transport context", "transport request", self.ws,
                    approval=kwargs.pop("approval", "full"), effort=kwargs.pop("effort", 100),
                    collaborative=False, summary_backend_name="meta",
                    summary_model=kwargs.pop("summary_model", "summary"),
                    on_event=None, inject_queue=None,
                    injected=[] if injected is None else injected, **kwargs,
                )

    async def check_real_bot(self, additions, *, truncated):
        original = "Repair all three parts and retain verification evidence."
        bot = ReplyDeliveryBot(self.ws, fail_final=False)
        bot.supports_transient_status = False
        entry = await persist_queue_entry(bot, "u1", original, "chat")
        bot._ensure_message_queue("u1").put_nowait((original, "chat", entry, False))
        await bot._save_job_checkpoint("u1", entry, {
            "state": "running", "workspace_path": self.ws,
            "request": original, "session_id": self.sid,
        })
        artifact = os.path.join(self.ws, "proof.txt")
        with open(artifact, "w", encoding="utf-8") as out:
            out.write("retained artifact")
        calls = Counter()
        roles = defaultdict(list)
        injected_once = False
        interrupted = set()
        checkpoints = []
        delivered = []
        fold_proofs = []
        real_merge = bot._merge_job_checkpoint

        def capture_checkpoint(previous, current, result):
            checkpoints.append(current)
            return real_merge(previous, current, result)

        async def drive(_backend, _ws, prompt, _model, _approval, **kwargs):
            nonlocal injected_once
            index = int(prompt.split("[Sub-task ")[1].split("/")[0])
            calls[index] += 1
            roles[index].append((await execution_state.record("role-probe")).key)
            saved = await execution_state.current_execution().record({
                "record": "flexible_plan", "worker": "main",
            })
            fold_proofs.append(saved.get("folded_signatures", []))
            if not injected_once:
                injected_once = True
                kwargs["injected"].extend(additions)
                return AgentResult(), True
            if index in (2, 3) and index not in interrupted:
                interrupted.add(index)
                raise agent._TurnBudgetExhaustedError("worker allowance reached")
            return AgentResult(
                text=f"Retained verification evidence {index}",
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
              mock.patch.object(bot, "_merge_job_checkpoint", capture_checkpoint),
              mock.patch.object(bot, "_send_result", deliver)):
            async with asyncio.timeout(10):
                await bot._drain_message_queue("u1")
        self.assertEqual(checkpoints[0]["accepted_additions_truncated"], truncated)
        self.assertEqual(fold_proofs[0], [])
        self.assertEqual(len(fold_proofs[1]), 2)
        self.assertTrue(all(proof == fold_proofs[1] for proof in fold_proofs[1:]))
        self.assertEqual(self.planner_calls, 2)  # Initial scope and newly accepted guidance only.
        self.assertEqual(calls, {1: 2, 2: 2, 3: 2})
        self.assertEqual(roles[2][0], roles[2][1])
        self.assertEqual(roles[3][0], roles[3][1])
        for index in range(1, 4):
            self.assertIn(f"Retained verification evidence {index}", self.merge_prompts[-1])
        self.assertEqual(len(delivered), 1)
        self.assertTrue(any(event.kind == "attachment" for event in delivered[0].events))
        self.assertEqual(bot._read_queue_file(), {})

    async def test_real_bot_keeps_plan_across_two_allowances_after_two_ordered_additions(self):
        await self.check_real_bot(["Keep exact `a  b` spacing.\n", "Then verify target B."], truncated=False)

    async def test_large_additions_recovered_from_session_keep_the_plan(self):
        await self.check_real_bot(["first:" + "a" * 40_000, "second:" + "b" * 40_000], truncated=True)

    async def test_new_guidance_order_text_or_settings_invalidate_a_folded_plan(self):
        base = "Original scope contains [Added while working]: literally."
        additions = ["Preserve `a  b`.", "Verify B.\n"]
        folded = base + "".join("\n\n[Added while working]:\n" + value for value in additions)
        cases = (
            {"canonical": folded, "injected": ["Now also verify C."]},
            {"canonical": base, "injected": list(reversed(additions))},
            {"canonical": base, "injected": ["Preserve `a b`.", additions[1]]},
            {"canonical": folded, "approval": "confirm"},
            {"canonical": folded, "effort": 80},
            {"canonical": folded, "summary_model": "different-summary"},
        )
        for changed in cases:
            with self.subTest(changed=changed):
                # Start a distinct execution to avoid reusing a preceding case.
                directory = self.enterContext(tempfile.TemporaryDirectory())
                old_ws, self.ws = self.ws, directory
                try:
                    for tier in flexible.TIERS:
                        workspace.set_flexible_backend_name(self.ws, tier, "meta")
                    fail = True

                    async def drive(_backend, _ws, _prompt, *_args, **_kwargs):
                        if fail:
                            raise asyncio.CancelledError
                        return AgentResult(text="verified"), False

                    before = self.planner_calls
                    with self.assertRaises(asyncio.CancelledError):
                        await self.invoke(drive, canonical=base, injected=additions)
                    fail = False
                    await self.invoke(drive, **changed)
                    self.assertEqual(self.planner_calls, before + 2)
                finally:
                    self.ws = old_ws
