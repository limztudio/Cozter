"""Cheap fallback work escalates with evidence while saved native owners stay pinned."""

import asyncio
from collections import Counter
import tempfile
import unittest
from unittest import mock
import uuid

from Cozter import agent, execution_state, flexible, session, workspace
from Cozter.backends_agent.base import AgentResult


class FlexibleLowDefaultTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.sid = session.create_session(self.ws, name="Low default native ownership")["id"]
        self.request = "Repair the bounded cache eviction bug, verify the regression, and preserve all original scope."
        self.tiers = {
            "low": ("codex", "gpt-6-luna"),
            "mid": ("codex", "gpt-6-sol"),
            "high": ("meta", "muse-spark-1.3"),
        }
        self.enterContext(mock.patch.object(workspace, "get_flexible_run_config", return_value=self.tiers))
        workspace.set_flexible_reasoning_effort(self.ws, "low", 10)
        workspace.set_flexible_reasoning_effort(self.ws, "mid", 60)
        self.plan_output = "The planner response has no usable plan."
        self.internal_labels = []
        self.calls = []
        self.native_ids = {model: str(uuid.uuid4()) for _backend, model in self.tiers.values()}
        self.record = None

    async def internal(self, _backend, _ws, _prompt, _model, *, label, **_kwargs):
        self.internal_labels.append(label)
        return self.plan_output if label == "Flexible planner" else "All original work verified."

    async def invoke(self, drive):
        async with execution_state.execution_scope(self.ws, "low-default", self.request):
            execution_state.bind_conversation(self.sid)
            with (
                mock.patch.object(agent, "run_internal_backend", self.internal),
                mock.patch.object(agent, "_drive_backend_impl", drive),
            ):
                try:
                    return await agent._run_flexible(
                        self.request, self.request, self.ws, approval="full", effort=30,
                        collaborative=False, summary_backend_name="meta", summary_model="summary",
                        on_event=None, inject_queue=None, injected=[],
                        session_data=session.load_session(self.ws, self.sid),
                    )
                finally:
                    self.record = await execution_state.record("flexible_plan")

    async def accept_native(self, prompt, model, approval, effort, callback):
        record = execution_state.current_run()
        self.assertIsNotNone(record)
        previous = record.get("native_session_id")
        native_id = previous or self.native_ids[model]
        await record.save(native_session_id=native_id, native_tools_started=True)
        if callback is not None:
            await callback()
        self.calls.append({"model": model, "approval": approval, "effort": effort,
                           "native_id": native_id, "previous_id": previous,
                           "key": record.key, "prompt": prompt})

    async def test_malformed_planner_starts_low_then_resumes_exact_mid_owner_before_dependent_work(self):
        attempts = Counter()

        async def drive(_backend, _ws, prompt, model, approval, *, effort, on_native_activity=None, **_kwargs):
            await self.accept_native(prompt, model, approval, effort, on_native_activity)
            attempts[model] += 1
            if model == "gpt-6-luna" and attempts[model] == 1:
                self.assertIn(self.request, prompt)
                return AgentResult(text=(
                    "PARTIAL: low-attempt-proof retained; the local regression still fails.\n"
                    "[followup:low] Verify dependent cache export after the repair\n"
                    "[ESCALATE]\nreason: Linked cache and registry invariants remain unresolved.\n"
                    "evidence: Local repair attempted; cache regression exited 1 with the old eviction order.\n"
                    "remaining: Repair the remaining invariant and verify the complete original cache scope.\n"
                    "[/ESCALATE]"
                )), False
            if model == "gpt-6-sol":
                if attempts[model] == 1:
                    self.assertIn(self.request, prompt)
                    self.assertIn("low-attempt-proof", prompt)
                    self.assertIn("remaining invariant", prompt)
                    self.assertIn("Verify dependent cache export", prompt)
                    raise asyncio.CancelledError
                self.assertIn("Continue the current job from this native conversation", prompt)
                return AgentResult(text="Repair, regression, and retained dependent cache export verified."), False
            self.fail(f"Unexpected replacement owner: {model}")

        with self.assertRaises(asyncio.CancelledError):
            await self.invoke(drive)
        self.assertEqual(self.record.get("cursor"), 0)
        self.assertEqual(self.record.get("queue")[0]["tier"], "mid")
        self.assertEqual(self.record.get("reports"), [])
        self.assertIn("low-attempt-proof", str(self.record.get("routing_evidence")))
        self.assertIn("Verify dependent cache export", str(self.record.get("interim_followups")))
        self.assertIn("Verify dependent cache export", str(self.record.get("owner_obligations")))
        self.plan_output = "[PLAN]\n1. [high] Repeat the entire audit\n[/PLAN]"

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual([row["model"] for row in self.calls],
                         ["gpt-6-luna", "gpt-6-sol", "gpt-6-sol"])
        self.assertEqual([row["effort"] for row in self.calls], [10, 60, 60])
        self.assertEqual({row["approval"] for row in self.calls}, {"full"})
        self.assertNotEqual(self.calls[0]["native_id"], self.calls[1]["native_id"])
        self.assertEqual(self.calls[1]["native_id"], self.calls[2]["native_id"])
        self.assertEqual(self.calls[1]["key"], self.calls[2]["key"])
        self.assertEqual(self.record.get("cursor"), 1)
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_previously_graded_mid_checkpoint_stays_mid_after_default_changes(self):
        async def drive(_backend, _ws, prompt, model, approval, *, effort, on_native_activity=None, **_kwargs):
            await self.accept_native(prompt, model, approval, effort, on_native_activity)
            if len(self.calls) == 1:
                raise asyncio.CancelledError
            return AgentResult(text="Previously graded linked invariants repaired and verified."), False

        with mock.patch.object(flexible, "FALLBACK_TIER", "mid"):
            with self.assertRaises(asyncio.CancelledError):
                await self.invoke(drive)
        self.assertEqual(self.record.get("queue")[0]["tier"], "mid")

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual([row["model"] for row in self.calls], ["gpt-6-sol", "gpt-6-sol"])
        self.assertEqual(self.calls[0]["native_id"], self.calls[1]["native_id"])
        self.assertEqual(self.calls[0]["key"], self.calls[1]["key"])
        self.assertEqual(self.internal_labels, ["Flexible planner"])
