"""Cached parallel results do not spend a new worker allowance."""

from collections import Counter
import tempfile
import unittest
from unittest import mock

from Cozter import agent, execution_state, workspace
from Cozter.backends_agent.base import AgentResult


class FlexibleCachedWorkerBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.tiers = {
            "low": ("llama", "local-low"),
            "mid": ("zai", "glm-5.3"),
            "high": ("meta", "muse-spark-1.3"),
        }
        self.enterContext(mock.patch.object(workspace, "get_flexible_run_config", return_value=self.tiers))
        self.plan = (
            "[PLAN]\n1. [low] [parallel-read] Inspect dependency A\n"
            "2. [mid] [parallel-read] Inspect dependency B\n[/PLAN]"
        )
        self.calls = Counter()
        self.internal_labels = []
        self.record = None

    async def internal(self, _backend, _ws, _prompt, _model, *, label, **_kwargs):
        self.internal_labels.append(label)
        return self.plan if label == "Flexible planner" else "Dependency evidence merged."

    async def drive(self, backend, _ws, prompt, *_args, **_kwargs):
        if backend.name == "llama":
            self.calls["A"] += 1
            if self.calls["A"] == 1:
                return AgentResult(text=(
                    "[LEASE]\ntier: high\nreason: Resolve an ambiguous dependency from existing source.\n"
                    "task: Inspect existing dependency A declarations without writes or execution.\n[/LEASE]"
                ), usage={"input_tokens": 1}), False
            self.assertIn("A helper evidence", prompt)
            return AgentResult(text="A inspection complete.", usage={"input_tokens": 4}), False
        if backend.name == "meta":
            self.calls["helper"] += 1
            return AgentResult(text="A helper evidence: cache.cpp:84 owns the key.", usage={"input_tokens": 3}), False
        index = int(prompt.split("[Sub-task ", 1)[1].split("/", 1)[0])
        if index == 2:
            self.calls["B"] += 1
            return AgentResult(text="B inspection complete.", usage={"input_tokens": 2}), False
        self.calls["writer"] += 1
        self.assertIn("A inspection complete", prompt)
        self.assertIn("B inspection complete", prompt)
        return AgentResult(text="Dependent repair complete.", usage={"input_tokens": 5}), False

    async def invoke(self):
        async with execution_state.execution_scope(self.ws, "cached-budget", "Inspect both dependencies"):
            with (
                mock.patch.object(agent, "run_internal_backend", self.internal),
                mock.patch.object(agent, "_drive_backend", self.drive),
                mock.patch.object(agent.config, "get_flexible_max_parallel_workers", return_value=2),
            ):
                try:
                    return await agent._run_flexible(
                        "Inspect both dependencies", "Inspect both dependencies", self.ws,
                        approval="full", effort=70, collaborative=False,
                        summary_backend_name="meta", summary_model="summary",
                        on_event=None, inject_queue=None, injected=[],
                    )
                finally:
                    self.record = await execution_state.record("flexible_plan")

    async def test_cached_sibling_finishes_after_last_actual_worker_uses_allowance(self):
        with agent.turn_budget() as budget:
            budget.max_workers = 4
            result, restarting = await self.invoke()

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(self.calls, {"A": 2, "B": 1, "helper": 1})
        self.assertEqual(budget.workers_started, 4)
        self.assertEqual(result.usage, {"input_tokens": 10})
        self.assertEqual(self.record.get("usage_processed_audit"), {"input_tokens": 10})
        self.assertEqual(self.record.get("cursor"), 2)
        self.assertFalse(self.record.get("active"))
        self.assertEqual(self.internal_labels, ["Flexible planner", "Flexible merge"])

    async def test_cached_sibling_is_consumed_before_unstarted_dependency_needs_new_allowance(self):
        self.plan = self.plan.replace("[/PLAN]", "3. [mid] Repair dependencies using the inspections\n[/PLAN]")
        with agent.turn_budget() as budget:
            budget.max_workers = 4
            with self.assertRaises(agent._TurnBudgetExhaustedError):
                await self.invoke()

        self.assertEqual(self.calls, {"A": 2, "B": 1, "helper": 1})
        self.assertEqual(budget.workers_started, 4)
        self.assertEqual(self.record.get("cursor"), 2)
        self.assertEqual(len(self.record.get("reports")), 2)
        self.assertEqual(self.record.get("usage_processed_audit"), {"input_tokens": 10})

        with agent.turn_budget() as resumed:
            resumed.max_workers = 1
            result, restarting = await self.invoke()

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(self.calls, {"A": 2, "B": 1, "helper": 1, "writer": 1})
        self.assertEqual(resumed.workers_started, 1)
        self.assertEqual(self.record.get("usage_processed_audit"), {"input_tokens": 15})
        self.assertEqual(self.record.get("cursor"), 3)
        self.assertEqual(self.internal_labels, ["Flexible planner", "Flexible merge"])
