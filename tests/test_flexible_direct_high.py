"""Direct high-tier plans keep native ownership without mandatory lower-tier work."""

import asyncio
import unittest
import uuid

from Cozter import workspace
from Cozter.backends_agent.base import AgentResult
from Cozter.tests import test_flexible_low_default as low_fixture


class FlexibleDirectHighTests(unittest.IsolatedAsyncioTestCase):
    internal = low_fixture.FlexibleLowDefaultTests.internal
    invoke = low_fixture.FlexibleLowDefaultTests.invoke
    accept_native = low_fixture.FlexibleLowDefaultTests.accept_native

    async def asyncSetUp(self):
        await low_fixture.FlexibleLowDefaultTests.asyncSetUp(self)
        self.tiers["high"] = ("codex", "gpt-6.1-sol")
        self.native_ids["gpt-6.1-sol"] = str(uuid.uuid4())
        workspace.set_flexible_reasoning_effort(self.ws, "high", 100)
        self.request = "Design and implement a coupled cache/registry transaction, balancing throughput and correctness."
        self.plan_output = (
            "[OWNER]\ntier: high\nreason: Design and implementation share coupled invariants\n"
            "task: Design the coupled cache/registry transaction, implement and verify its tradeoffs\n[/OWNER]"
        )

    async def test_direct_high_resumes_exact_native_owner_without_lower_tier_probation(self):
        async def drive(_backend, _ws, prompt, model, approval, *, effort, on_native_activity=None, **_kwargs):
            await self.accept_native(prompt, model, approval, effort, on_native_activity)
            self.assertEqual(model, "gpt-6.1-sol")
            if len(self.calls) == 1:
                self.assertIn(self.request, prompt)
                self.assertIn("coupled cache/registry transaction", prompt)
                raise asyncio.CancelledError
            self.assertIn("Continue the current job from this native conversation", prompt)
            self.assertNotIn(self.request, prompt)
            return AgentResult(text="Transaction design implemented; throughput and correctness verified."), False

        with self.assertRaises(asyncio.CancelledError):
            await self.invoke(drive)
        self.assertEqual(self.record.get("cursor"), 0)
        self.assertEqual(self.record.get("queue")[0]["tier"], "high")
        self.assertEqual(self.record.get("reports"), [])
        self.plan_output = "[PLAN]\n1. [low] Repeat initial inspection\n[/PLAN]"

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual([row["model"] for row in self.calls], ["gpt-6.1-sol", "gpt-6.1-sol"])
        self.assertEqual([row["effort"] for row in self.calls], [100, 100])
        self.assertEqual(self.calls[0]["native_id"], self.calls[1]["native_id"])
        self.assertEqual(self.calls[0]["key"], self.calls[1]["key"])
        self.assertEqual(self.calls[1]["previous_id"], self.calls[0]["native_id"])
        self.assertEqual(self.record.get("cursor"), 1)
        self.assertFalse(self.record.get("active"))
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_explicit_low_mid_high_plan_preserves_each_native_model_and_effort(self):
        self.plan_output = (
            "[PLAN]\n1. [low] Apply the bounded configuration change using existing patterns\n"
            "2. [mid] Implement the understood multi-part repair using the established design\n"
            "3. [high] Resolve coupled transaction invariants and verify throughput/correctness tradeoffs\n[/PLAN]"
        )

        async def drive(_backend, _ws, prompt, model, approval, *, effort, on_native_activity=None, **_kwargs):
            await self.accept_native(prompt, model, approval, effort, on_native_activity)
            return AgentResult(text=f"Verified planned work on {model}."), False

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual([row["model"] for row in self.calls], ["gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol"])
        self.assertEqual([row["effort"] for row in self.calls], [10, 60, 100])
        self.assertEqual({row["approval"] for row in self.calls}, {"full"})
        self.assertEqual(len({row["native_id"] for row in self.calls}), 3)
        self.assertEqual([row["tier"] for row in self.record.get("queue")], ["low", "mid", "high"])
        self.assertEqual(self.internal_labels, ["Flexible planner", "Flexible merge"])
