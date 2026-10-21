"""Tier changes keep unfinished scope, completed siblings, and durable evidence."""

import asyncio
from collections import Counter
import tempfile
import unittest
from unittest import mock

from Cozter import agent, execution_state, flexible, workspace
from Cozter.backends_agent.base import AgentResult


def escalation(tag):
    return (
        f"PARTIAL: {tag}; verified repair did not resolve the failure.\n"
        "[ESCALATE]\n"
        "reason: The bounded repair left an unresolved data dependency.\n"
        f"evidence: {tag}; tests/test_cache.py::test_eviction exited 1 after the repair.\n"
        "remaining: Fix the remaining eviction dependency and rerun the failing test.\n"
        "[/ESCALATE]"
    )


def task_index(prompt):
    return int(prompt.split("[Sub-task ", 1)[1].split("/", 1)[0])


class FlexibleRoutingEscalationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.tiers = {
            "low": ("llama", "local-low"),
            "mid": ("zai", "glm-5.3"),
            "high": ("meta", "muse-spark-1.3"),
        }
        self.enterContext(mock.patch.object(workspace, "get_flexible_run_config", return_value=self.tiers))
        for tier, effort in zip(flexible.TIERS, (10, 60, 100)):
            workspace.set_flexible_reasoning_effort(self.ws, tier, effort)
        self.plan = "[PLAN]\n1. [mid] Repair the bounded cache bug and verify it\n[/PLAN]"
        self.internal_labels = []
        self.merge_prompts = []
        self.record = None

    async def internal(self, _backend, _ws, prompt, _model, *, label, **_kwargs):
        self.internal_labels.append(label)
        if label == "Flexible planner":
            return self.plan
        self.merge_prompts.append(prompt)
        return "Verified requested work complete."

    async def invoke(self, drive, *, job="routing-job"):
        async with execution_state.execution_scope(self.ws, job, "Repair and verify the requested cache work"):
            with (
                mock.patch.object(agent, "run_internal_backend", self.internal),
                mock.patch.object(agent, "_drive_backend", drive),
            ):
                try:
                    return await agent._run_flexible(
                        "Original request", "Original request", self.ws,
                        approval="full", effort=30, collaborative=False,
                        summary_backend_name="meta", summary_model="summary",
                        on_event=None, inject_queue=None, injected=[],
                    )
                finally:
                    self.record = await execution_state.record("flexible_plan")

    async def test_failed_repairs_move_one_tier_at_a_time_before_dependent_work(self):
        self.plan = (
            "[PLAN]\n1. [low] Repair and verify the cache eviction dependency\n"
            "2. [mid] Commit the verified cache repair\n[/PLAN]"
        )
        calls = []

        async def drive(backend, _ws, prompt, model, approval, *, effort, **_kwargs):
            index = task_index(prompt)
            calls.append((index, backend.name, model, effort, approval))
            if index == 1 and backend.name == "llama":
                return AgentResult(text=escalation("low-attempt-evidence"), usage={"input_tokens": 1}), False
            if index == 1 and backend.name == "zai":
                self.assertIn("low-attempt-evidence", prompt)
                self.assertIn("remaining eviction dependency", prompt)
                return AgentResult(text=escalation("mid-attempt-evidence"), usage={"input_tokens": 2}), False
            if index == 1:
                self.assertIn("mid-attempt-evidence", prompt)
                return AgentResult(text="High repair verified: eviction regression now passes.",
                                   usage={"input_tokens": 3}), False
            self.assertIn("eviction regression now passes", prompt)
            return AgentResult(text="Verified repair committed.", usage={"input_tokens": 4}), False

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertEqual(calls, [
            (1, "llama", "local-low", 10, "full"),
            (1, "zai", "glm-5.3", 60, "full"),
            (1, "meta", "muse-spark-1.3", 100, "full"),
            (2, "zai", "glm-5.3", 60, "full"),
        ])
        self.assertEqual(result.usage, {"input_tokens": 10})
        self.assertEqual(self.internal_labels, ["Flexible planner", "Flexible merge"])
        self.assertEqual(self.record.get("cursor"), 2)
        self.assertEqual(len(self.record.get("reports")), 2)

    async def test_explicit_mid_plan_keeps_its_backend_without_artificial_tiers_or_routing_calls(self):
        self.plan = "[PLAN]\n1. [mid] Resolve interacting cache and registry invariants, repair and verify\n[/PLAN]"
        calls = []

        async def drive(backend, _ws, prompt, model, _approval, *, effort, **_kwargs):
            calls.append((backend.name, model, effort))
            return AgentResult(text="Bounded cache repair and regression verified."), False

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(calls, [("zai", "glm-5.3", 60)])
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_quota_await_and_prerequisite_do_not_become_reasoning_escalations(self):
        self.plan = "[PLAN]\n1. [low] Inspect the existing cache report\n[/PLAN]"
        cases = {
            "quota": AgentResult(text=escalation("quota-report"), error="You've hit your monthly spend limit"),
            "await": AgentResult(text=escalation("await-report") + "\n[[await]]"),
            "prerequisite": AgentResult(text=escalation("desktop-missing"), continuation_reason="prerequisite"),
        }
        for label, value in cases.items():
            with self.subTest(label=label):
                calls = []

                async def drive(backend, *_args, **_kwargs):
                    calls.append(backend.name)
                    return value, False

                await self.invoke(drive, job=label)
                self.assertEqual(calls, ["llama"])
                self.assertEqual(self.record.get("queue")[0]["tier"], "low")

    async def test_promotion_is_durable_without_replanning_or_replaying_completed_work(self):
        self.plan = (
            "[PLAN]\n1. [low] Read the existing cache fixture\n"
            "2. [low] Repair and verify the cache eviction dependency\n"
            "3. [mid] Commit the verified cache repair\n[/PLAN]"
        )
        calls = Counter()
        interrupted = False

        async def drive(backend, _ws, prompt, *_args, **_kwargs):
            nonlocal interrupted
            index = task_index(prompt)
            calls[index, backend.name] += 1
            if index == 1:
                return AgentResult(text="Retained fixture evidence: expected count is eight."), False
            if index == 2 and backend.name == "llama":
                return AgentResult(text=escalation("durable-low-evidence")), False
            if index == 2:
                self.assertIn("Retained fixture evidence", prompt)
                self.assertIn("durable-low-evidence", prompt)
                if not interrupted:
                    interrupted = True
                    raise asyncio.CancelledError
                return AgentResult(text="Remaining eviction dependency repaired and verified."), False
            self.assertIn("dependency repaired and verified", prompt)
            return AgentResult(text="Verified repair committed."), False

        with self.assertRaises(asyncio.CancelledError):
            await self.invoke(drive)
        self.assertEqual(self.record.get("cursor"), 1)
        self.assertEqual(self.record.get("queue")[1]["tier"], "mid")
        self.assertEqual(len(self.record.get("reports")), 1)
        self.assertIn("durable-low-evidence", str(self.record.get("routing_evidence")))

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(calls, {(1, "llama"): 1, (2, "llama"): 1, (2, "zai"): 2, (3, "zai"): 1})
        self.assertEqual(self.internal_labels.count("Flexible planner"), 1)
        self.assertEqual(self.internal_labels.count("Flexible merge"), 1)
        self.assertEqual(self.record.get("cursor"), 3)

    async def test_promoted_parallel_reader_retains_its_completed_sibling(self):
        self.plan = (
            "[PLAN]\n1. [low] [parallel-read] Inspect cache dependency A\n"
            "2. [mid] [parallel-read] Inspect cache dependency B\n"
            "3. [mid] Repair and verify the cache findings\n[/PLAN]"
        )
        sibling_finished = asyncio.Event()
        calls = Counter()

        async def drive(backend, _ws, prompt, _model, approval, **_kwargs):
            index = task_index(prompt)
            calls[index, backend.name] += 1
            if index == 1 and backend.name == "llama":
                await asyncio.wait_for(sibling_finished.wait(), 2)
                return AgentResult(text=escalation("reader-low-evidence")), False
            if index == 2:
                self.assertEqual(approval, "confirm")
                sibling_finished.set()
                return AgentResult(text="Dependency B inspected once: source matches fixture."), False
            if index == 1:
                self.assertEqual(approval, "confirm")
                self.assertIn("reader-low-evidence", prompt)
                return AgentResult(text="Dependency A inspected after bounded reasoning failure."), False
            self.assertIn("Dependency A inspected", prompt)
            self.assertIn("Dependency B inspected once", prompt)
            self.assertEqual(approval, "full")
            return AgentResult(text="Cache findings repaired and verified."), False

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(calls, {(1, "llama"): 1, (1, "zai"): 1, (2, "zai"): 1, (3, "zai"): 1})
        self.assertEqual(self.internal_labels, ["Flexible planner", "Flexible merge"])

    async def test_high_failure_holds_dependent_work_and_does_not_retry_on_resume(self):
        self.plan = (
            "[PLAN]\n1. [high] Repair and verify the cache eviction dependency\n"
            "2. [low] Commit the verified cache repair\n[/PLAN]"
        )
        calls = []

        async def drive(backend, _ws, prompt, *_args, **_kwargs):
            calls.append((task_index(prompt), backend.name))
            return AgentResult(text=escalation("high-unresolved-evidence"), usage={"input_tokens": 7}), False

        for _resume in range(2):
            result, restarting = await self.invoke(drive)
            self.assertFalse(restarting)
            self.assertEqual(
                result.error,
                "Flexible reasoning escalation exhausted at high; the current task remains unfinished.",
            )
            self.assertFalse(result.completion_verified)
            self.assertIsNone(result.continuation_reason)
            self.assertIn("PARTIAL", result.text)
            self.assertIn("remaining eviction dependency", result.text)
            self.assertTrue(self.record.get("active"))
            self.assertEqual(self.record.get("cursor"), 0)
            self.assertEqual(self.record.get("usage_processed_audit"), {"input_tokens": 7})

        self.assertEqual(calls, [(1, "meta")])
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_followup_discovered_before_escalation_is_retained_without_duplicates(self):
        self.plan = "[PLAN]\n1. [low] Repair and verify the cache eviction dependency\n[/PLAN]"
        followup = "[followup:low] Update the related cache fixture and verify it"
        calls = []

        async def drive(backend, _ws, prompt, *_args, **_kwargs):
            index = task_index(prompt)
            calls.append((index, backend.name))
            if index == 1 and backend.name == "llama":
                return AgentResult(text=followup + "\n" + escalation("followup-before-escalation")), False
            if index == 1:
                return AgentResult(text="Eviction dependency repaired and verified.\n" + followup), False
            self.assertIn("related cache fixture", prompt)
            self.assertIn("Eviction dependency repaired and verified", prompt)
            return AgentResult(text="Related cache fixture updated and verified."), False

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(calls, [(1, "llama"), (1, "zai"), (2, "llama")])
        self.assertEqual(len(self.record.get("queue")), 2)
        self.assertEqual(self.record.get("cursor"), 2)
        self.assertEqual(self.internal_labels, ["Flexible planner", "Flexible merge"])

    async def test_read_only_http_worker_escalates_to_nonparallel_native_with_confirm_permission(self):
        self.plan = "[PLAN]\n1. [low] [parallel-read] Inspect the existing registry lock traces\n[/PLAN]"
        self.tiers["mid"] = ("copilot", "auto")
        calls = []

        async def drive(backend, _ws, prompt, _model, approval, **_kwargs):
            calls.append((backend.name, approval))
            if backend.name == "llama":
                return AgentResult(text=(
                    "[ESCALATE]\nreason: Existing traces leave an ambiguous lock dependency.\n"
                    "evidence: cache.log:84 records a timeout with registry held.\n"
                    "remaining: Inspect registry lock order and report exact source evidence.\n[/ESCALATE]"
                )), False
            self.assertIn("Inspection only", prompt)
            self.assertIn("cache.log:84", prompt)
            return AgentResult(text="Registry lock order inspected; source evidence retained."), False

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(calls, [("llama", "confirm"), ("copilot", "confirm")])
