"""A bounded helper can change tiers while the parent keeps its task and state."""

import asyncio
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from Cozter import agent, execution_state, flexible, workspace
from Cozter.backends_agent.base import AgentResult


def lease(tier, tag="cache-inspection"):
    return (
        "[LEASE]\n"
        f"tier: {tier}\n"
        "reason: A separate read-only view will resolve this bounded question.\n"
        f"task: Inspect existing source and logs for {tag}; report exact paths and findings without edits or execution.\n"
        "[/LEASE]"
    )


class FlexibleTierLeaseTests(unittest.IsolatedAsyncioTestCase):
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
        self.plan = "[PLAN]\n1. [high] Repair and verify the original cache dependency\n[/PLAN]"
        self.internal_labels = []
        self.record = None

    async def internal(self, _backend, _ws, _prompt, _model, *, label, **_kwargs):
        self.internal_labels.append(label)
        return self.plan if label == "Flexible planner" else "Requested work verified."

    async def invoke(self, drive, *, job="lease-job", approval="full"):
        async with execution_state.execution_scope(self.ws, job, "Repair the original cache dependency"):
            with (
                mock.patch.object(agent, "run_internal_backend", self.internal),
                mock.patch.object(agent, "_drive_backend", drive),
            ):
                try:
                    return await agent._run_flexible(
                        "Original request", "Original request", self.ws,
                        approval=approval, effort=30, collaborative=False,
                        summary_backend_name="meta", summary_model="summary",
                        on_event=None, inject_queue=None, injected=[],
                    )
                finally:
                    self.record = await execution_state.record("flexible_plan")

    async def test_high_parent_leases_low_and_resumes_same_owner_with_findings(self):
        calls = []
        roles = []

        async def drive(backend, _ws, prompt, model, approval, *, effort, **_kwargs):
            calls.append((backend.name, model, approval, effort))
            roles.append((await execution_state.record("lease-role-probe")).key)
            if len(calls) == 1:
                return AgentResult(text=lease("low"), usage={"input_tokens": 2}), False
            if backend.name == "llama":
                self.assertIn("cache-inspection", prompt)
                return AgentResult(text="Cheap inspection finding: cache.cpp:84 holds the stale key.",
                                   usage={"input_tokens": 3}), False
            self.assertIn("original cache dependency", prompt)
            self.assertIn("cache.cpp:84 holds the stale key", prompt)
            return AgentResult(text="Original cache dependency repaired and verified.",
                               usage={"input_tokens": 5}), False

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(calls, [
            ("meta", "muse-spark-1.3", "full", 100),
            ("llama", "local-low", "confirm", 10),
            ("meta", "muse-spark-1.3", "full", 100),
        ])
        self.assertEqual(roles[0], roles[2])
        self.assertNotEqual(roles[0], roles[1])
        self.assertEqual(result.usage, {"input_tokens": 10})
        self.assertEqual(self.internal_labels, ["Flexible planner"])
        self.assertEqual(self.record.get("queue")[0]["tier"], "high")

    async def test_low_parent_can_borrow_high_reasoning_without_surrendering_task(self):
        self.plan = "[PLAN]\n1. [low] Apply the specified cache rename and verify it\n[/PLAN]"
        calls = []

        async def drive(backend, _ws, prompt, _model, approval, *, effort, **_kwargs):
            calls.append((backend.name, approval, effort))
            if len(calls) == 1:
                return AgentResult(text=lease("high", "ambiguous-lock-order")), False
            if backend.name == "meta":
                return AgentResult(text="High helper finding: the existing lock order is registry then cache."), False
            self.assertIn("registry then cache", prompt)
            self.assertIn("specified cache rename", prompt)
            return AgentResult(text="Specified rename applied and verified."), False

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(calls, [("llama", "full", 10), ("meta", "confirm", 100), ("llama", "full", 10)])
        self.assertEqual(self.record.get("queue")[0]["tier"], "low")
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_interrupted_helper_resumes_without_replaying_parent_yield(self):
        calls = Counter()
        order = []

        async def drive(backend, _ws, prompt, *_args, **_kwargs):
            calls[backend.name] += 1
            order.append(backend.name)
            if backend.name == "meta" and calls["meta"] == 1:
                return AgentResult(text=lease("low", "durable-helper-question")), False
            if backend.name == "llama":
                if calls["llama"] == 1:
                    raise asyncio.CancelledError
                return AgentResult(text="Durable helper finding: cache.cpp:44 owns the key."), False
            self.assertIn("cache.cpp:44 owns the key", prompt)
            return AgentResult(text="Original dependency verified using retained helper evidence."), False

        with self.assertRaises(asyncio.CancelledError):
            await self.invoke(drive)
        self.assertEqual(self.record.get("cursor"), 0)
        self.assertEqual(len(self.record.get("tier_leases")["0"]), 1)

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(order, ["meta", "llama", "llama", "meta"])
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_completed_helper_is_not_repeated_when_parent_resume_is_interrupted(self):
        calls = Counter()

        async def drive(backend, _ws, prompt, *_args, **_kwargs):
            calls[backend.name] += 1
            if backend.name == "llama":
                return AgentResult(text="Completed helper finding: source has exactly six gate participants."), False
            if calls["meta"] == 1:
                return AgentResult(text=lease("low", "gate-participant-count")), False
            self.assertIn("exactly six gate participants", prompt)
            if calls["meta"] == 2:
                raise asyncio.CancelledError
            return AgentResult(text="Original gate repair verified."), False

        with self.assertRaises(asyncio.CancelledError):
            await self.invoke(drive)
        self.assertIn("exactly six gate participants", str(self.record.get("tier_leases")))

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(calls, {"meta": 3, "llama": 1})
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_helper_lease_request_cannot_spawn_another_helper(self):
        nested_requests = {
            "lease": lease("mid", "nested-helper-must-not-run"),
            "escalation": (
                "[ESCALATE]\nreason: Helper could not resolve the inspection question.\n"
                "evidence: Existing cache.cpp:84 lacks the expected key declaration.\n"
                "remaining: Inspect the key declaration in another module.\n[/ESCALATE]"
            ),
        }
        for kind, nested in nested_requests.items():
            with self.subTest(kind=kind):
                calls = []

                async def drive(backend, *_args, **_kwargs):
                    calls.append(backend.name)
                    if len(calls) == 1:
                        return AgentResult(text=lease("low", "outer-helper-question")), False
                    if backend.name == "llama":
                        return AgentResult(text=nested), False
                    return AgentResult(text="Original parent finished its bounded task."), False

                result, restarting = await self.invoke(drive, job=f"helper-protocol-{kind}")

                self.assertFalse(restarting)
                self.assertFalse(result.error)
                self.assertEqual(calls, ["meta", "llama", "meta"])
                self.assertEqual(len(self.record.get("tier_leases")["0"]), 1)

    async def test_helper_prerequisite_stays_pending_and_does_not_promote_parent(self):
        self.plan = "[PLAN]\n1. [low] Apply the specified cache rename and verify it\n[/PLAN]"
        calls = Counter()
        order = []

        async def drive(backend, _ws, prompt, *_args, **_kwargs):
            calls[backend.name] += 1
            order.append(backend.name)
            if backend.name == "llama" and calls["llama"] == 1:
                return AgentResult(text=lease("high", "required-local-evidence")), False
            if backend.name == "meta":
                if calls["meta"] == 1:
                    return AgentResult(text="Required local evidence is temporarily unavailable.",
                                       continuation_reason="prerequisite"), False
                return AgentResult(text="Available helper evidence: key identity follows the existing registry."), False
            self.assertIn("key identity follows the existing registry", prompt)
            return AgentResult(text="Specified rename verified with helper evidence."), False

        first, restarting = await self.invoke(drive)
        self.assertFalse(restarting)
        self.assertEqual(first.continuation_reason, "prerequisite")
        self.assertEqual(self.record.get("cursor"), 0)
        self.assertEqual(self.record.get("queue")[0]["tier"], "low")

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(order, ["llama", "meta", "meta", "llama"])
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_lease_cannot_broaden_deny_permission(self):
        approvals = []

        async def drive(backend, _ws, _prompt, _model, approval, **_kwargs):
            approvals.append(approval)
            if len(approvals) == 1:
                return AgentResult(text=lease("low", "denied-tool-inspection")), False
            return AgentResult(text="Read-only model response without additional tool permission."), False

        result, restarting = await self.invoke(drive, approval="deny")

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(approvals, ["deny", "deny", "deny"])

    async def test_parallel_reader_lease_keeps_completed_sibling_while_helper_waits(self):
        self.plan = (
            "[PLAN]\n1. [high] [parallel-read] Inspect cache dependency A\n"
            "2. [mid] [parallel-read] Inspect cache dependency B\n"
            "3. [mid] Repair and verify the cache findings\n[/PLAN]"
        )
        sibling_done = asyncio.Event()
        helper_started = asyncio.Event()
        release_helper = asyncio.Event()
        calls = Counter()

        async def drive(backend, _ws, prompt, _model, approval, **_kwargs):
            if backend.name == "llama":
                calls["helper"] += 1
                saved = await execution_state.current_execution().record({
                    "record": "flexible_plan", "worker": "main",
                })
                self.assertIn("B finding retained", saved.get("worker_outputs")["1"]["text"])
                helper_started.set()
                await release_helper.wait()
                return AgentResult(text="Helper finding: A dependency follows cache.cpp:84.",
                                   usage={"input_tokens": 5}), False
            index = 1 if backend.name == "meta" else int(prompt.split("[Sub-task ", 1)[1].split("/", 1)[0])
            calls[index] += 1
            if index == 2:
                self.assertEqual(approval, "confirm")
                sibling_done.set()
                return AgentResult(text="B finding retained: existing registry order is valid.",
                                   usage={"input_tokens": 3}), False
            if index == 1:
                self.assertEqual(approval, "confirm")
                if calls[index] == 1:
                    await asyncio.wait_for(sibling_done.wait(), 2)
                    return AgentResult(text=lease("low", "dependency-A-source"),
                                       usage={"input_tokens": 2}), False
                self.assertIn("A dependency follows cache.cpp:84", prompt)
                return AgentResult(text="A inspection verified using helper evidence.",
                                   usage={"input_tokens": 7}), False
            self.assertEqual(approval, "full")
            self.assertIn("A inspection verified", prompt)
            self.assertIn("B finding retained", prompt)
            return AgentResult(text="Cache findings repaired and verified.", usage={"input_tokens": 11}), False

        running = asyncio.create_task(self.invoke(drive))
        try:
            await asyncio.wait_for(helper_started.wait(), 3)
            self.assertEqual(calls[2], 1)
            self.assertFalse(running.done())
        finally:
            release_helper.set()
            result, restarting = await asyncio.wait_for(running, 3)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(calls, {1: 2, 2: 1, 3: 1, "helper": 1})
        self.assertEqual(result.usage, {"input_tokens": 28})
        self.assertEqual(self.record.get("usage_processed_audit"), {"input_tokens": 28})
        self.assertEqual(self.internal_labels, ["Flexible planner", "Flexible merge"])

    async def test_partial_helper_usage_and_attachment_survive_prerequisite_resume(self):
        artifact = Path(self.ws, "existing-helper-proof.txt")
        artifact.write_text("Existing read-only inspection evidence", encoding="utf-8")
        marker = "[[attach: existing-helper-proof.txt]]"
        calls = Counter()

        async def drive(backend, _ws, prompt, *_args, **_kwargs):
            calls[backend.name] += 1
            if backend.name == "meta" and calls["meta"] == 1:
                return AgentResult(text=lease("low", "partial-helper-evidence"), usage={"input_tokens": 2}), False
            if backend.name == "llama":
                if calls["llama"] == 1:
                    return AgentResult(
                        text=f"PARTIAL: existing evidence identified; required local prerequisite is missing.\n{marker}",
                        usage={"input_tokens": 3}, continuation_reason="prerequisite",
                    ), False
                return AgentResult(text="Helper resumed: existing evidence confirms the key owner.",
                                   usage={"input_tokens": 7}), False
            self.assertIn("existing evidence confirms the key owner", prompt)
            return AgentResult(text="Original dependency verified.", usage={"input_tokens": 11}), False

        first, restarting = await self.invoke(drive)
        self.assertFalse(restarting)
        self.assertEqual(first.continuation_reason, "prerequisite")
        self.assertEqual(first.usage, {"input_tokens": 5})
        self.assertIn(marker, first.text)
        self.assertEqual(self.record.get("usage_processed_audit"), {"input_tokens": 5})

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(calls, {"meta": 2, "llama": 2})
        self.assertEqual(result.usage, {"input_tokens": 18})
        self.assertEqual(self.record.get("usage_processed_audit"), {"input_tokens": 23})
        self.assertIn(marker, result.text)
        self.assertEqual(artifact.read_text(encoding="utf-8"), "Existing read-only inspection evidence")

    async def test_followup_discovered_before_lease_runs_once_after_parent_finishes(self):
        followup = "[followup:mid] Update the related cache fixture and verify it"
        calls = Counter()

        async def drive(backend, _ws, prompt, *_args, **_kwargs):
            calls[backend.name] += 1
            if backend.name == "meta":
                if calls["meta"] == 1:
                    return AgentResult(text=followup + "\n" + lease("low", "retained-followup-proof")), False
                return AgentResult(text="Original cache dependency repaired and verified.\n" + followup), False
            if backend.name == "llama":
                return AgentResult(text="Helper findings verified against the existing cache source."), False
            self.assertIn("related cache fixture", prompt)
            self.assertIn("Original cache dependency repaired and verified", prompt)
            return AgentResult(text="Related fixture updated and verified."), False

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(calls, {"meta": 2, "llama": 1, "zai": 1})
        self.assertEqual(len(self.record.get("queue")), 2)
        self.assertEqual(self.record.get("cursor"), 2)
        self.assertEqual(self.internal_labels, ["Flexible planner", "Flexible merge"])

    async def test_fresh_identical_helper_question_rechecks_inputs_changed_by_parent(self):
        source = Path(self.ws, "cache-state.txt")
        source.write_text("before-parent-edit", encoding="utf-8")
        request = lease("low", "current-cache-state")
        parent_calls = 0
        helper_reads = []

        async def drive(backend, _ws, prompt, *_args, **_kwargs):
            nonlocal parent_calls
            if backend.name == "llama":
                current = source.read_text(encoding="utf-8")
                helper_reads.append(current)
                return AgentResult(text=f"Helper observation: cache state is {current}."), False
            parent_calls += 1
            if parent_calls == 1:
                return AgentResult(text=request), False
            if parent_calls == 2:
                self.assertIn("cache state is before-parent-edit", prompt)
                source.write_text("after-parent-edit", encoding="utf-8")
                return AgentResult(text=request), False
            self.assertIn("cache state is after-parent-edit", prompt)
            self.assertNotIn("cache state is before-parent-edit", prompt)
            return AgentResult(text="Original cache dependency verified against current state."), False

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(parent_calls, 3)
        self.assertEqual(helper_reads, ["before-parent-edit", "after-parent-edit"])
        self.assertEqual(len(self.record.get("tier_leases")["0"]), 2)
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_three_lease_bound_holds_dependent_work_and_survives_resume(self):
        self.plan = (
            "[PLAN]\n1. [high] Repair and verify the original cache dependency\n"
            "2. [mid] Commit the verified repair\n[/PLAN]"
        )
        calls = Counter()

        async def drive(backend, *_args, **_kwargs):
            calls[backend.name] += 1
            if backend.name == "meta":
                return AgentResult(text=lease("low", f"distinct-helper-question-{calls['meta']}")), False
            self.assertEqual(backend.name, "llama", "dependent commit must remain held")
            return AgentResult(text="Bounded helper inspection completed."), False

        for _resume in range(2):
            result, restarting = await self.invoke(drive)
            self.assertFalse(restarting)
            self.assertTrue(result.error)
            self.assertFalse(result.completion_verified)
            self.assertTrue(self.record.get("active"))
            self.assertEqual(self.record.get("cursor"), 0)
            self.assertEqual(len(self.record.get("tier_leases")["0"]), 3)

        self.assertEqual(calls, {"meta": 4, "llama": 3})
        self.assertEqual(self.internal_labels, ["Flexible planner"])
