"""An early specialist advises one durable owner without another planning turn."""

import asyncio
from collections import Counter
import json
import unittest
from unittest import mock

from Cozter import execution_state
from Cozter.backends_agent.base import AgentResult
from Cozter.tests import test_flexible_tier_leases as lease_fixture


def owner_plan(*, tier="low", specialists=None):
    questions = specialists if specialists is not None else [
        {"tier": "high", "reason": "Resolve the design ambiguity before implementation",
         "task": "Inspect the cache/registry invariant and recommend a design without edits or execution"},
    ]
    return (
        "[SPECIALISTS]\n" + json.dumps(questions) + "\n[/SPECIALISTS]\n"
        f"[OWNER]\ntier: {tier}\nreason: Keep implementation and verification with one owner\n"
        "task: Implement and verify the complete original cache task\n[/OWNER]"
    )


class FlexibleOwnerRuntimeTests(unittest.IsolatedAsyncioTestCase):
    internal = lease_fixture.FlexibleTierLeaseTests.internal
    invoke = lease_fixture.FlexibleTierLeaseTests.invoke

    async def asyncSetUp(self):
        await lease_fixture.FlexibleTierLeaseTests.asyncSetUp(self)
        self.plan = owner_plan()

    async def test_early_specialist_runs_before_first_owner_without_throwaway_owner_call(self):
        calls = []
        roles = []

        async def drive(backend, _ws, prompt, _model, approval, *, effort, **_kwargs):
            calls.append((backend.name, approval, effort))
            roles.append((await execution_state.record("owner-role-probe")).key)
            if backend.name == "meta":
                return AgentResult(text="Initial specialist finding: registry owns the cache key.",
                                   usage={"input_tokens": 3}), False
            self.assertIn("complete original cache task", prompt)
            self.assertIn("registry owns the cache key", prompt)
            return AgentResult(text="Original task implemented and verified.", usage={"input_tokens": 5}), False

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(calls, [("meta", "confirm", 100), ("llama", "full", 10)])
        self.assertNotEqual(roles[0], roles[1])
        self.assertEqual(result.usage, {"input_tokens": 8})
        self.assertEqual(self.record.get("usage_processed_audit"), {"input_tokens": 8})
        self.assertEqual(len(self.record.get("queue")), 1)
        self.assertEqual(self.record.get("cursor"), 1)
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_initial_specialist_cancellation_resumes_before_any_owner_starts(self):
        calls = Counter()
        order = []

        async def drive(backend, _ws, prompt, *_args, **_kwargs):
            calls[backend.name] += 1
            order.append(backend.name)
            if backend.name == "meta":
                if calls["meta"] == 1:
                    raise asyncio.CancelledError
                return AgentResult(text="Resumed initial specialist: cache lifetime is registry-bound."), False
            self.assertIn("cache lifetime is registry-bound", prompt)
            return AgentResult(text="Original cache task completed using the retained decision."), False

        with self.assertRaises(asyncio.CancelledError):
            await self.invoke(drive)
        self.assertEqual(order, ["meta"])
        self.assertEqual(self.record.get("cursor"), 0)
        self.assertEqual(len(self.record.get("tier_leases")["0"]), 1)

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(order, ["meta", "meta", "llama"])
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_completed_initial_specialist_is_not_repeated_after_owner_cancellation(self):
        calls = Counter()

        async def drive(backend, _ws, prompt, *_args, **_kwargs):
            calls[backend.name] += 1
            if backend.name == "meta":
                return AgentResult(text="Completed initial decision: preserve registry ownership.",
                                   usage={"input_tokens": 3}), False
            self.assertIn("preserve registry ownership", prompt)
            if calls["llama"] == 1:
                raise asyncio.CancelledError
            return AgentResult(text="Original owner implementation verified.", usage={"input_tokens": 5}), False

        with self.assertRaises(asyncio.CancelledError):
            await self.invoke(drive)
        self.assertIn("preserve registry ownership", str(self.record.get("tier_leases")))

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(calls, {"meta": 1, "llama": 2})
        self.assertEqual(self.record.get("usage_processed_audit"), {"input_tokens": 8})
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_owner_native_prerequisite_returns_all_new_usage_and_resume_does_not_rebill_helper(self):
        self.tiers["low"] = ("codex", "gpt-6.1-sol")
        calls = Counter()

        async def drive(backend, _ws, prompt, *_args, **_kwargs):
            calls[backend.name] += 1
            if backend.name == "meta":
                return AgentResult(text="Initial advisory evidence remains valid.",
                                   usage={"input_tokens": 3}), False
            self.assertIn("Initial advisory evidence remains valid", prompt)
            if calls["codex"] == 1:
                return AgentResult(text="PARTIAL: native prerequisite needs recovery.",
                                   usage={"input_tokens": 5}, continuation_reason="prerequisite"), False
            return AgentResult(text="Original native owner completed the retained task.",
                               usage={"input_tokens": 2}), False

        first, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertEqual(first.continuation_reason, "prerequisite")
        self.assertEqual(first.usage, {"input_tokens": 8})
        self.assertEqual(self.record.get("usage_processed_audit"), {"input_tokens": 8})
        self.assertEqual(self.record.get("cursor"), 0)
        self.assertEqual(self.record.get("tier_leases")["0"][0]["state"], "complete")

        resumed, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(resumed.error)
        self.assertEqual(calls, {"meta": 1, "codex": 2})
        self.assertEqual(resumed.usage, {"input_tokens": 2})
        self.assertEqual(self.record.get("usage_processed_audit"), {"input_tokens": 10})
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_distinct_supported_specialists_overlap_and_finish_before_owner(self):
        self.plan = owner_plan(specialists=[
            {"tier": "mid", "reason": "Inspect the understood migration boundary", "task": "Read the migration metadata"},
            {"tier": "high", "reason": "Resolve the coupled invariant", "task": "Analyze existing cache ownership"},
        ])
        started = set()
        finished = set()
        both = asyncio.Event()

        async def drive(backend, *_args, **_kwargs):
            if backend.name != "llama":
                started.add(backend.name)
                if len(started) == 2:
                    both.set()
                await asyncio.wait_for(both.wait(), 2)
                finished.add(backend.name)
                return AgentResult(text=f"Specialist {backend.name} findings retained."), False
            self.assertEqual(finished, {"zai", "meta"})
            return AgentResult(text="Original owner completed implementation and verification."), False

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_parallel_initial_consultation_reuses_saved_sibling_after_other_is_cancelled(self):
        self.plan = owner_plan(specialists=[
            {"tier": "mid", "reason": "Inspect migration metadata", "task": "Read the migration metadata"},
            {"tier": "high", "reason": "Resolve the coupled invariant", "task": "Analyze cache ownership"},
        ])
        calls = Counter()
        saved_fast_helper = asyncio.Event()
        original_save = execution_state.ExecutionRecord.save

        async def save(record, **values):
            await original_save(record, **values)
            if any(item["tier"] == "mid" and item["state"] == "complete"
                   for item in values.get("tier_leases", {}).get("0", [])):
                saved_fast_helper.set()

        async def drive(backend, _ws, prompt, *_args, **_kwargs):
            calls[backend.name] += 1
            if backend.name == "zai":
                return AgentResult(text="Saved migration finding: retain version three.",
                                   usage={"input_tokens": 3}), False
            if backend.name == "meta":
                await asyncio.wait_for(saved_fast_helper.wait(), 2)
                if calls["meta"] == 1:
                    raise asyncio.CancelledError
                return AgentResult(text="Resumed invariant finding: registry owns cache lifetime.",
                                   usage={"input_tokens": 5}), False
            self.assertIn("retain version three", prompt)
            self.assertIn("registry owns cache lifetime", prompt)
            return AgentResult(text="Original owner implemented and verified both findings.",
                               usage={"input_tokens": 7}), False

        with mock.patch.object(execution_state.ExecutionRecord, "save", save):
            with self.assertRaises(asyncio.CancelledError):
                await self.invoke(drive)
            self.assertEqual(calls, {"zai": 1, "meta": 1})
            self.assertEqual(self.record.get("cursor"), 0)
            self.assertEqual([item["state"] for item in self.record.get("tier_leases")["0"]],
                             ["complete", "pending"])

            result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(calls, {"zai": 1, "meta": 2, "llama": 1})
        self.assertEqual(self.record.get("usage_processed_audit"), {"input_tokens": 15})
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_owner_followup_stays_with_same_cursor_model_and_scope(self):
        calls = []

        async def drive(backend, _ws, prompt, *_args, **_kwargs):
            calls.append(backend.name)
            if backend.name == "meta":
                return AgentResult(text="Initial specialist decision retained."), False
            if calls.count("llama") == 1:
                return AgentResult(text=(
                    "Initial cache repair verified.\n"
                    "[followup:high] Verify the newly discovered dependent cache export\n"
                )), False
            self.assertIn("newly discovered dependent cache export", prompt)
            return AgentResult(text="Original cache task and dependent export verified."), False

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(calls, ["meta", "llama", "llama"])
        self.assertEqual([row["tier"] for row in self.record.get("queue")], ["low"])
        self.assertEqual(self.record.get("cursor"), 1)
        self.assertIn("newly discovered dependent cache export", str(self.record.get("owner_obligations")))
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_initial_consultations_share_the_three_request_helper_bound(self):
        self.plan = owner_plan(specialists=[
            {"tier": "mid", "reason": "Inspect migration metadata", "task": "Read the current migration metadata"},
            {"tier": "high", "reason": "Resolve the design decision", "task": "Analyze the existing invariant"},
        ])
        calls = Counter()

        async def drive(backend, *_args, **_kwargs):
            calls[backend.name] += 1
            if backend.name == "llama":
                return AgentResult(text=lease_fixture.lease("high", f"later-question-{calls['llama']}")), False
            return AgentResult(text="Read-only specialist findings retained."), False

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertTrue(result.error)
        self.assertFalse(result.completion_verified)
        self.assertEqual(calls, {"zai": 1, "meta": 2, "llama": 2})
        self.assertEqual(self.record.get("cursor"), 0)
        self.assertEqual(self.record.get("lease_rounds")["0"], 3)
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_tier_changed_identical_followup_is_one_bounded_obligation_and_terminal_resume_is_cached(self):
        self.plan = owner_plan(specialists=[])
        calls = []
        roles = []
        hints = ("high", "mid", "low", "high")

        async def drive(backend, _ws, prompt, *_args, **_kwargs):
            self.assertEqual(backend.name, "llama")
            roles.append((await execution_state.record("repeated-owner-role-probe")).key)
            calls.append(prompt)
            if len(calls) == 2:
                self.assertIn("Verify dependent cache export", prompt)
            return AgentResult(text=f"PARTIAL: dependency unresolved.\n[followup:{hints[len(calls) - 1]}] "
                                    "Verify dependent cache export", usage={"input_tokens": 2}), False

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertIn("repeatedly returned unresolved obligations", result.error)
        self.assertFalse(result.completion_verified)
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(set(roles)), 1)
        self.assertEqual(self.record.get("cursor"), 0)
        obligations = self.record.get("owner_obligations")["0"]
        self.assertEqual(len(obligations), 1)
        self.assertEqual(obligations[0]["repeats"], 3)
        self.assertEqual(self.record.get("usage_processed_audit"), {"input_tokens": 8})

        resumed, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertEqual(resumed.error, result.error)
        self.assertFalse(resumed.completion_verified)
        self.assertEqual(len(calls), 4)
        self.assertEqual(self.record.get("usage_processed_audit"), {"input_tokens": 8})
        self.assertEqual(self.internal_labels, ["Flexible planner"])
