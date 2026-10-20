"""Definite helper failures suppress new calls without discarding completed work."""

import asyncio
from collections import Counter
import tempfile
import unittest
from unittest import mock

from Cozter import agent, execution_state, workspace
from Cozter.backends_agent.base import AgentResult
from Cozter.tests.test_flexible_tier_leases import lease


class FlexibleHelperUnavailabilityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.tiers = {
            "low": ("llama", "local-low"),
            "mid": ("llama", "local-mid"),
            "high": ("meta", "muse-spark-1.3"),
        }
        self.enterContext(mock.patch.object(workspace, "get_flexible_run_config", return_value=self.tiers))
        self.plan = "[PLAN]\n1. [high] Repair and verify the parent obligation\n[/PLAN]"
        self.labels = []
        self.record = None

    async def internal(self, _backend, _ws, _prompt, _model, *, label, **_kwargs):
        self.labels.append(label)
        return self.plan if label == "Flexible planner" else "Verified requested work."

    async def invoke(self, drive, *, job="helper-cache"):
        async with execution_state.execution_scope(self.ws, job, "Repair the parent obligation"):
            with (
                mock.patch.object(agent, "run_internal_backend", self.internal),
                mock.patch.object(agent, "_drive_backend", drive),
            ):
                try:
                    return await agent._run_flexible(
                        "Original request", "Original request", self.ws,
                        approval="full", effort=0, collaborative=False,
                        summary_backend_name="meta", summary_model="summary",
                        on_event=None, inject_queue=None, injected=[],
                    )
                finally:
                    self.record = await execution_state.record("flexible_plan")

    async def test_failed_helper_target_is_cached_but_other_model_and_parent_can_finish(self):
        for index, error in enumerate((
            "quota exhausted",
            "llama returned HTTP 401: unauthorized",
            "You've hit your monthly spend limit. Try again next month.",
        )):
            with self.subTest(error=error):
                calls = Counter()

                async def drive(backend, _ws, prompt, model, _approval, **_kwargs):
                    calls[backend.name, model] += 1
                    if backend.name == "meta":
                        turn = calls[backend.name, model]
                        if turn <= 3:
                            target = "mid" if turn == 3 else "low"
                            return AgentResult(text=lease(target, f"question-{turn}"),
                                               usage={"input_tokens": 2}), False
                        self.assertIn("other model inspected the current inputs", prompt)
                        return AgentResult(text="Parent obligation repaired and verified.",
                                           usage={"input_tokens": 2}), False
                    if model == "local-low":
                        return AgentResult(error=error, usage={"input_tokens": 3}), False
                    return AgentResult(text="The other model inspected the current inputs.",
                                       usage={"input_tokens": 3}), False

                result, restarting = await self.invoke(drive, job=f"definite-{index}")

                self.assertFalse(restarting)
                self.assertIsNone(result.error, "an optional helper failure must remain advisory")
                self.assertEqual(calls, {
                    ("meta", "muse-spark-1.3"): 4, ("llama", "local-low"): 1, ("llama", "local-mid"): 1,
                })
                self.assertEqual(result.usage, {"input_tokens": 14})
                self.assertEqual(self.record.get("unavailable_models"), [
                    {"backend": "llama", "model": "local-low", "cause": error},
                ])
                self.assertEqual(self.record.get("cursor"), 1)

    async def test_transient_throttling_and_task_refusal_do_not_poison_helper_cache(self):
        for index, error in enumerate((
            "llama returned HTTP 429: too many requests; retry later",
            "llama returned HTTP 403: forbidden for this request",
        )):
            with self.subTest(error=error):
                calls = Counter()

                async def drive(backend, *_args, **_kwargs):
                    calls[backend.name] += 1
                    if backend.name == "llama":
                        return AgentResult(error=error), False
                    if calls["meta"] <= 3:
                        return AgentResult(text=lease("low", f"question-{calls['meta']}")), False
                    return AgentResult(text="Parent verified its obligation without helper findings."), False

                result, restarting = await self.invoke(drive, job=f"transient-{index}")

                self.assertFalse(restarting)
                self.assertIsNone(result.error)
                self.assertEqual(calls, {"meta": 4, "llama": 3})
                self.assertEqual(self.record.get("unavailable_models"), [])

    async def test_helper_failure_makes_later_required_target_skip_honestly_terminal(self):
        self.plan = (
            "[PLAN]\n1. [high] Repair and verify the parent obligation\n"
            "2. [low] Verify the separate required component\n[/PLAN]"
        )
        calls = Counter()

        async def drive(backend, *_args, **_kwargs):
            calls[backend.name] += 1
            if backend.name == "llama":
                return AgentResult(error="quota exhausted"), False
            if calls["meta"] == 1:
                return AgentResult(text=lease("low", "optional-parent-inspection")), False
            return AgentResult(text="Parent obligation repaired and verified."), False

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertEqual(calls, {"meta": 2, "llama": 1})
        self.assertEqual(result.error, "quota exhausted")
        self.assertIn("Parent obligation repaired and verified", result.text)
        self.assertIn("sub-task 2 was not run: Verify the separate required component", result.text)
        self.assertFalse(result.completion_verified)
        self.assertEqual(self.labels, ["Flexible planner"])

    async def test_failed_helper_cache_survives_interruption_before_parent_continuation(self):
        calls = Counter()

        async def drive(backend, *_args, **_kwargs):
            calls[backend.name] += 1
            if backend.name == "llama":
                return AgentResult(error="invalid API key"), False
            if calls["meta"] == 1:
                return AgentResult(text=lease("low", "first-question")), False
            if calls["meta"] == 2:
                raise asyncio.CancelledError
            if calls["meta"] == 3:
                return AgentResult(text=lease("low", "different-question")), False
            return AgentResult(text="Parent verified its obligation without unavailable helper access."), False

        with self.assertRaises(asyncio.CancelledError):
            await self.invoke(drive)
        self.assertEqual(self.record.get("cursor"), 0)
        self.assertEqual(self.record.get("unavailable_models")[0]["cause"], "invalid API key")

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertIsNone(result.error)
        self.assertEqual(calls, {"meta": 4, "llama": 1})
        self.assertEqual(self.labels, ["Flexible planner"])

    async def test_completed_parallel_sibling_wins_over_retained_unavailability_on_resume(self):
        self.tiers["mid"] = self.tiers["low"]
        self.plan = (
            "[PLAN]\n1. [low] [parallel-read] Inspect component A\n"
            "2. [mid] [parallel-read] Inspect component B\n[/PLAN]"
        )
        calls = Counter()
        sibling_done = asyncio.Event()

        async def drive(_backend, _ws, prompt, *_args, **_kwargs):
            index = int(prompt.split("[Sub-task ", 1)[1].split("/", 1)[0])
            calls[index] += 1
            if index == 1:
                await asyncio.wait_for(sibling_done.wait(), 2)
                return AgentResult(error="quota exhausted", usage={"input_tokens": 1}), False
            sibling_done.set()
            return AgentResult(text="Retained B evidence: source matches the existing fixture.",
                               usage={"input_tokens": 2}), False

        save = execution_state.ExecutionRecord.save
        interrupted = False

        async def save_then_interrupt(record, **values):
            nonlocal interrupted
            await save(record, **values)
            if values.get("cursor") == 1 and not interrupted:
                interrupted = True
                raise asyncio.CancelledError

        with mock.patch.object(execution_state.ExecutionRecord, "save", save_then_interrupt):
            with self.assertRaises(asyncio.CancelledError):
                await self.invoke(drive)
        self.assertEqual(self.record.get("cursor"), 1)
        self.assertTrue(self.record.get("worker_outputs")["1"]["complete"])

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertEqual(calls, {1: 1, 2: 1})
        self.assertEqual(result.error, "quota exhausted")
        self.assertIn("Retained B evidence", result.text)
        self.assertNotIn("sub-task 2 was not run", result.text)
        self.assertEqual(self.record.get("usage_processed_audit"), {"input_tokens": 3})
