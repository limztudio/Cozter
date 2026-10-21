"""Initial specialists and persistent owners retain separate native CLI lanes."""

import asyncio
import unittest
from unittest import mock

from Cozter import agent, execution_state, flexible, session, workspace
from Cozter.tests import test_flexible_native_sessions as native_fixture
from Cozter.tests.test_flexible_owner_runtime import owner_plan


class FlexibleOwnerNativeTests(unittest.IsolatedAsyncioTestCase):
    _started = staticmethod(native_fixture.FlexibleNativeSessionsTests._started)
    _answer = staticmethod(native_fixture.FlexibleNativeSessionsTests._answer)
    _missing = staticmethod(native_fixture.FlexibleNativeSessionsTests._missing)
    _conversation = native_fixture.FlexibleNativeSessionsTests._conversation
    _select_workers = native_fixture.FlexibleNativeSessionsTests._select_workers
    _internal = native_fixture.FlexibleNativeSessionsTests._internal
    _create_process = native_fixture.FlexibleNativeSessionsTests._create_process

    async def asyncSetUp(self):
        await native_fixture.FlexibleNativeSessionsTests.asyncSetUp(self)
        self.plan = owner_plan()
        self.cancel_initial_once = False
        self.record = None
        workspace.set_flexible_reasoning_effort(self.ws, "low", 20)
        workspace.set_flexible_reasoning_effort(self.ws, "high", 100)

    async def _drive(self, backend, workspace_path, prompt, model, approval, *, effort, **kwargs):
        value = await native_fixture.FlexibleNativeSessionsTests._drive(
            self, backend, workspace_path, prompt, model, approval, effort=effort, **kwargs,
        )
        self.attempts[-1]["effort"] = effort
        if self.cancel_initial_once and approval == "confirm":
            self.cancel_initial_once = False
            raise asyncio.CancelledError
        return value

    async def invoke(self, sid, *, job, request):
        async with execution_state.execution_scope(self.ws, job, request):
            execution_state.bind_conversation(sid)
            try:
                return await agent._run_flexible(
                    request, request, self.ws, approval="full", effort=0,
                    collaborative=False, summary_backend_name="codex", summary_model="offline-summary",
                    on_event=None, inject_queue=None, injected=[], session_data=session.load_session(self.ws, sid),
                )
            finally:
                self.record = await execution_state.record("flexible_plan")

    async def run_job(self, sid, *, job, request, original=None):
        canonical = original or request
        result = await agent.run(
            request, self.ws, 7, model="unused-flexible-model", approval="full",
            summary_model="offline-summary", backend_name="flexible", summary_backend_name="codex",
            session_id=sid, execution_id=job, original_request=canonical,
        )
        async with execution_state.execution_scope(self.ws, job, canonical):
            self.record = await execution_state.record("flexible_plan")
        return result

    async def test_all_four_start_specialist_first_and_reuse_separate_owner_and_helper_uuids(self):
        for _module, cls, _model in native_fixture._CASES:
            with self.subTest(backend=cls.name):
                self._select_workers(cls.name)
                sid = self._conversation()
                self.plan = owner_plan()
                start = len(self.attempts)

                result, restarting = await self.invoke(sid, job=f"first-owner-{cls.name}", request="FIRST-OWNER-REQUEST")

                self.assertFalse(restarting)
                self.assertFalse(result.error)
                first_helper, first_owner = self.attempts[start:]
                self.assertEqual([first_helper["approval"], first_owner["approval"]], ["confirm", "full"])
                self.assertEqual([first_helper["effort"], first_owner["effort"]], [100, 20])
                self.assertNotEqual(first_helper["native_id"], first_owner["native_id"])
                helper_answer = f"WORKER-ANSWER-MARKER-{start + 1}"
                self.assertEqual(first_owner["prompt"].count(helper_answer), 1)
                self.assertIn("FIRST-OWNER-REQUEST", first_owner["prompt"])
                self.plan = owner_plan(specialists=[{
                    "tier": "high", "reason": "Resolve a new coupled invariant",
                    "task": "Inspect the changed cache lifetime without editing or execution",
                }])
                second_start = len(self.attempts)

                result, restarting = await self.invoke(sid, job=f"second-owner-{cls.name}", request="SECOND-OWNER-REQUEST")

                self.assertFalse(restarting)
                self.assertFalse(result.error)
                later_helper, later_owner = self.attempts[second_start:]
                self.assertEqual(later_helper["previous_id"], first_helper["native_id"])
                self.assertEqual(later_owner["previous_id"], first_owner["native_id"])
                self.assertEqual(later_owner["prompt"].count(f"WORKER-ANSWER-MARKER-{second_start + 1}"), 1)
                self.assertNotIn(helper_answer, later_owner["prompt"])
                self.assertEqual(self.record.get("cursor"), 1)
        self.assertEqual(self.merge_prompts, [])

    async def test_all_four_resume_interrupted_initial_specialist_before_starting_owner(self):
        for _module, cls, _model in native_fixture._CASES:
            with self.subTest(backend=cls.name):
                self._select_workers(cls.name)
                sid = self._conversation()
                self.plan = owner_plan()
                self.cancel_initial_once = True
                start = len(self.attempts)
                job = f"interrupted-owner-{cls.name}"

                with self.assertRaises(asyncio.CancelledError):
                    await self.invoke(sid, job=job, request="INTERRUPTED-OWNER-REQUEST")
                self.assertEqual(len(self.attempts) - start, 1)
                initial_helper = self.attempts[start]
                self.assertEqual(initial_helper["approval"], "confirm")
                self.assertEqual(self.record.get("cursor"), 0)
                planner_count = len(self.planner_prompts)

                result, restarting = await self.invoke(sid, job=job, request="INTERRUPTED-OWNER-REQUEST")

                self.assertFalse(restarting)
                self.assertFalse(result.error)
                resumed_helper, first_owner = self.attempts[start + 1:]
                self.assertEqual(resumed_helper["previous_id"], initial_helper["native_id"])
                self.assertNotEqual(first_owner["native_id"], initial_helper["native_id"])
                self.assertEqual([resumed_helper["approval"], first_owner["approval"]], ["confirm", "full"])
                self.assertIn(f"WORKER-ANSWER-MARKER-{start + 2}", first_owner["prompt"])
                self.assertEqual(len(self.planner_prompts), planner_count)
        self.assertEqual(self.merge_prompts, [])

    async def test_judge_reopens_same_high_owner_without_replanning_or_replaying_initial_advice(self):
        for _module, cls, _model in native_fixture._CASES:
            for missing in (False, True):
                with self.subTest(backend=cls.name, missing=missing):
                    self._select_workers(cls.name)
                    sid = self._conversation()
                    self.plan = owner_plan(tier="high", specialists=[{
                        "tier": "low", "reason": "Inventory the existing cache contract",
                        "task": "Read the existing cache contract and exact paths without edits or execution",
                    }])
                    start = len(self.attempts)
                    planners_before = len(self.planner_prompts)
                    judge_calls = 0

                    async def judge(*_args, **_kwargs):
                        nonlocal judge_calls
                        judge_calls += 1
                        if judge_calls == 1:
                            self.plan = owner_plan(tier="low", specialists=[])
                            self.missing_provider = missing
                            return flexible.JudgeVerdict(True, "Documentation remains unverified",
                                                         "VERIFY-REMAINING-DOCS-MARKER"), False
                        return flexible.JudgeVerdict(False, verified=True), False

                    with mock.patch.object(agent, "_judge_draft", side_effect=judge):
                        result = await self.run_job(sid, job=f"judge-owner-{cls.name}-{missing}",
                                                    request="FULL-ORIGINAL-CACHE-SCOPE-MARKER")

                    self.assertFalse(result.error)
                    self.assertTrue(result.completion_verified)
                    attempts = self.attempts[start:]
                    self.assertEqual(len(attempts), 4 if missing else 3)
                    advisor, owner, continued = attempts[:3]
                    self.assertEqual([advisor["approval"], owner["approval"], continued["approval"]],
                                     ["confirm", "full", "full"])
                    self.assertEqual([advisor["effort"], owner["effort"], continued["effort"]], [20, 100, 100])
                    self.assertEqual(continued["previous_id"], owner["native_id"])
                    self.assertIn("VERIFY-REMAINING-DOCS-MARKER", continued["prompt"])
                    self.assertNotIn("FULL-ORIGINAL-CACHE-SCOPE-MARKER", continued["prompt"])
                    self.assertNotIn(f"WORKER-ANSWER-MARKER-{start + 1}", continued["prompt"])
                    self.assertNotIn(f"WORKER-ANSWER-MARKER-{start + 2}", continued["prompt"])
                    if missing:
                        bootstrap = attempts[3]["prompt"]
                        self.assertIn("FULL-ORIGINAL-CACHE-SCOPE-MARKER", bootstrap)
                        self.assertIn("VERIFY-REMAINING-DOCS-MARKER", bootstrap)
                        self.assertIn(f"WORKER-ANSWER-MARKER-{start + 1}", bootstrap)
                        self.assertIn(f"WORKER-ANSWER-MARKER-{start + 2}", bootstrap)
                    self.assertEqual(len(self.planner_prompts), planners_before + 1)
                    self.assertEqual([row["tier"] for row in self.record.get("queue")], ["high"])
                    self.assertFalse(self.record.get("active"))
        self.assertEqual(self.merge_prompts, [])

    async def test_judge_limit_keeps_high_owner_and_initial_advice_for_next_allowance(self):
        self._select_workers("codex")
        sid = self._conversation()
        self.plan = owner_plan(tier="high", specialists=[{
            "tier": "low", "reason": "Read the cache inventory", "task": "Read existing cache paths without edits",
        }])
        job = "high-owner-judge-limit"
        original = "FULL-ORIGINAL-JUDGE-LIMIT-SCOPE-MARKER"
        judge_calls = 0

        async def judge(*_args, **_kwargs):
            nonlocal judge_calls
            judge_calls += 1
            return flexible.JudgeVerdict(True, "Documentation still needs verification",
                                         f"VERIFY-REMAINING-DOCS-{judge_calls}"), False

        with (mock.patch.object(flexible, "JUDGE_MAX_CONTINUES", 1),
              mock.patch.object(agent, "_judge_draft", side_effect=judge)):
            partial = await self.run_job(sid, job=job, request=original)

        self.assertFalse(partial.error)
        self.assertEqual(partial.continuation_reason, "judge_limit")
        self.assertFalse(partial.completion_verified)
        self.assertEqual(len(self.attempts), 3)
        owner = self.attempts[1]
        self.assertTrue(self.record.get("active"))
        self.assertEqual(self.record.get("cursor"), 0)
        self.assertEqual(self.record.get("tier_leases")["0"][0]["state"], "complete")
        self.assertIn("VERIFY-REMAINING-DOCS-2", str(self.record.get("owner_obligations")))
        self.plan = owner_plan(tier="low", specialists=[])

        with mock.patch.object(agent, "_judge_draft", return_value=(flexible.JudgeVerdict(False, verified=True), False)):
            result = await self.run_job(sid, job=job, request=partial.continue_instruction, original=original)

        self.assertFalse(result.error)
        self.assertTrue(result.completion_verified)
        self.assertEqual(len(self.attempts), 4)
        resumed = self.attempts[-1]
        self.assertEqual(resumed["previous_id"], owner["native_id"])
        self.assertEqual(resumed["effort"], 100)
        self.assertIn("VERIFY-REMAINING-DOCS-2", resumed["prompt"])
        self.assertEqual(len(self.planner_prompts), 1)
        self.assertEqual(self.merge_prompts, [])
