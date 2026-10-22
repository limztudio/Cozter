"""The owner receives initial findings once, then only new answers unless history is lost."""

import asyncio
import copy
import unittest
from unittest import mock

from Cozter import agent, execution_state, flexible, session
from Cozter.backends_agent import _openai_agent as oa
from Cozter.tests import test_flexible_owner_runtime as owner_fixture
from Cozter.tests import test_flexible_tier_leases as lease_fixture


class FlexibleOwnerHttpTests(unittest.IsolatedAsyncioTestCase):
    internal = owner_fixture.FlexibleOwnerRuntimeTests.internal

    async def asyncSetUp(self):
        await owner_fixture.FlexibleOwnerRuntimeTests.asyncSetUp(self)
        self.requests = []
        self.owner_calls = 0
        self.specialist_calls = 0
        self.parent_record = None
        self.clear_owner_history = False
        self.owner_followup = False
        self.owner_lease = True
        self.enterContext(mock.patch.object(oa, "_system_prompt", return_value="Current owner policy"))
        self.tools = self.enterContext(mock.patch.object(oa, "_tools_for_approval", return_value=None))
        self.enterContext(mock.patch.object(oa, "_stream_completion", side_effect=self.stream))

    async def stream(self, _endpoint, payload, *_args, **_kwargs):
        self.assertIsNone(oa._HTTP_CONTINUATION_BOOTSTRAP.get())
        model = payload.get("model")
        self.requests.append((model, copy.deepcopy(payload)))
        if model == "muse-spark-1.3":
            self.specialist_calls += 1
            if self.specialist_calls == 2 and self.clear_owner_history:
                await self.parent_record.save(http_messages=[])
            return ("INITIAL-SPECIALIST-FINDING-A" if self.specialist_calls == 1
                    else "LATER-SPECIALIST-FINDING-B"), "", []
        self.assertEqual(model, "local-low")
        self.owner_calls += 1
        self.parent_record = oa.current_run()
        if self.owner_calls == 1:
            report = lease_fixture.lease("high", "second-focused-decision") if self.owner_lease else ""
            if self.owner_followup:
                report = "[followup:high] Verify newly discovered dependent cache export\n" + report
            return report, "", []
        return "Original owner completed implementation and verification.", "", []

    async def invoke(self):
        original_driver = agent._drive_backend
        return await lease_fixture.FlexibleTierLeaseTests.invoke(self, original_driver)

    def changing_policy(self):
        original = flexible.build_subtask_sections
        self.policy_revision = "ROUTING-POLICY-OLD"

        def build(*args, **kwargs):
            sections = original(*args, **kwargs)
            sections["worker_instructions"] += "\n" + self.policy_revision
            return sections

        self.enterContext(mock.patch.object(flexible, "build_subtask_sections", side_effect=build))

    async def test_saved_markerless_low_owner_receives_new_policy_once_then_consults_high(self):
        self.plan = owner_fixture.owner_plan(specialists=[])
        self.changing_policy()
        owner_payloads = []
        helper_calls = 0
        original_driver = agent._drive_backend

        async def drive(backend, *args, **kwargs):
            value = await original_driver(backend, *args, **kwargs)
            if backend.name == "llama" and len(owner_payloads) == 1:
                messages = self.parent_record.get("http_messages")
                messages[0]["content"] = agent._HTTP_WORKER_POLICY_MARKER.sub("", messages[0]["content"])
                await self.parent_record.save(http_messages=messages)
            return value

        async def stream(_endpoint, payload, *_args, **_kwargs):
            nonlocal helper_calls
            if payload["model"] == "muse-spark-1.3":
                helper_calls += 1
                return "HIGH-DECISION: retain registry ownership.", "", []
            self.parent_record = oa.current_run()
            owner_payloads.append(copy.deepcopy(payload))
            if len(owner_payloads) == 1:
                self.policy_revision = "ROUTING-POLICY-NEW"
                return "[followup:low] Resolve the remaining registry design question", "", []
            if len(owner_payloads) == 2:
                return lease_fixture.lease("high", "registry-design-after-policy-update"), "", []
            if len(owner_payloads) == 3:
                return "[followup:low] Verify the selected registry invariant", "", []
            return "The same low owner implemented and verified the original scope.", "", []

        with (mock.patch.object(oa, "_stream_completion", side_effect=stream),
              mock.patch.object(agent, "_drive_backend", side_effect=drive)):
            result, restarting = await self.invoke()

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(helper_calls, 1)
        self.assertEqual(len(owner_payloads), 4)
        users = [[row["content"] for row in payload["messages"] if row["role"] == "user"]
                 for payload in owner_payloads]
        self.assertIn("ROUTING-POLICY-NEW", users[1][-1])
        self.assertNotIn("Original request", users[1][-1])
        self.assertNotIn("ROUTING-POLICY-OLD", users[1][-1])
        self.assertIn("HIGH-DECISION", users[2][-1])
        self.assertNotIn("ROUTING-POLICY-NEW", users[2][-1])
        self.assertNotIn("ROUTING-POLICY-NEW", users[3][-1])
        self.assertEqual(sum(text.count("ROUTING-POLICY-NEW") for text in users[-1]), 1)
        self.assertEqual(self.record.get("queue")[0]["tier"], "low")
        self.assertEqual(self.internal_labels, ["Flexible planner"])
        self.assertEqual(self.parent_record.get("http_worker_policy")["http_identity"],
                         self.parent_record.get("http_identity"))

    async def test_unaccepted_policy_survives_cancel_without_false_receipt_or_duplicate_on_resume(self):
        self.plan = owner_fixture.owner_plan(specialists=[])
        self.changing_policy()
        owner_payloads = []
        waiting_for_provider = asyncio.Event()

        async def stream(_endpoint, payload, *_args, **_kwargs):
            self.parent_record = oa.current_run()
            owner_payloads.append(copy.deepcopy(payload))
            if len(owner_payloads) == 1:
                self.policy_revision = "ROUTING-POLICY-NEW"
                return "[followup:low] Verify the remaining registry invariant", "", []
            if len(owner_payloads) == 2:
                waiting_for_provider.set()
                await asyncio.Event().wait()
            if len(owner_payloads) == 3:
                return "[followup:low] Verify the final registry export", "", []
            return "All original obligations verified by the same owner.", "", []

        with mock.patch.object(oa, "_stream_completion", side_effect=stream):
            running = asyncio.create_task(self.invoke())
            try:
                await asyncio.wait_for(waiting_for_provider.wait(), 5)
            finally:
                running.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await running
            first_record = self.parent_record
            self.assertIsNone(first_record.get("http_worker_policy"))
            self.assertEqual(first_record.get("http_messages")[-1]["role"], "user")
            self.assertEqual(self.record.get("cursor"), 0)
            result, restarting = await self.invoke()

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(first_record.key, self.parent_record.key)
        self.assertEqual(len(owner_payloads), 4)
        self.assertEqual(owner_payloads[2]["messages"], owner_payloads[1]["messages"])
        resumed_users = [row["content"] for row in owner_payloads[2]["messages"] if row["role"] == "user"]
        self.assertEqual(sum(text.count("Verify the remaining registry invariant") for text in resumed_users), 1)
        self.assertEqual(sum(text.count("Continue parent task:") for text in resumed_users), 1)
        final_users = [row["content"] for row in owner_payloads[-1]["messages"] if row["role"] == "user"]
        self.assertEqual(sum(text.count("ROUTING-POLICY-NEW") for text in final_users), 1)
        self.assertEqual(sum(text.count("Original request") for text in final_users), 1)
        self.assertIsNotNone(self.parent_record.get("http_worker_policy"))
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_returning_to_an_older_policy_replaces_the_latest_policy_once(self):
        self.plan = owner_fixture.owner_plan(specialists=[])
        self.changing_policy()
        owner_payloads = []

        async def stream(_endpoint, payload, *_args, **_kwargs):
            self.parent_record = oa.current_run()
            owner_payloads.append(copy.deepcopy(payload))
            count = len(owner_payloads)
            if count == 1:
                self.policy_revision = "ROUTING-POLICY-NEW"
            elif count == 2:
                self.policy_revision = "ROUTING-POLICY-OLD"
            if count < 4:
                return f"[followup:low] Verify remaining invariant {count}", "", []
            return "All registry invariants verified.", "", []

        with mock.patch.object(oa, "_stream_completion", side_effect=stream):
            result, restarting = await self.invoke()

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        deltas = [[row["content"] for row in payload["messages"] if row["role"] == "user"][-1]
                  for payload in owner_payloads]
        self.assertIn("ROUTING-POLICY-NEW", deltas[1])
        self.assertIn("ROUTING-POLICY-OLD", deltas[2])
        self.assertNotIn("Original request", deltas[2])
        self.assertNotIn("ROUTING-POLICY-OLD", deltas[3])
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_initial_findings_bootstrap_owner_then_later_answer_is_a_delta(self):
        result, restarting = await self.invoke()

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual([model for model, _payload in self.requests],
                         ["muse-spark-1.3", "local-low", "muse-spark-1.3", "local-low"])
        owner_payloads = [payload for model, payload in self.requests if model == "local-low"]
        first_users = [row["content"] for row in owner_payloads[0]["messages"] if row["role"] == "user"]
        later_users = [row["content"] for row in owner_payloads[1]["messages"] if row["role"] == "user"]
        self.assertEqual(len(first_users), 1)
        self.assertIn("Original request", first_users[0])
        self.assertEqual(first_users[0].count("INITIAL-SPECIALIST-FINDING-A"), 1)
        self.assertEqual(later_users[0], first_users[0])
        self.assertEqual(len(later_users), 2)
        self.assertIn("LATER-SPECIALIST-FINDING-B", later_users[1])
        self.assertNotIn("INITIAL-SPECIALIST-FINDING-A", later_users[1])
        self.assertNotIn("Original request", later_users[1])
        self.assertEqual([call.args[0] for call in self.tools.call_args_list], ["confirm", "full", "confirm", "full"])
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_missing_owner_history_bootstraps_original_scope_and_all_retained_answers(self):
        self.clear_owner_history = True

        result, restarting = await self.invoke()

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        owner_payloads = [payload for model, payload in self.requests if model == "local-low"]
        later_users = [row["content"] for row in owner_payloads[1]["messages"] if row["role"] == "user"]
        self.assertEqual(len(later_users), 1)
        self.assertIn("Original request", later_users[0])
        self.assertIn("complete original cache task", later_users[0])
        self.assertEqual(later_users[0].count("INITIAL-SPECIALIST-FINDING-A"), 1)
        self.assertEqual(later_users[0].count("LATER-SPECIALIST-FINDING-B"), 1)
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_obligation_without_any_specialist_continues_as_http_delta(self):
        self.plan = owner_fixture.owner_plan(specialists=[])
        self.owner_lease = False
        self.owner_followup = True

        result, restarting = await self.invoke()

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual([model for model, _payload in self.requests], ["local-low", "local-low"])
        first, resumed = [payload for _model, payload in self.requests]
        first_users = [row["content"] for row in first["messages"] if row["role"] == "user"]
        resumed_users = [row["content"] for row in resumed["messages"] if row["role"] == "user"]
        self.assertEqual(len(first_users), 1)
        self.assertEqual(len(resumed_users), 2)
        self.assertEqual(resumed_users[0], first_users[0])
        self.assertIn("Verify newly discovered dependent cache export", resumed_users[1])
        self.assertNotIn("Original request", resumed_users[1])
        self.assertNotIn("Choose the cheapest capable owner", resumed_users[1])
        self.assertEqual(len(self.record.get("owner_obligations")["0"]), 1)
        self.assertEqual(self.record.get("queue")[0]["tier"], "low")
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_followup_and_lease_report_retains_both_obligation_and_new_answer_in_owner_delta(self):
        self.owner_followup = True

        result, restarting = await self.invoke()

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        owner_payloads = [payload for model, payload in self.requests if model == "local-low"]
        resumed_users = [row["content"] for row in owner_payloads[1]["messages"] if row["role"] == "user"]
        self.assertEqual(len(resumed_users), 2)
        delta = resumed_users[1]
        self.assertEqual(delta.count("LATER-SPECIALIST-FINDING-B"), 1)
        self.assertEqual(delta.count("Verify newly discovered dependent cache export"), 1)
        self.assertNotIn("INITIAL-SPECIALIST-FINDING-A", delta)
        self.assertNotIn("Original request", delta)
        self.assertEqual(self.record.get("cursor"), 1)
        self.assertEqual(len(self.record.get("owner_obligations")["0"]), 1)
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_resolved_repeat_limit_does_not_poison_judge_scope_and_regression_gets_fresh_delta(self):
        self.plan = owner_fixture.owner_plan(specialists=[])
        sid = session.create_session(self.ws, name="Owner regression generations")["id"]
        original = "Repair the cache export and documentation, retaining the whole original scope."
        job = "owner-regression-generations"
        old_scope = "Verify dependent cache export"
        old_followup = f"[followup:high] {old_scope}"
        owner_reports = [
            old_followup, old_followup, old_followup,
            old_followup + "\n" + lease_fixture.lease("high", "resolve-export-proof"),
            "Cache export now verified using the helper's evidence.",
            "[followup:mid] Verify newly discovered documentation links",
            "Documentation links verified; valid cache export evidence retained.",
            old_followup,
            "The new export regression and all original scope are verified.",
        ]
        owner_payloads = []
        helper_calls = 0
        judge_calls = 0

        async def stream(_endpoint, payload, *_args, **_kwargs):
            nonlocal helper_calls
            if payload["model"] == "muse-spark-1.3":
                helper_calls += 1
                return "Helper export proof: the registry owns the key.", "", []
            self.assertEqual(payload["model"], "local-low")
            owner_payloads.append(copy.deepcopy(payload))
            if len(owner_payloads) == 6:
                self.assertNotIn(old_scope, agent._TURN_BUDGET.get().pending)
            return owner_reports[len(owner_payloads) - 1], "", []

        async def judge(*_args, **_kwargs):
            nonlocal judge_calls
            judge_calls += 1
            if judge_calls == 1:
                return flexible.JudgeVerdict(True, "Documentation remains", "VERIFY-NEW-DOCS-MARKER"), False
            if judge_calls == 2:
                return flexible.JudgeVerdict(True, "A new export regression appeared", "RECHECK-EXPORT-MARKER"), False
            return flexible.JudgeVerdict(False, verified=True), False

        with (mock.patch.object(oa, "_stream_completion", side_effect=stream),
              mock.patch.object(agent, "run_internal_backend", self.internal),
              mock.patch.object(agent, "_judge_draft", side_effect=judge),
              mock.patch.object(agent, "_compaction_context_targets", return_value=()),
              mock.patch.object(agent, "_snapshot_attachment_images", return_value={}),
              mock.patch.object(agent, "_append_new_attachment_images", new=mock.AsyncMock()),
              mock.patch.object(agent, "_run_post_turn_maintenance", new=mock.AsyncMock())):
            result = await agent.run(
                original, self.ws, 7, approval="full", model="unused-flexible-model",
                backend_name="flexible", summary_backend_name="meta", summary_model="offline-summary",
                session_id=sid, execution_id=job,
            )

        self.assertFalse(result.error)
        self.assertTrue(result.completion_verified)
        self.assertEqual(len(owner_payloads), 9)
        self.assertEqual(helper_calls, 1)
        judge_delta = [row["content"] for row in owner_payloads[5]["messages"] if row["role"] == "user"][-1]
        self.assertIn("VERIFY-NEW-DOCS-MARKER", judge_delta)
        self.assertIn("Previously reported completed obligations: 1.", judge_delta)
        self.assertNotIn(old_scope, judge_delta)
        fresh_delta = [row["content"] for row in owner_payloads[8]["messages"] if row["role"] == "user"][-1]
        self.assertEqual(fresh_delta.count(old_scope), 1)
        self.assertIn("Previously reported completed obligations: 1, 2, 3.", fresh_delta)
        self.assertIn("Remaining obligations: 4, 5.", fresh_delta)
        async with execution_state.execution_scope(self.ws, job, original):
            self.record = await execution_state.record("flexible_plan")
        generations = [row for row in self.record.get("owner_obligations")["0"] if row["instruction"] == old_scope]
        self.assertEqual(len(generations), 2)
        self.assertEqual([row["generation"] for row in generations], [0, 1])
        self.assertEqual([row["repeats"] for row in generations], [3, 0])
        self.assertNotEqual(generations[0]["id"], generations[1]["id"])
        self.assertEqual(generations[0]["scope"], generations[1]["scope"])
        self.assertTrue(all(row["resolved"] for row in generations))
        self.assertEqual(self.internal_labels, ["Flexible planner"])
