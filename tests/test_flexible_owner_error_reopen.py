"""Reopened owners retain evidence without replaying an earlier phase failure."""

import unittest

from Cozter import agent, execution_state
from Cozter.backends_agent.base import AgentResult
from Cozter.tests import test_flexible_low_default as native_fixture
from Cozter.tests.test_flexible_owner_runtime import owner_plan


class FlexibleOwnerErrorReopenTests(unittest.IsolatedAsyncioTestCase):
    internal = native_fixture.FlexibleLowDefaultTests.internal
    invoke = native_fixture.FlexibleLowDefaultTests.invoke
    accept_native = native_fixture.FlexibleLowDefaultTests.accept_native
    old_error = "meta returned HTTP 400: arguments must be valid JSON"

    async def asyncSetUp(self):
        await native_fixture.FlexibleLowDefaultTests.asyncSetUp(self)
        self.plan_output = owner_plan(specialists=[])

    async def test_failed_phase_reopens_exact_native_owner_and_current_result_decides_success(self):
        for new_error in (None, "meta returned HTTP 400: current request remains invalid"):
            with self.subTest(new_error=new_error):
                self.calls = []
                self.internal_labels = []
                attempts = 0

                async def drive(_backend, _ws, prompt, model, approval, *, effort,
                                on_native_activity=None, **_kwargs):
                    nonlocal attempts
                    attempts += 1
                    await self.accept_native(prompt, model, approval, effort, on_native_activity)
                    if attempts == 1:
                        return AgentResult(
                            text="PARTIAL: completed mutation and verification proof retained before provider failure.",
                            error=self.old_error, usage={"input_tokens": 3},
                        ), False
                    self.assertIn("Continue the current job from this native conversation", prompt)
                    self.assertIn("Recover the provider transcript and finish the original task", prompt)
                    return AgentResult(text="Original task verified after recovery.", error=new_error,
                                       usage={"input_tokens": 2}), False

                first, _ = await self.invoke(drive)
                self.assertEqual(first.error, self.old_error)
                async with execution_state.execution_scope(self.ws, "low-default", self.request) as execution:
                    execution_state.bind_conversation(self.sid)
                    saved = await execution_state.record("flexible_plan")
                    transcript = await execution.record({"untouched_http_evidence": new_error})
                    await transcript.save(http_messages=[{"role": "tool", "content": "already completed write"}])
                    original_http = transcript.path.read_bytes()
                    owner_request = saved.get("owner_request")
                    queue = saved.get("queue")
                    await saved.save(blocked=[0], output={"error": self.old_error,
                                                         "events": [], "usage": None})
                    self.assertTrue(await agent._continue_flexible_owner(
                        "Recover the provider transcript and finish the original task", first.text,
                    ))
                    self.assertEqual(saved.get("worker_errors"), [])
                    self.assertEqual(saved.get("blocked"), [])
                    self.assertIsNone(saved.get("output")["error"])
                    self.assertEqual(saved.get("owner_error_audit")[-1], self.old_error)
                    self.assertEqual(saved.get("owner_request"), owner_request)
                    self.assertEqual(saved.get("queue"), queue)
                    self.assertIn("completed mutation", saved.get("phase_previous_draft"))
                    self.assertEqual(transcript.path.read_bytes(), original_http)

                resumed, _ = await self.invoke(drive)

                self.assertEqual(resumed.error, new_error)
                self.assertEqual(resumed.usage, {"input_tokens": 2})
                self.assertEqual(self.calls[0]["native_id"], self.calls[1]["native_id"])
                self.assertEqual(self.calls[0]["key"], self.calls[1]["key"])
                self.assertEqual(self.internal_labels, ["Flexible planner"])
                self.assertIn(self.old_error, self.record.get("owner_error_audit"))
                async with execution_state.execution_scope(self.ws, "low-default", self.request):
                    await (await execution_state.record("flexible_plan")).save(active=False)

    async def test_error_audit_stays_bounded_while_last_failure_and_evidence_survive(self):
        async with execution_state.execution_scope(self.ws, "bounded-reopen", self.request):
            saved = await execution_state.record("flexible_plan")
            await saved.save(owner_mode=True, queue=[{"tier": "low", "instruction": self.request}],
                             owner_obligations={}, owner_error_audit=["old " + str(index) for index in range(20)],
                             worker_errors=[self.old_error + "x" * 7000], output={})
            await agent._continue_flexible_owner("Finish retained scope", "Prior verified artifact proof")
            self.assertEqual(len(saved.get("owner_error_audit")), 8)
            self.assertLessEqual(len(saved.get("owner_error_audit")[-1]), 6000)
            self.assertIn(self.old_error, saved.get("owner_error_audit")[-1])
            self.assertEqual(saved.get("phase_previous_draft"), "Prior verified artifact proof")

