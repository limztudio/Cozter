"""Reopened owners retain evidence without replaying an earlier phase failure."""

import unittest

from Cozter import agent, execution_state, flexible
from Cozter.backends_agent.base import AgentResult
from Cozter.tests import test_flexible_low_default as native_fixture
from Cozter.tests.test_flexible_owner_runtime import owner_plan
from Cozter.utils import json_digest


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

    async def test_judge_reopens_generation_48_in_same_native_owner_with_completed_history(self):
        instruction = "Verify the dependent export after the newly discovered generated binding change"
        scope = json_digest(flexible.instruction_identity(instruction))
        history = [{
            "id": json_digest([scope, generation]), "scope": scope, "generation": generation,
            "instruction": instruction, "repeats": 0, "resolved": True,
        } for generation in range(48)]

        async def drive(_backend, _ws, prompt, model, approval, *, effort,
                        on_native_activity=None, **_kwargs):
            await self.accept_native(prompt, model, approval, effort, on_native_activity)
            if len(self.calls) == 1:
                return AgentResult(text="Verified original cache changes; build receipt retained.",
                                   usage={"input_tokens": 3}), False
            self.assertIn(instruction, prompt)
            self.assertIn("Continue the current job from this native conversation", prompt)
            return AgentResult(text="Verified the new export generation using the retained build receipt.",
                               usage={"input_tokens": 2}), False

        initial, _ = await self.invoke(drive)
        async with execution_state.execution_scope(self.ws, "low-default", self.request) as execution:
            execution_state.bind_conversation(self.sid)
            saved = await execution_state.record("flexible_plan")
            original_queue = saved.get("queue")
            original_request = saved.get("owner_request")
            evidence = await execution.record({"completed_build_receipt": "preserve"})
            await evidence.save(exit_code=0, artifact="retained-build.log")
            original_evidence = evidence.path.read_bytes()
            await saved.save(owner_obligations={"0": history}, owner_delivery={"0:low": 48})
            self.assertTrue(await agent._continue_flexible_owner(instruction, initial.text))
            rows = saved.get("owner_obligations")["0"]
            self.assertEqual(rows[:48], history)
            self.assertEqual((rows[-1]["generation"], rows[-1]["resolved"], rows[-1]["repeats"]),
                             (48, False, 0))
            self.assertEqual(saved.get("owner_request"), original_request)
            self.assertEqual(saved.get("queue"), original_queue)
            self.assertEqual(evidence.path.read_bytes(), original_evidence)

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[0]["key"], self.calls[1]["key"])
        self.assertEqual(self.calls[0]["native_id"], self.calls[1]["native_id"])
        self.assertEqual(self.record.get("owner_obligations")["0"][:48], history)
        self.assertTrue(self.record.get("owner_obligations")["0"][48]["resolved"])
        self.assertEqual(self.record.get("cursor"), 1)
        self.assertEqual(self.record.get("usage_processed_audit"), {"input_tokens": 5})
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_cached_capacity_result_imports_full_followups_before_resuming_native_owner(self):
        instructions = [
            "LARGE-FIRST-SCOPE " + "Retain the exact generated binding rule and its source references. " * 210
            + "END-LARGE-FIRST-SCOPE",
            *(f"CAPACITY-SCOPE-{number:03d} Verify the retained source target and generated binding contract."
              for number in range(1, 49)),
        ]
        report = "Verified changes and build evidence remain valid.\n" + "\n".join(
            f"[followup:low] {instruction}" for instruction in instructions
        )
        capacity_error = "Flexible owner obligation capacity reached; unfinished scope is retained."
        page_count = 0

        async def drive(_backend, _ws, prompt, model, approval, *, effort,
                        on_native_activity=None, **_kwargs):
            nonlocal page_count
            await self.accept_native(prompt, model, approval, effort, on_native_activity)
            if len(self.calls) == 1:
                return AgentResult(text=report, usage={"input_tokens": 3},
                                   continuation_reason="prerequisite"), False
            saved = await execution_state.current_execution().record({"record": "flexible_plan", "worker": "main"})
            rows = saved.get("owner_obligations")["0"]
            self.assertEqual([row["instruction"] for row in rows], instructions)
            if page_count == 0:
                self.assertIn(instructions[0], prompt)
                self.assertNotIn(instructions[1], prompt)
            else:
                self.assertIn(instructions[-1], prompt)
            page_count += 1
            return AgentResult(text="Verified the supplied page using the saved build evidence.",
                               usage={"input_tokens": 2}), False

        initial, _ = await self.invoke(drive)
        self.assertEqual(initial.continuation_reason, "prerequisite")
        async with execution_state.execution_scope(self.ws, "low-default", self.request):
            saved = await execution_state.record("flexible_plan")
            rows = []
            for instruction in instructions[:48]:
                scope = json_digest(flexible.instruction_identity(instruction))
                rows.append({"id": json_digest([scope, 0]), "scope": scope, "generation": 0,
                             "instruction": instruction, "repeats": 0, "resolved": False})
            previous_ids = [row["id"] for row in rows]
            outputs = saved.get("worker_outputs")
            outputs["0"].update(complete=True, terminal_error=capacity_error,
                                 terminal_text="PARTIAL: The owner discovered more scope than can fit its bounded continuation context.",
                                 usage_processed=True)
            outputs["0"].pop("owner_obligation_ids", None)
            self.assertLessEqual(len(outputs["0"]["text"]), 6000)
            self.assertIn(instructions[-1], outputs["0"]["followups"])
            await saved.save(owner_obligations={"0": rows}, owner_delivery={"0:low": 0},
                             worker_outputs=outputs, output={"error": capacity_error, "events": []})

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(page_count, 2)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(len({call["native_id"] for call in self.calls}), 1)
        self.assertEqual(len({call["key"] for call in self.calls}), 1)
        rows = self.record.get("owner_obligations")["0"]
        self.assertEqual([row["id"] for row in rows[:48]], previous_ids)
        self.assertEqual([row["instruction"] for row in rows], instructions)
        self.assertTrue(all(row["resolved"] for row in rows))
        self.assertEqual(self.record.get("cursor"), 1)
        self.assertEqual(self.record.get("usage_processed_audit"), {"input_tokens": 7})
        self.assertEqual(self.internal_labels, ["Flexible planner"])

    async def test_resolved_owner_obligations_pruned_beyond_retention_cap_with_delivery_shift(self):
        async def drive(*_args, **_kwargs):
            return AgentResult(text="Done"), False

        await self.invoke(drive)
        async with execution_state.execution_scope(self.ws, "low-default", self.request):
            saved = await execution_state.record("flexible_plan")
            rows = []
            for idx in range(150):
                inst = f"Resolved obligation {idx}"
                scope = json_digest(flexible.instruction_identity(inst))
                rows.append({
                    "id": json_digest([scope, 0]), "scope": scope, "generation": 0,
                    "instruction": inst, "repeats": 0, "resolved": True,
                })
            unresolved_inst = "Still unresolved obligation"
            unresolved_scope = json_digest(flexible.instruction_identity(unresolved_inst))
            rows.append({
                "id": json_digest([unresolved_scope, 0]), "scope": unresolved_scope, "generation": 0,
                "instruction": unresolved_inst, "repeats": 1, "resolved": False,
            })
            await saved.save(owner_obligations={"0": rows}, owner_delivery={"0:low": 140})

            self.assertTrue(await agent._continue_flexible_owner("New judge obligation", "latest"))
            kept = saved.get("owner_obligations")["0"]
            delivery = saved.get("owner_delivery")["0:low"]
            resolved_kept = [row for row in kept if row["resolved"]]
            unresolved_kept = [row for row in kept if not row["resolved"]]
            self.assertEqual(len(resolved_kept), agent._MAX_RETAINED_RESOLVED_OBLIGATIONS)
            self.assertEqual(resolved_kept[0]["instruction"], "Resolved obligation 22")
            self.assertEqual(resolved_kept[-1]["instruction"], "Resolved obligation 149")
            self.assertEqual([row["instruction"] for row in unresolved_kept], [unresolved_inst, "New judge obligation"])
            # 22 dropped rows before delivery frontier 140 -> 140 - 22 = 118.
            self.assertEqual(delivery, 118)
            self.assertLessEqual(delivery, len(kept))

