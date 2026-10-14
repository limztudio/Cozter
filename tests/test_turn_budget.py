"""Execution limits retain enough progress to finish the same queued request."""

import asyncio
import gc
import json
import os
import tempfile
import unittest
import weakref
from unittest import mock

from Cozter import agent, config, flexible, session, workspace
from Cozter.backends_agent.base import AgentResult, ChatEvent, append_text_result
from Cozter.backends_bot.base import _InjectQueue
from Cozter.tests.helpers import create_python_script_process, temporary_config
from Cozter.tests.test_reply_delivery import _ReplyDeliveryBot


class TurnBudgetConfigTests(unittest.TestCase):
    def test_default_explicit_disable_and_invalid_overrides(self):
        for value, expected in ((None, 3600), (0, 0), (1800, 1800),
                                (False, 3600), (-1, 3600), ("0", 3600)):
            with self.subTest(value=value), temporary_config({"agent_turn_timeout": value}):
                self.assertEqual(config.get_agent_turn_timeout(), expected)


class SynchronousTurnBudgetTests(unittest.TestCase):
    def test_budget_can_be_constructed_before_event_loop_startup(self):
        budget = agent.TurnBudget(3600)
        self.assertIsNone(budget.owner)
        self.assertFalse(budget.started)
        self.assertIsNone(budget.deadline)

    def test_synchronous_context_reuses_and_restores_budget(self):
        with mock.patch.object(config, "get_agent_turn_timeout", return_value=3600):
            with agent.turn_budget() as budget:
                with agent.turn_budget() as nested:
                    self.assertIs(nested, budget)
                with agent.turn_budget(fresh=True) as renewed:
                    self.assertIsNot(renewed, budget)
                self.assertIs(agent._TURN_BUDGET.get(), budget)
        self.assertIsNone(agent._TURN_BUDGET.get())


class _BlockingBackend:
    name = "budget-test"
    supports_typed_plugins = True
    default_summary_model = "offline"

    def __init__(self):
        self.proc = None
        self.saw_output = False

    async def launch(self, *_args, **_kwargs):
        self.proc = await create_python_script_process(
            'import json,time; print(json.dumps({"text":"Active partial output"}), flush=True); time.sleep(60)'
        )
        return self.proc

    def parse_event(self, event, result):
        self.saw_output = True
        append_text_result(result, event["text"])
        result.events.append(ChatEvent(kind="attachment", content="partial.txt"))
        result.usage = {"input_tokens": 2}


class TurnBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = self.tmp.name
        self.data = await asyncio.to_thread(session.create_session, self.ws, name="Budget test")
        for tier in flexible.TIERS:
            workspace.set_flexible_backend_name(self.ws, tier, "meta")
        workspace.set_reasoning_effort(self.ws, 100)
        self.enterContext(mock.patch.object(agent, "_run_post_turn_maintenance", mock.AsyncMock()))
        self.enterContext(mock.patch.object(config, "get_agent_turn_timeout", return_value=3600))

    async def asyncTearDown(self):
        await agent._wait_for_pending_session_writes(self.ws)
        await asyncio.sleep(0)

    async def run_flexible(self, **kwargs):
        return await agent.run(
            "Fix and verify requested scope", self.ws, 1, backend_name="flexible",
            summary_backend_name="meta", session_id=self.data["id"], **kwargs,
        )

    async def test_worker_allowance_is_shared_with_judge_followup(self):
        artifact = os.path.join(self.ws, "result.txt")
        with open(artifact, "w", encoding="utf-8") as output:
            output.write("completed work")
        plans = mock.AsyncMock(side_effect=[
            "[PLAN]\n1. [low] repair retry\n[/PLAN]",
            "[PLAN]\n1. [low] verify remaining runtime\n[/PLAN]",
        ])
        worker = mock.AsyncMock(return_value=(AgentResult(
            text="Retry fixed; tests passed.\n[[attach: result.txt]]",
            events=[ChatEvent(kind="text", content="[[attach: result.txt]]")],
            usage={"input_tokens": 10},
        ), False))
        judge = mock.AsyncMock(return_value=(flexible.JudgeVerdict(
            True, "runtime", "verify runtime",
        ), False))
        with agent.turn_budget() as budget:
            budget.max_workers = 1
            with (mock.patch.object(agent, "run_internal_backend", plans),
                  mock.patch.object(agent, "_drive_backend", worker),
                  mock.patch.object(agent, "_judge_draft", judge)):
                result = await self.run_flexible()
        self.assertEqual(worker.await_count, 1)
        self.assertEqual(judge.await_count, 1)
        self.assertIn("PARTIAL", result.text)
        self.assertIn("Retry fixed; tests passed", result.text)
        self.assertIn("verify remaining runtime", result.text)
        self.assertEqual(result.usage, {"input_tokens": 10})
        self.assertEqual([event.content for event in result.events if event.kind == "attachment"], [artifact])
        self.assertIn("Turn budget exhausted", result.error)
        self.assertTrue(result.continue_instruction)
        self.assertEqual(result.continuation_reason, "worker_budget")
        self.assertEqual(result.continuation_checkpoint["pending"], ["verify remaining runtime"])
        self.assertEqual(result.continuation_checkpoint["reports"], [{
            "instruction": "repair retry", "report": "Retry fixed; tests passed.",
        }])
        self.assertIn("PARTIAL", session.load_session(self.ws, self.data["id"])["messages"][-1]["content"])

    async def test_injected_replanning_does_not_reset_worker_allowance(self):
        async def interrupted(*_args, injected, **_kwargs):
            injected.append("inspect revised scope")
            return AgentResult(text="Interrupted work"), True

        worker = mock.AsyncMock(side_effect=interrupted)
        with agent.turn_budget() as budget:
            budget.max_workers = 1
            with (
                mock.patch.object(agent, "run_internal_backend", mock.AsyncMock(side_effect=[
                    "[PLAN]\n1. [low] original scope\n[/PLAN]",
                    "[PLAN]\n1. [low] revised scope\n[/PLAN]",
                ])),
                mock.patch.object(agent, "_drive_backend", worker),
                mock.patch.object(agent, "_judge_draft", mock.AsyncMock()) as judge,
            ):
                result = await self.run_flexible(inject_queue=_InjectQueue(maxsize=10))
        self.assertEqual(worker.await_count, 1)
        judge.assert_not_awaited()
        self.assertIn("revised scope", result.text)

    async def test_expiry_cancels_backend_and_retains_completed_reports_and_usage(self):
        backend = _BlockingBackend()
        original_driver = agent._drive_backend
        calls = 0

        async def drive(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return AgentResult(text="Completed repair evidence", usage={"input_tokens": 10}), False
            return await original_driver(backend, *args[1:], **kwargs)

        with agent.turn_budget() as budget:
            budget.started = True
            budget.deadline = asyncio.get_running_loop().time() + 1
            with (
                mock.patch.object(agent, "run_internal_backend", mock.AsyncMock(return_value=
                    "[PLAN]\n1. [low] repair code\n2. [low] verify runtime\n3. [low] final release check\n[/PLAN]")),
                mock.patch.object(agent, "_drive_backend", side_effect=drive),
                mock.patch.object(agent, "_judge_draft", mock.AsyncMock()) as judge,
            ):
                async with asyncio.timeout(4):
                    result = await self.run_flexible()
        self.assertTrue(backend.saw_output)
        self.assertIsNotNone(backend.proc.returncode)
        judge.assert_not_awaited()
        self.assertIn("Completed repair evidence", result.text)
        self.assertIn("Active partial output", result.text)
        self.assertIn("verify runtime", result.text)
        self.assertIn("final release check", result.text)
        self.assertEqual(result.usage, {"input_tokens": 12})
        self.assertTrue(any(event.kind == "attachment" for event in result.events))
        self.assertEqual(result.continuation_reason, "time_budget")
        self.assertEqual(result.continuation_checkpoint["pending"], ["verify runtime", "final release check"])
        self.assertEqual(result.continuation_checkpoint["reports"], [{
            "instruction": "repair code", "report": "Completed repair evidence",
        }])
        self.assertEqual(result.continuation_checkpoint["streamed_output"], "Active partial output")

    async def test_expiry_during_merge_preserves_worker_reports(self):
        async def internal(*_args, label, **_kwargs):
            if label == "Flexible planner":
                return "[PLAN]\n1. [low] first\n2. [low] second\n[/PLAN]"
            await asyncio.Event().wait()

        with agent.turn_budget() as budget:
            budget.started = True
            budget.deadline = asyncio.get_running_loop().time() + 0.05
            with (
                mock.patch.object(agent, "run_internal_backend", internal),
                mock.patch.object(agent, "_drive_backend", mock.AsyncMock(side_effect=[
                    (AgentResult(text="First verification evidence"), False),
                    (AgentResult(text="Second verification evidence"), False),
                ])),
            ):
                result = await self.run_flexible()
        self.assertIn("First verification evidence", result.text)
        self.assertIn("Second verification evidence", result.text)
        self.assertIn("completion verification remains unfinished", result.text)
        self.assertEqual(result.continuation_checkpoint["pending"], [
            "Verify completion of the full original request and finish any remaining work.",
        ])

    async def test_tool_budget_retains_partial_and_pending_scope_for_next_attempt(self):
        with (
            mock.patch.object(agent, "run_internal_backend", mock.AsyncMock(return_value=
                "[PLAN]\n1. [low] inspect code\n2. [low] verify code\n[/PLAN]")),
            mock.patch.object(agent, "_drive_backend", mock.AsyncMock(return_value=(
                AgentResult(text="Evidence gathered", error="Tool budget exhausted after 320 tool rounds."), False,
            ))) as driver,
            mock.patch.object(agent, "_judge_draft", mock.AsyncMock()) as judge,
        ):
            result = await self.run_flexible()
        driver.assert_awaited_once()
        judge.assert_not_awaited()
        self.assertIn("Evidence gathered", result.text)
        self.assertIn("verify code", result.text)
        self.assertEqual(result.continuation_reason, "tool_budget")
        self.assertEqual(result.continuation_checkpoint["pending"], ["inspect code", "verify code"])
        self.assertIn("Evidence gathered", result.continuation_checkpoint["reports"][0]["report"])

    async def test_direct_backend_tool_cap_is_partial_and_cannot_be_judged_again(self):
        with (
            mock.patch.object(agent.backends_agent, "get_backend", return_value=_BlockingBackend()),
            mock.patch.object(agent, "_drive_backend", mock.AsyncMock(return_value=(
                AgentResult(text="Repair evidence", error="Tool budget exhausted after 40 tool rounds."), False,
            ))),
            mock.patch.object(agent, "_judge_draft", mock.AsyncMock()) as judge,
        ):
            result = await agent.run(
                "repair requested scope", self.ws, 1, backend_name="budget-test", session_id=self.data["id"],
            )
        judge.assert_not_awaited()
        self.assertTrue(result.text.startswith("PARTIAL"))
        self.assertIn("Repair evidence", result.text)
        self.assertEqual(result.continuation_reason, "tool_budget")
        self.assertEqual(result.continuation_checkpoint["latest_draft"], "Repair evidence")
        self.assertEqual(result.continuation_checkpoint["pending"], [agent._DEFAULT_PENDING_SCOPE])

    async def test_artifact_discovered_after_result_capture_survives_judge_timeout(self):
        artifact = os.path.join(self.ws, "late.txt")
        with open(artifact, "w", encoding="utf-8") as output:
            output.write("late artifact")

        async def drive(*_args, **_kwargs):
            result = AgentResult(text="Repair completed", usage={"input_tokens": 3})
            agent._TURN_BUDGET.get().capture(result, self.ws)
            return result, False

        async def judge(*_args, **_kwargs):
            agent._TURN_TIMEOUT.get().reschedule(asyncio.get_running_loop().time() + 0.03)
            await asyncio.Event().wait()

        with (
            mock.patch.object(agent.backends_agent, "get_backend", return_value=_BlockingBackend()),
            mock.patch.object(agent, "_drive_backend", drive),
            mock.patch.object(agent, "_collect_new_attachment_images", return_value=[artifact]),
            mock.patch.object(agent, "_judge_draft", judge),
        ):
            result = await agent.run(
                "repair requested scope", self.ws, 1, backend_name="budget-test", session_id=self.data["id"],
            )
        self.assertEqual(result.usage, {"input_tokens": 3})
        self.assertEqual([event.content for event in result.events if event.kind == "attachment"], [artifact])

    async def test_job_continuation_renews_budget_without_duplicating_artifacts(self):
        seen = []
        bot = _ReplyDeliveryBot(self.ws, fail_final=False)
        artifact = os.path.join(self.ws, "result.txt")

        async def implementation(turn, *_args, **_kwargs):
            budget = agent._TURN_BUDGET.get()
            seen.append(budget)
            budget.max_workers = 1
            turn.session_id = self.data["id"]
            budget.pending = ["finish remaining scope"]
            budget.claim_worker()
            result = AgentResult(
                text="Draft" if len(seen) == 1 else "Full scope completed",
                continue_instruction="finish remaining scope" if len(seen) == 1 else "",
                events=[ChatEvent(kind="attachment", content=artifact)], usage={"input_tokens": 10},
                session_id=self.data["id"],
            )
            budget.reports.append(("initial work", "Initial evidence"))
            budget.capture(result, self.ws)
            return result

        with (
            mock.patch.object(agent, "_run_turn_impl", implementation),
            mock.patch.object(bot, "_send_result", mock.AsyncMock()) as send,
            mock.patch.object(workspace, "get_run_config", return_value=("flexible", "", "", "full", "meta")),
        ):
            await bot._run_turn("u1", "chat", "original scope")
        result = send.await_args.args[2]
        self.assertEqual(len(seen), 2)
        self.assertIsNot(seen[0], seen[1])
        self.assertEqual(result.usage, {"input_tokens": 20})
        self.assertEqual([event.content for event in result.events if event.kind == "attachment"], [artifact])
        self.assertEqual(result.text, "Full scope completed")

    async def test_expired_shared_deadline_retains_pinned_session(self):
        with agent.turn_budget() as budget:
            budget.started = True
            budget.deadline = asyncio.get_running_loop().time() - 1
            async def blocked(*_args, **_kwargs):
                await asyncio.Event().wait()
            with mock.patch.object(agent, "_run_turn_impl", blocked):
                result = await self.run_flexible()
        self.assertEqual(result.session_id, self.data["id"])
        self.assertIn("PARTIAL", session.load_session(self.ws, self.data["id"])["messages"][-1]["content"])

    async def test_fresh_attempt_replaces_allowance_and_restores_outer_scope(self):
        with agent.turn_budget() as original:
            original.started = True
            original.deadline = asyncio.get_running_loop().time() - 1
            original.workers_started = original.max_workers
            with agent.turn_budget(fresh=True) as following:
                self.assertIsNot(following, original)
                self.assertFalse(following.started)
                following.claim_worker()
                self.assertEqual(following.workers_started, 1)
                with agent.turn_budget() as nested:
                    self.assertIs(nested, following)
            self.assertIs(agent._TURN_BUDGET.get(), original)

    async def test_budget_expiry_retains_accepted_injections_in_resume_request(self):
        queue = _InjectQueue(maxsize=10)

        async def exhausted(turn, *_args, **_kwargs):
            turn.injected.append("Keep the first accepted correction")
            queue.put_nowait("Also verify the deployment")
            raise agent._TurnBudgetExhaustedError("worker limit reached")

        with mock.patch.object(agent, "_run_turn_impl", exhausted):
            result = await self.run_flexible(inject_queue=queue)
        self.assertEqual(queue.put_if_active("Too late"), "finished")
        self.assertTrue(queue.empty())
        self.assertIn("Fix and verify requested scope", result.continuation_checkpoint["request"])
        self.assertIn("Keep the first accepted correction", result.continuation_checkpoint["request"])
        self.assertIn("Also verify the deployment", result.continuation_checkpoint["request"])
        self.assertFalse(result.continuation_checkpoint["request_truncated"])
        self.assertEqual(result.continuation_checkpoint["accepted_additions"], [
            "Keep the first accepted correction", "Also verify the deployment",
        ])
        self.assertFalse(result.continuation_checkpoint["accepted_additions_truncated"])

    async def test_checkpoint_bounds_large_accepted_additions_with_omission_evidence(self):
        with agent.turn_budget() as budget:
            additions = ["start\n" + "x" * 100000 + "\nfinish"]
            final = budget.finish("worker cap reached", agent._InterruptedTurn(
                self.ws, "request", injected=additions,
            ))
        checkpoint = final.continuation_checkpoint
        self.assertTrue(checkpoint["truncated"])
        self.assertTrue(checkpoint["accepted_additions_truncated"])
        self.assertTrue(checkpoint["accepted_additions"][0].startswith("start"))
        self.assertTrue(checkpoint["accepted_additions"][0].endswith("finish"))
        self.assertIn("middle omitted", checkpoint["accepted_additions"][0])
        self.assertLess(len(json.dumps(checkpoint)), 75000)

    async def test_full_accepted_additions_survive_a_bounded_checkpoint_in_session(self):
        queue = _InjectQueue(maxsize=10)
        addition = "start\n" + "x" * 100000 + "\nfinish"

        async def exhausted(*_args, **_kwargs):
            queue.put_nowait(addition)
            raise agent._TurnBudgetExhaustedError("worker limit reached")

        with mock.patch.object(agent, "_run_turn_impl", exhausted):
            result = await self.run_flexible(inject_queue=queue)
        self.assertTrue(result.continuation_checkpoint["accepted_additions_truncated"])
        messages = session.load_session(self.ws, self.data["id"])["messages"]
        saved_request = next(message["content"] for message in reversed(messages) if message["role"] == "user")
        self.assertTrue(saved_request.endswith(addition))
        self.assertIn("Fix and verify requested scope", saved_request)

    async def test_routed_session_is_persisted_before_worker_work(self):
        session.set_last_session(self.ws, 1, self.data["id"])
        order = []

        async def pin_session(session_id):
            order.append(("pinning", session_id))
            await asyncio.sleep(0)
            order.append(("pinned", session_id))

        async def drive(*_args, **_kwargs):
            self.assertEqual(order[-1], ("pinned", self.data["id"]))
            order.append(("work", self.data["id"]))
            return AgentResult(text="Work complete"), False

        with (
            mock.patch.object(agent.backends_agent, "get_backend", return_value=_BlockingBackend()),
            mock.patch.object(agent, "_drive_backend", drive),
            mock.patch.object(agent, "_judge_draft", mock.AsyncMock(return_value=(
                flexible.JudgeVerdict(False), False,
            ))),
        ):
            result = await agent.run(
                "repair requested scope", self.ws, 1, backend_name="budget-test", on_session=pin_session,
            )
        self.assertEqual(order, [("pinning", self.data["id"]), ("pinned", self.data["id"]),
                                 ("work", self.data["id"])])
        self.assertEqual(result.session_id, self.data["id"])
        self.assertTrue(result.completion_verified)

    async def test_independent_child_tasks_do_not_inherit_another_requests_allowance(self):
        with agent.turn_budget() as parent:
            parent.start()
            parent.workers_started = 24
            async def child():
                with agent.turn_budget() as child_budget:
                    self.assertIsNot(child_budget, parent)
                    self.assertFalse(child_budget.started)
                    child_budget.claim_worker()
                    return child_budget.workers_started
            self.assertEqual(await asyncio.create_task(child()), 1)
            self.assertEqual(parent.workers_started, 24)

    async def test_budget_does_not_retain_transcripts_and_final_report_is_bounded(self):
        with agent.turn_budget() as budget:
            result = AgentResult(events=[ChatEvent(kind="tool", content="x" * 100000)], usage={"input_tokens": 3})
            reference = weakref.ref(result)
            budget.capture(result, self.ws)
            del result
            gc.collect()
            self.assertIsNone(reference())
            self.assertEqual(budget.output.events, [])
            budget.reports = [(f"task {index}", "detail\n" * 4000) for index in range(24)]
            budget.pending = ["scope " + "long " * 1000 for _ in range(24)]
            turn = agent._InterruptedTurn(self.ws, "request", events=[ChatEvent(kind="tool", content="x" * 10000)])
            final = budget.finish("worker cap reached", turn)
        self.assertLess(len(final.text), 14000)
        self.assertIn("middle omitted", final.text)
        self.assertEqual(final.usage, {"input_tokens": 3})
        self.assertTrue(final.continuation_checkpoint["truncated"])
        self.assertEqual(len(final.continuation_checkpoint["reports"]), 24)
        self.assertLess(len(json.dumps(final.continuation_checkpoint)), 75000)

    async def test_checkpoint_retains_worker_evidence_omitted_from_partial_reply(self):
        with agent.turn_budget() as budget:
            budget.reports = [(f"task {index}", "Opening\n" + str(index) * 3500 + "\nVerified")
                              for index in range(5)]
            budget.pending = ["Finish verification and push the completed work"]
            final = budget.finish("worker cap reached", agent._InterruptedTurn(self.ws, "request"))
        self.assertIn("middle omitted", final.text)
        self.assertEqual(final.continuation_checkpoint["reports"][2]["report"], budget.reports[2][1])
        self.assertEqual(final.continuation_checkpoint["pending"], budget.pending)
        self.assertFalse(final.continuation_checkpoint["truncated"])

    async def test_provider_error_is_not_resumable_budget_exhaustion(self):
        with (
            mock.patch.object(agent.backends_agent, "get_backend", return_value=_BlockingBackend()),
            mock.patch.object(agent, "_drive_backend", mock.AsyncMock(return_value=(
                AgentResult(text="Account unavailable", error="invalid API key"), False,
            ))),
            mock.patch.object(agent, "_judge_draft", mock.AsyncMock()) as judge,
        ):
            result = await agent.run(
                "repair requested scope", self.ws, 1, backend_name="budget-test", session_id=self.data["id"],
            )
        judge.assert_not_awaited()
        self.assertEqual(result.error, "invalid API key")
        self.assertIsNone(result.continuation_reason)
        self.assertIsNone(result.continuation_checkpoint)
        self.assertFalse(result.continue_instruction)

    async def test_unavailable_or_malformed_judge_retains_draft_without_verifying_completion(self):
        for reply in (None, "", "looks fine", "[VERDICT]\nCONTINUE\n[/VERDICT]",
                      OSError("judge executable unavailable"), RuntimeError("judge could not run")):
            with (
                self.subTest(reply=reply),
                mock.patch.object(agent.backends_agent, "get_backend", return_value=_BlockingBackend()),
                mock.patch.object(agent, "_drive_backend", mock.AsyncMock(return_value=(
                    AgentResult(text="Repaired the requested scope", completion_verified=True), False,
                ))),
                mock.patch.object(agent, "run_internal_backend", mock.AsyncMock(
                    side_effect=reply if isinstance(reply, Exception) else None,
                    return_value=reply if not isinstance(reply, Exception) else None,
                )),
            ):
                result = await agent.run(
                    "repair requested scope", self.ws, 1, backend_name="budget-test", session_id=self.data["id"],
                )
            self.assertEqual(result.text, "Repaired the requested scope")
            self.assertIsNone(result.error)
            self.assertFalse(result.completion_verified)
            self.assertFalse(result.continue_instruction)

    async def test_unavailable_tail_judge_does_not_verify_a_followup_result(self):
        with (
            mock.patch.object(agent.backends_agent, "get_backend", return_value=_BlockingBackend()),
            mock.patch.object(agent, "_drive_backend", mock.AsyncMock(return_value=(
                AgentResult(text="Repaired the requested scope"), False,
            ))) as worker,
            mock.patch.object(flexible, "JUDGE_MAX_CONTINUES", 1),
            mock.patch.object(agent, "_judge_draft", mock.AsyncMock(side_effect=[
                (flexible.JudgeVerdict(True, "verification", "verify remaining work"), False),
                (flexible.JudgeVerdict(False, verified=False), False),
            ])),
        ):
            result = await agent.run(
                "repair requested scope", self.ws, 1, backend_name="budget-test", session_id=self.data["id"],
            )
        self.assertEqual(worker.await_count, 2)
        self.assertEqual(result.text, "Repaired the requested scope")
        self.assertFalse(result.completion_verified)
        self.assertFalse(result.continue_instruction)

    async def test_cancel_during_budget_result_write_propagates_and_owned_write_finishes(self):
        write_started = asyncio.Event()
        write_release = asyncio.Event()

        async def write(*_args, **_kwargs):
            write_started.set()
            await write_release.wait()
            return True

        async def exhausted(*_args, **_kwargs):
            raise agent._TurnBudgetExhaustedError("worker limit reached")

        with (mock.patch.object(agent, "_run_turn_impl", exhausted),
              mock.patch.object(agent, "_persist_session_write", write)):
            active = asyncio.create_task(self.run_flexible())
            await asyncio.wait_for(write_started.wait(), timeout=1)
            active.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(active, timeout=1)
            write_release.set()
            await agent._wait_for_pending_session_writes(self.ws)

    async def test_timed_out_request_finishes_remainder_before_next_queue_entry(self):
        self.enterContext(mock.patch.object(workspace, "CONFIG_DIR", self.ws))
        bot = _ReplyDeliveryBot(self.ws, fail_final=False)
        started = []

        async def implementation(turn, prompt, *_args, **_kwargs):
            started.append(prompt)
            turn.session_id = self.data["id"]
            if prompt == "slow task" and len(started) == 1:
                timer = agent._TURN_TIMEOUT.get()
                timer.reschedule(asyncio.get_running_loop().time() + 0.03)
                await asyncio.Event().wait()
            answer = "Slow task completed" if "slow task" in prompt else "Next task completed"
            return AgentResult(text=answer, events=[ChatEvent(kind="text", content=answer)], session_id=self.data["id"])

        for text in ("slow task", "next task"):
            entry = await bot._persist_enqueue("u1", text, "chat")
            bot._ensure_message_queue("u1").put_nowait((text, "chat", entry, False))
        with (mock.patch.object(agent, "_run_turn_impl", implementation),
              mock.patch.object(workspace, "get_run_config", return_value=("flexible", "", "", "full", "meta"))):
            await asyncio.wait_for(bot._drain_message_queue("u1"), timeout=2)
        self.assertEqual(started[0], "slow task")
        self.assertIn("slow task", started[1])
        self.assertEqual(started[2], "next task")
        self.assertLess(bot.sent.index("Slow task completed"), bot.sent.index("Next task completed"))
        self.assertIn("Next task completed", bot.sent)
        self.assertEqual(bot._read_queue_file(), {})
        self.assertEqual(await bot._list_reply_delivery_records(), [])


class VerificationConvergencePromptTests(unittest.TestCase):
    def test_expected_diagnostics_and_unavailable_services_do_not_expand_scope(self):
        plan = flexible.Plan("verify", (flexible.Subtask("low", "verify tests"),))
        for prompt in (
            flexible.build_plan_prompt("request", collaborative=False),
            flexible.build_subtask_prompt("request", plan, 0, []),
            flexible.build_merge_prompt("request", plan, ["passed"], collaborative=False, blocked=[]),
            flexible.build_judge_prompt("request", "passed with expected negative-path warnings"),
        ):
            self.assertIn("negative-path", prompt)
            self.assertIn("platform skips", prompt)
        judge = flexible.build_judge_prompt("request", "unconfigured service")
        self.assertIn("current environment", judge)
        self.assertIn("unconfigured service", judge)
