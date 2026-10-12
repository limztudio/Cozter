"""Whole requests finish and release their queue despite expanding agent work."""

import asyncio
import gc
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
        self.assertFalse(result.continue_instruction)
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

    async def test_terminal_tool_budget_delivers_partial_without_judge_or_more_workers(self):
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

    async def test_autochain_shares_budget_without_duplicating_artifacts_or_accounting(self):
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
                text="Draft", continue_instruction="finish remaining scope",
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
        self.assertIs(seen[0], seen[1])
        self.assertEqual(result.usage, {"input_tokens": 10})
        self.assertEqual([event.content for event in result.events if event.kind == "attachment"], [artifact])
        self.assertIn("Initial evidence", result.text)

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

    async def test_timed_out_request_delivers_partial_and_next_queue_entry_starts(self):
        self.enterContext(mock.patch.object(workspace, "CONFIG_DIR", self.ws))
        bot = _ReplyDeliveryBot(self.ws, fail_final=False)
        started = []

        async def implementation(turn, prompt, *_args, **_kwargs):
            started.append(prompt)
            turn.session_id = self.data["id"]
            if prompt == "slow task":
                timer = agent._TURN_TIMEOUT.get()
                timer.reschedule(asyncio.get_running_loop().time() + 0.03)
                await asyncio.Event().wait()
            return AgentResult(text="Next task completed", events=[ChatEvent(kind="text", content="Next task completed")])

        for text in ("slow task", "next task"):
            entry = await bot._persist_enqueue("u1", text, "chat")
            bot._ensure_message_queue("u1").put_nowait((text, "chat", entry, False))
        with (mock.patch.object(agent, "_run_turn_impl", implementation),
              mock.patch.object(workspace, "get_run_config", return_value=("flexible", "", "", "full", "meta"))):
            await asyncio.wait_for(bot._drain_message_queue("u1"), timeout=2)
        self.assertEqual(started, ["slow task", "next task"])
        self.assertTrue(any("PARTIAL" in text for text in bot.sent))
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

