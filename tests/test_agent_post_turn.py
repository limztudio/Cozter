"""Regression coverage for latency-sensitive post-turn maintenance."""

import asyncio
import os
import tempfile
import unittest
from unittest import mock

from Cozter import agent, flexible, session
from Cozter.backends_agent.base import AgentResult, ChatEvent
from Cozter.backends_bot.base import _InjectQueue
from Cozter.tests.helpers import TestBot


class _ImmediateBackend:
    name = "post-turn-test"
    supports_typed_plugins = True
    default_summary_model = "summary-default"


class AgentPostTurnMaintenanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_omitted_summary_model_uses_backend_default(self) -> None:
        maintenance_tasks: list[asyncio.Task] = []
        backend = _ImmediateBackend()

        async def fast_backend(*_args, **_kwargs):
            return AgentResult(text="reply"), False

        original_background_task = agent.create_background_task

        def capture_background_task(coro, *, name, log=None):
            task = original_background_task(coro, name=name, log=log)
            maintenance_tasks.append(task)
            return task

        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path, name="Manual")
            with (
                mock.patch.object(
                    agent.backends_agent, "get_backend", return_value=backend,
                ),
                mock.patch.object(
                    agent, "_drive_backend", side_effect=fast_backend,
                ),
                mock.patch.object(
                    agent.compaction, "maybe_compact", new_callable=mock.AsyncMock,
                ) as compact,
                mock.patch.object(
                    agent.titling, "maybe_auto_title", new_callable=mock.AsyncMock,
                ),
                mock.patch.object(
                    agent, "create_background_task",
                    side_effect=capture_background_task,
                ),
            ):
                result = await agent.run(
                    "hello", workspace_path, 1,
                    backend_name=backend.name, session_id=data["id"],
                )
                await asyncio.gather(*maintenance_tasks)

        self.assertEqual(result.text, "reply")
        self.assertEqual(len(maintenance_tasks), 1)
        self.assertEqual(compact.await_args.args[2], "summary-default")

    async def test_compaction_does_not_block_the_agent_result(self) -> None:
        """A slow compaction must not delay delivery to a chat platform."""
        started = asyncio.Event()
        release = asyncio.Event()
        maintenance_tasks: list[asyncio.Task] = []
        backend = _ImmediateBackend()

        async def slow_compaction(*_args, **_kwargs) -> None:
            started.set()
            await release.wait()

        async def fast_backend(*_args, **_kwargs):
            return AgentResult(text="reply"), False

        original_background_task = agent.create_background_task

        def capture_background_task(coro, *, name, log=None):
            task = original_background_task(coro, name=name, log=log)
            maintenance_tasks.append(task)
            return task

        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path, name="Manual")
            with (
                mock.patch.object(
                    agent.backends_agent, "get_backend", return_value=backend,
                ),
                mock.patch.object(
                    agent, "_drive_backend", side_effect=fast_backend,
                ),
                mock.patch.object(
                    agent.compaction, "maybe_compact", side_effect=slow_compaction,
                ),
                mock.patch.object(
                    agent, "create_background_task",
                    side_effect=capture_background_task,
                ),
            ):
                turn = asyncio.create_task(agent.run(
                    "hello", workspace_path, 1,
                    backend_name=backend.name, session_id=data["id"],
                ))
                try:
                    await asyncio.wait_for(started.wait(), timeout=1)
                    result = await asyncio.wait_for(asyncio.shield(turn), 0.1)
                    self.assertEqual(result.text, "reply")
                    self.assertEqual(len(maintenance_tasks), 1)
                    self.assertFalse(maintenance_tasks[0].done())
                finally:
                    release.set()
                    await turn
                    await asyncio.gather(*maintenance_tasks)


class AgentContinuationTests(unittest.IsolatedAsyncioTestCase):
    def test_repeated_continuations_do_not_multiply_attachment_events(self):
        with tempfile.TemporaryDirectory() as ws:
            artifact = os.path.join(ws, "result.txt")
            with open(artifact, "w", encoding="utf-8") as output:
                output.write("artifact")
            result = AgentResult(events=[
                ChatEvent(kind="attachment", content=artifact),
                ChatEvent(kind="text", content="[[attach: result.txt]]"),
            ])
            for _ in range(3):
                continuation = AgentResult(text="continued")
                agent.carry_forward_turn_output(result, continuation, ws)
                result = continuation
        self.assertEqual([event.kind for event in result.events], ["attachment"])

    async def run_turn(self, ws, drive, judge, *, inject_queue=None):
        backend = _ImmediateBackend()
        data = session.create_session(ws, name="Manual")
        tasks = []
        original = agent.create_background_task

        def capture(coro, *, name, log=None):
            task = original(coro, name=name, log=log)
            tasks.append(task)
            return task

        with (
            mock.patch.object(agent.backends_agent, "get_backend", return_value=backend),
            mock.patch.object(agent, "_drive_backend", side_effect=drive),
            mock.patch.object(agent, "_judge_draft", side_effect=judge),
            mock.patch.object(agent, "_run_post_turn_maintenance", new_callable=mock.AsyncMock),
            mock.patch.object(agent, "create_background_task", side_effect=capture),
            mock.patch.object(flexible, "JUDGE_MAX_CONTINUES", 1),
        ):
            result = await agent.run(
                "finish the task", ws, 1, backend_name=backend.name,
                session_id=data["id"], inject_queue=inject_queue,
            )
            await asyncio.gather(*tasks)
        return result

    async def test_continuation_keeps_previous_draft_artifacts_and_usage(self):
        prompts = []
        calls = 0

        async def drive(_backend, _ws, prompt, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            prompts.append(prompt)
            if calls == 1:
                return AgentResult(
                    text="Stage one completed\n[[attach: result.txt]]",
                    events=[ChatEvent(kind="text", content="Stage one completed\n[[attach: result.txt]]")],
                    usage={"input_tokens": 10, "output_tokens": 5, "total_cost_usd": 0.1},
                ), False
            return AgentResult(
                text="All stages completed",
                usage={"input_tokens": 20, "output_tokens": 6, "total_cost_usd": 0.2},
            ), False

        judge = mock.AsyncMock(side_effect=[
            (flexible.JudgeVerdict(should_continue=True, next_instruction="finish stage two"), False),
            (flexible.JudgeVerdict(should_continue=False), False),
        ])
        with tempfile.TemporaryDirectory() as ws:
            artifact = os.path.join(ws, "result.txt")
            with open(artifact, "w", encoding="utf-8") as output:
                output.write("stage one artifact")
            result = await self.run_turn(ws, drive, judge)
        self.assertIn("Stage one completed", prompts[1])
        self.assertIn("finish stage two", prompts[1])
        self.assertEqual(result.text, "All stages completed")
        self.assertTrue(any(event.kind == "text" and event.content == "All stages completed" for event in result.events))
        self.assertTrue(any(event.kind == "attachment" and event.content == artifact for event in result.events))
        self.assertEqual(result.usage["input_tokens"], 30)
        self.assertEqual(result.usage["output_tokens"], 11)
        self.assertAlmostEqual(result.usage["total_cost_usd"], 0.3)

    async def test_inject_during_cap_judge_restarts_with_added_context(self):
        prompts = []
        inject_queue = asyncio.Queue()
        calls = 0

        async def drive(_backend, _ws, prompt, *_args, **_kwargs):
            prompts.append(prompt)
            return AgentResult(text=f"draft {len(prompts)}"), False

        async def judge(*_args, injected, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return flexible.JudgeVerdict(should_continue=True, next_instruction="finish"), False
            if calls == 2:
                injected.append("also verify output")
                return flexible.JudgeVerdict(should_continue=False), True
            return flexible.JudgeVerdict(should_continue=False), False

        with tempfile.TemporaryDirectory() as ws:
            result = await self.run_turn(ws, drive, judge, inject_queue=inject_queue)
        self.assertEqual(len(prompts), 3)
        self.assertIn("also verify output", prompts[2])
        self.assertEqual(result.text, "draft 3")

    async def test_failed_draft_is_delivered_without_judge_continuation(self):
        async def drive(*_args, **_kwargs):
            return AgentResult(text="partial response", error="backend failed"), False

        judge = mock.AsyncMock(side_effect=AssertionError("failed drafts must not be judged"))
        with tempfile.TemporaryDirectory() as ws:
            result = await self.run_turn(ws, drive, judge)
        self.assertEqual(result.error, "backend failed")
        judge.assert_not_awaited()


    async def test_owned_injection_queue_stays_open_through_judging(self):
        inject_queue = _InjectQueue(maxsize=2)
        prompts = []

        async def drive(_backend, _ws, prompt, *_args, **kwargs):
            self.assertFalse(kwargs["close_inject_on_completion"])
            prompts.append(prompt)
            return AgentResult(text="reply"), False

        async def judge(*_args, **_kwargs):
            if len(prompts) == 1:
                self.assertEqual(inject_queue.put_if_active("new requirement"), "accepted")
            return flexible.JudgeVerdict(should_continue=False), False

        with tempfile.TemporaryDirectory() as ws:
            await self.run_turn(ws, drive, judge, inject_queue=inject_queue)
        self.assertEqual(len(prompts), 2)
        self.assertIn("new requirement", prompts[1])
        self.assertEqual(inject_queue.put_if_active("after completion"), "finished")


class BotContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def test_chained_turn_keeps_artifacts_usage_and_active_inject_queue(self):
        with tempfile.TemporaryDirectory() as ws:
            artifact = os.path.join(ws, "first.txt")
            with open(artifact, "w", encoding="utf-8") as output:
                output.write("first turn output")
            first = AgentResult(
                text="First draft", session_id="session-id", continue_instruction="finish task",
                events=[ChatEvent(kind="text", content="First draft\n[[attach: first.txt]]")],
                usage={"input_tokens": 10},
            )
            second = AgentResult(text="Finished", events=[ChatEvent(kind="text", content="Finished")], usage={"input_tokens": 20})
            bot = TestBot(["u1"])
            calls = 0

            async def run(*_args, **kwargs):
                nonlocal calls
                calls += 1
                self.assertTrue(kwargs["inject_queue"]._accepting)
                kwargs["inject_queue"].close()
                return first if calls == 1 else second

            with (
                mock.patch.object(bot, "_current_workspace_for_turn", new=mock.AsyncMock(return_value=ws)),
                mock.patch.object(bot, "_send_result", new_callable=mock.AsyncMock) as send,
                mock.patch.object(agent, "run", side_effect=run),
                mock.patch.object(agent.workspace_mod, "get_run_config", return_value=("codex", None, None, "auto", "codex")),
            ):
                await bot._run_turn("u1", "chat", "original task")
            result = send.await_args.args[2]
        self.assertEqual(result.text, "Finished")
        self.assertEqual(result.usage, {"input_tokens": 30})
        self.assertTrue(any(event.kind == "attachment" and event.content == artifact for event in result.events))

    async def test_chain_cap_reports_remaining_work(self):
        with tempfile.TemporaryDirectory() as ws:
            bot = TestBot(["u1"])

            async def run(*_args, **_kwargs):
                return AgentResult(text="Draft", events=[ChatEvent(kind="text", content="Draft")], continue_instruction="verify the deployment")

            with (
                mock.patch.object(bot, "_current_workspace_for_turn", new=mock.AsyncMock(return_value=ws)),
                mock.patch.object(bot, "_send_result", new_callable=mock.AsyncMock) as send,
                mock.patch.object(agent, "run", side_effect=run),
                mock.patch.object(agent.workspace_mod, "get_run_config", return_value=("codex", None, None, "auto", "codex")),
                mock.patch.object(flexible, "MAX_AUTO_CHAIN_TURNS", 1),
            ):
                await bot._run_turn("u1", "chat", "original task")
            result = send.await_args.args[2]
        self.assertIn("PARTIAL", result.text)
        self.assertIn("verify the deployment", result.text)


if __name__ == "__main__":
    unittest.main()
