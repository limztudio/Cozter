"""Regressions for interactive continuations, schedule origins and safe refresh."""

import asyncio
from collections import deque
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from Cozter import agent, flexible, session, workspace
from Cozter.backends_agent.base import AgentResult, ChatEvent
from Cozter.backends_bot.base import BotPlatform, MessageHandle, _RecoverableTurnStatus
from Cozter.tests.helpers import TestBot
from Cozter.tests.test_agent_post_turn import _ImmediateBackend
from Cozter.tests.test_reply_delivery import _ReplyDeliveryBot


class InteractiveContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.object(workspace, "CONFIG_DIR", self.ws))
        workspace.set_interaction_style(self.ws, "collaborative")
        self.sid = session.create_session(self.ws, name="Interactive")['id']

    async def test_pinned_session_preserves_policy_through_public_agent_run(self):
        for scheduled in (False, True):
            with (
                self.subTest(scheduled=scheduled),
                mock.patch.object(agent.backends_agent, "get_backend", return_value=_ImmediateBackend()),
                mock.patch.object(agent, "_drive_backend", new=mock.AsyncMock(
                    return_value=(AgentResult(text="Completed"), False),
                )),
                mock.patch.object(agent, "_judge_draft", new=mock.AsyncMock(
                    return_value=(flexible.JudgeVerdict(should_continue=False, verified=True), False),
                )),
                mock.patch.object(agent, "_run_post_turn_maintenance", new=mock.AsyncMock()),
                mock.patch.object(agent, "_build_backend_prompt", wraps=agent._build_backend_prompt) as build,
            ):
                await agent.run("Finish work", self.ws, "u1", session_id=self.sid,
                                is_scheduled=scheduled)
                self.assertEqual(build.call_args.kwargs['collaborative'], not scheduled)
                self.assertEqual(build.call_args.kwargs['allow_detached_requests'], not scheduled)
                await asyncio.sleep(0)

    async def test_pinned_bot_turn_preserves_background_and_delivery_ownership(self):
        bot = _ReplyDeliveryBot(self.ws, fail_final=False)
        for scheduled in (False, True):
            result = AgentResult(text="Completed", session_id=self.sid)
            with (
                self.subTest(scheduled=scheduled),
                mock.patch.object(agent, "run", new=mock.AsyncMock(return_value=result)) as run,
                mock.patch.object(bot, "_launch_result_detached_tasks", new=mock.AsyncMock()) as launch,
                mock.patch.object(bot, "_register_result_detached_tasks", new=mock.AsyncMock()) as register,
                mock.patch.object(bot, "_stage_reply_delivery", new=mock.AsyncMock(return_value={})) as stage,
                mock.patch.object(bot, "_deliver_staged_reply", new=mock.AsyncMock()) as deliver,
            ):
                await bot._run_turn_with_budget(
                    "u1", "chat", "Finish work", workspace_path=self.ws,
                    session_id=self.sid, queue_entry_id="entry", is_scheduled=scheduled,
                )
                self.assertEqual(run.call_args.kwargs['is_scheduled'], scheduled)
                self.assertEqual(launch.await_count, 0 if scheduled else 1)
                self.assertEqual(register.call_args.kwargs['session_id'], None if scheduled else self.sid)
                self.assertEqual(stage.call_args.kwargs['allow_await'], not scheduled)
                self.assertEqual(deliver.call_args.kwargs['awaiting_uid'], None if scheduled else "u1")

    async def test_checkpoint_continuation_retains_interactive_flag(self):
        bot = _ReplyDeliveryBot(self.ws, fail_final=False)
        flags = []
        sessions = []

        async def run(_prompt, _ws, **kwargs):
            flags.append(kwargs['is_scheduled'])
            sessions.append(kwargs['session_id'])
            await kwargs['on_session'](self.sid)
            if len(flags) == 1:
                return AgentResult(
                    text="PARTIAL: verification remains", session_id=self.sid,
                    continuation_reason="time_budget", continue_instruction="Verify remaining work",
                )
            return AgentResult(text="Completed", session_id=self.sid)

        with mock.patch.object(agent, "run", side_effect=run):
            self.assertTrue(await bot._run_turn("u1", "chat", "Finish work"))
        self.assertEqual(flags, [False, False])
        self.assertEqual(sessions, [None, self.sid])


class ScheduleWorkspaceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.origin = self.enterContext(tempfile.TemporaryDirectory())
        self.other = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.object(workspace, "CONFIG_DIR", self.origin))
        self.bot = _ReplyDeliveryBot(self.origin, fail_final=False)

    async def test_admission_pins_origin_before_first_execution_and_restart(self):
        schedule = {'command': 'scheduled work', 'chat_id': 'chat'}
        with mock.patch.object(self.bot, "_start_queue_drain"):
            await self.bot._fire_schedule("u1", schedule, workspace_path=self.origin)
        entry = self.bot._read_queue_file()['u1'][0]
        self.assertEqual(entry['job']['workspace_path'], self.origin)

        # Simulate /open B followed by process restart before any session exists.
        restarted = _ReplyDeliveryBot(self.other, fail_final=False)
        calls = []

        async def run(_uid, _chat, _text, **kwargs):
            checkpoint = await restarted._load_job_checkpoint("u1", entry['id'])
            calls.append((checkpoint['workspace_path'], kwargs['is_scheduled']))
            self.assertIsNotNone(session.load_session(self.origin, kwargs['session_id']))
            self.assertFalse(os.path.isdir(os.path.join(self.other, '.cozter', 'sessions')))
            return True

        with mock.patch.object(restarted, "_run_turn", side_effect=run):
            await restarted._run_ephemeral_turn(
                "u1", "chat", schedule['command'], queue_entry_id=entry['id'],
            )
        self.assertEqual(calls, [(self.origin, True)])

    async def test_missing_origin_pauses_without_creating_workspace(self):
        missing = os.path.join(self.origin, 'missing')
        entry = await self.bot._persist_enqueue(
            "u1", "scheduled work", "chat", ephemeral=True, workspace_path=missing,
        )
        with mock.patch.object(self.bot, "_current_workspace_for_turn", new=mock.AsyncMock()) as current:
            self.assertFalse(await self.bot._run_ephemeral_turn(
                "u1", "chat", "scheduled work", queue_entry_id=entry,
            ))
        current.assert_not_awaited()
        self.assertFalse(os.path.exists(missing))
        self.assertEqual((await self.bot._load_job_checkpoint("u1", entry))['state'], 'paused')

    async def test_legacy_entry_without_origin_pauses_instead_of_using_current_workspace(self):
        entry = await self.bot._persist_enqueue("u1", "scheduled work", "chat")
        with mock.patch.object(self.bot, "_current_workspace_for_turn", new=mock.AsyncMock()) as current:
            self.assertFalse(await self.bot._run_ephemeral_turn(
                "u1", "chat", "scheduled work", queue_entry_id=entry,
            ))
        current.assert_not_awaited()
        self.assertFalse(os.path.isdir(os.path.join(self.origin, '.cozter', 'sessions')))
        saved = await self.bot._load_job_checkpoint("u1", entry)
        self.assertEqual(saved['state'], 'paused')
        self.assertIn("Restore the scheduled job's workspace", saved['instruction'])


class RefreshTests(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_preserves_project_and_job_state_for_every_backend(self):
        with tempfile.TemporaryDirectory() as ws:
            config = Path(ws, '.codex', 'config.toml')
            config.parent.mkdir()
            config.write_text('project-specific configuration', encoding='utf-8')
            state = Path(ws, '.cozter', 'executions', 'job', 'state.json')
            state.parent.mkdir(parents=True)
            state.write_text('job-owned continuation', encoding='utf-8')
            bot = TestBot([])
            ctx = bot.make_context("u1", "chat")
            for backend in workspace.AVAILABLE_BACKENDS:
                with (
                    self.subTest(backend=backend),
                    mock.patch.object(bot, "_require_ws", new=mock.AsyncMock(return_value=ws)),
                    mock.patch.object(workspace, "get_backend_name", return_value=backend),
                    mock.patch.object(bot, "send_text", new=mock.AsyncMock()) as send,
                ):
                    await bot.cmd_refresh(ctx)
                    response = send.call_args.args[1]
                    self.assertIn(backend, response)
                    self.assertIn('no disposable workspace CLI cache', response)
                    self.assertNotIn('refreshed', response)
                    self.assertEqual(config.read_text(encoding='utf-8'), 'project-specific configuration')
                    self.assertEqual(state.read_text(encoding='utf-8'), 'job-owned continuation')


class DetachedLoggingAndStatusTests(unittest.IsolatedAsyncioTestCase):
    async def test_recoverable_status_retains_five_lines_and_full_omitted_count(self):
        status = _RecoverableTurnStatus(TestBot([]), 'u1', 'chat')
        for index in range(30):
            status.on_event(ChatEvent(kind='tool', content=f'step {index}'))
        self.assertEqual(len(status.lines), 5)
        self.assertEqual(status.total_lines, 30)
        self.assertIn('25 older update(s) omitted', status.pending)
        self.assertIn('step 29', status.pending)
        self.assertNotIn('step 0', status.pending)

    async def test_slow_detached_completion_does_not_block_loop_and_cancel_settles_write(self):
        with tempfile.TemporaryDirectory() as ws:
            started = threading.Event()
            release = threading.Event()
            lock = workspace.get_lock(ws)

            def append(*_args):
                started.set()
                if not release.wait(5):
                    raise TimeoutError('test did not release write')
                return 1

            with mock.patch.object(session, 'append_messages', side_effect=append):
                task = asyncio.create_task(agent.log_detached_task_completion(ws, 'session', 'Complete'))
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 2))
                    await asyncio.wait_for(asyncio.sleep(0.01), timeout=.5)
                    task.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(task.done())
                    self.assertTrue(lock.locked())
                finally:
                    release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=2)
                self.assertFalse(lock.locked())

    async def test_live_status_retains_five_lines_and_full_omitted_count(self):
        class StatusBot(_ReplyDeliveryBot):
            async def send_text(self, _chat, _text, *, rich=False):
                return MessageHandle(chat_id=_chat, message_id='thinking')

        with tempfile.TemporaryDirectory() as ws:
            bot = StatusBot(ws, fail_final=False)
            snapshots = []
            original = BotPlatform._compose_thinking_display

            def compose(lines, text, **kwargs):
                self.assertIsInstance(lines, deque)
                snapshots.append((len(lines), kwargs['total_status_lines']))
                return original(lines, text, **kwargs)

            async def run(*_args, **kwargs):
                for index in range(30):
                    await kwargs['on_event'](ChatEvent(kind='tool', content=f'step {index}'))
                return AgentResult(text='Complete')

            with (
                mock.patch.object(agent, 'run', side_effect=run),
                mock.patch.object(bot, '_compose_thinking_display', side_effect=compose),
            ):
                await bot._run_turn_with_budget('u1', 'chat', 'Finish', workspace_path=ws)
                await asyncio.sleep(0)
            self.assertEqual(snapshots[-1], (5, 30))
            self.assertLessEqual(max(length for length, _total in snapshots), 5)
            self.assertIn('25 older update(s) omitted', original(
                deque(['step'] * 5, maxlen=5), '', total_status_lines=30,
            ))


if __name__ == '__main__':
    unittest.main()
