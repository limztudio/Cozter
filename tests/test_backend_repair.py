"""Regression coverage for private Copilot home ownership on failed launches."""

import asyncio
import json
import os
import tempfile
import time
import unittest
from unittest import mock

from Cozter.backends_agent import copilot
from Cozter import agent, workspace
from Cozter.backends_agent.base import DetachedTaskStatus
from Cozter.backends_bot import base as bot_base
from Cozter.backends_bot import signal
from Cozter.tests.helpers import TestBot


class CopilotHomeSetupTests(unittest.TestCase):
    def test_source_home_resolution_failure_removes_allocated_home(self):
        with tempfile.TemporaryDirectory() as directory:
            private_home = os.path.join(directory, "private")
            os.mkdir(private_home)
            with (
                mock.patch.object(copilot.tempfile, "mkdtemp", return_value=private_home),
                mock.patch.dict(os.environ, {"COPILOT_HOME": ""}),
                mock.patch.object(copilot.os.path, "expanduser", side_effect=ValueError("bad home")),
            ):
                with self.assertRaisesRegex(ValueError, "bad home"):
                    copilot._create_isolated_copilot_home()
            self.assertFalse(os.path.exists(private_home))

    def test_interrupted_metadata_copy_removes_allocated_home(self):
        with tempfile.TemporaryDirectory() as directory:
            private_home = os.path.join(directory, "private")
            source_home = os.path.join(directory, "source")
            os.mkdir(private_home)
            os.mkdir(source_home)
            with open(os.path.join(source_home, "config.json"), "w", encoding="utf-8") as handle:
                handle.write("{}")
            with (
                mock.patch.object(copilot.tempfile, "mkdtemp", return_value=private_home),
                mock.patch.dict(os.environ, {"COPILOT_HOME": source_home}),
                mock.patch.object(copilot.shutil, "copy2", side_effect=KeyboardInterrupt),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    copilot._create_isolated_copilot_home()
            self.assertFalse(os.path.exists(private_home))
            self.assertTrue(os.path.isfile(os.path.join(source_home, "config.json")))

    def test_discovery_stop_failure_still_removes_private_home(self):
        with tempfile.TemporaryDirectory() as directory:
            private_home = os.path.join(directory, "private")
            os.mkdir(private_home)
            proc = mock.Mock(stdout=None)
            with (
                mock.patch.object(copilot.shutil, "which", return_value="copilot"),
                mock.patch.object(copilot, "executable_command", return_value=["copilot"]),
                mock.patch.object(copilot, "_create_isolated_copilot_home", return_value=private_home),
                mock.patch.object(copilot.subprocess, "Popen", return_value=proc),
                mock.patch.object(copilot, "_stop_acp_process", side_effect=RuntimeError("stop failed")),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop failed"):
                    copilot.CopilotBackend()._discover_models(directory)
            self.assertFalse(os.path.exists(private_home))


class CopilotLaunchOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_environment_setup_failure_removes_private_home(self):
        with tempfile.TemporaryDirectory() as directory:
            private_home = os.path.join(directory, "private")
            os.mkdir(private_home)
            with (
                mock.patch.object(copilot, "executable_command", return_value=["copilot"]),
                mock.patch.object(copilot, "_create_isolated_copilot_home", return_value=private_home),
                mock.patch.object(os.environ, "copy", side_effect=RuntimeError("environment failed")),
                mock.patch.object(copilot, "create_prompt_subprocess", new=mock.AsyncMock()) as spawn,
            ):
                with self.assertRaisesRegex(RuntimeError, "environment failed"):
                    await copilot.CopilotBackend().launch(directory, "hello", None, "deny")
            spawn.assert_not_awaited()
            self.assertFalse(os.path.exists(private_home))

    async def test_cancelled_launch_removes_private_home(self):
        with tempfile.TemporaryDirectory() as directory:
            private_home = os.path.join(directory, "private")
            os.mkdir(private_home)
            with (
                mock.patch.object(copilot, "executable_command", return_value=["copilot"]),
                mock.patch.object(copilot, "_create_isolated_copilot_home", return_value=private_home),
                mock.patch.object(copilot, "create_prompt_subprocess", side_effect=asyncio.CancelledError),
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await copilot.CopilotBackend().launch(directory, "hello", None, "deny")
            self.assertFalse(os.path.exists(private_home))


class SignalMalformedJsonTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_created_rows_do_not_hide_valid_outbound_provenance(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(workspace, "CONFIG_DIR", directory):
            rows = [{"group": "a" * 64, "kind": "text", "value": "b" * 64, "created": created}
                    for created in (10 ** 1000, -(10 ** 1000), float("inf"), float("nan"))]
            rows.append({"group": "a" * 64, "kind": "text", "value": "c" * 64, "created": time.time()})
            with open(signal._SignalOutboundJournal._path(), "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "records": rows}, handle)
            journal = signal._SignalOutboundJournal()
            await journal.load()
            self.assertEqual(list(journal._records), [("a" * 64, "text", "c" * 64)])

    async def test_nested_corrupt_journal_reports_controlled_transport_error(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(workspace, "CONFIG_DIR", directory):
            with open(signal._SignalOutboundJournal._path(), "w", encoding="utf-8") as handle:
                handle.write("[" * 2000 + "0" + "]" * 2000)
            # Decoder depth limits differ between Python versions; inject the
            # documented decoder failure at the real journal boundary.
            with (mock.patch.object(signal.json, "loads", side_effect=RecursionError("nested JSON")),
                  self.assertRaisesRegex(signal.SignalCliError, "journal is unreadable")):
                await signal._SignalOutboundJournal().load()

    async def test_reader_skips_extreme_json_then_delivers_next_valid_response(self):
        bot = signal.SignalBot([], jsonrpc_socket="tcp://127.0.0.1:7583")
        reader = asyncio.StreamReader()
        reader.feed_data(("[" * 2000 + "0" + "]" * 2000 + "\n").encode())
        reader.feed_data(("{\"id\": " + "9" * 5000 + "}\n").encode())
        reader.feed_data(b'{"id": "next", "result": "ok"}\n')
        reader.feed_eof()
        response = asyncio.get_running_loop().create_future()
        bot._jsonrpc_pending["next"] = response
        # EOF cleanup is unrelated to decoding and must not launch reconnection.
        with (mock.patch.object(bot, "_schedule_jsonrpc_reconnect"),
              self.assertLogs(signal.logger, "WARNING")):
            await bot._jsonrpc_reader_loop(reader, mock.Mock())
        self.assertEqual(response.result(), {"id": "next", "result": "ok"})


class BackgroundLaunchCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.object(workspace, "CONFIG_DIR", self.directory))
        self.bot = TestBot(["user"])
        self.bot._require_ws = mock.AsyncMock(return_value=self.directory)
        self.bot.send_text = mock.AsyncMock()
        self.bot.start_detached_task_watcher = mock.Mock()
        self.enterContext(mock.patch.object(workspace, "get_run_config", return_value=(
            "claude_code", "sonnet", "haiku", "auto", "claude_code",
        )))
        self.backend = mock.Mock()
        self.backend.get_detached_task_status = mock.AsyncMock(return_value=DetachedTaskStatus("working"))
        self.backend.stop_detached_task = mock.AsyncMock(return_value=True)
        self.enterContext(mock.patch.object(bot_base.backends_agent, "get_backend", return_value=self.backend))
        self.launch = agent.DetachedTaskLaunch("claude_code", "fixture-task", "fixture-session")
        self.ctx = self.bot.make_context("user", "chat", command="bg", args="run checks")

    async def test_stop_cancels_a_launch_without_overwriting_foreground_tracking(self):
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def launch(*_args, **_kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        foreground = asyncio.create_task(asyncio.Event().wait())
        self.bot._running_tasks["user"] = foreground
        with mock.patch.object(agent, "launch_detached", side_effect=launch):
            command = asyncio.create_task(self.bot.cmd_background(self.ctx))
            await asyncio.wait_for(entered.wait(), 1)
            self.assertIs(self.bot._running_tasks["user"], foreground)
            self.assertTrue(self.bot.has_active_turns())
            running, removed = await asyncio.wait_for(self.bot._cancel_user_work("user"), 1)
            await asyncio.wait_for(command, 1)
        await asyncio.gather(foreground, return_exceptions=True)
        self.bot._running_tasks.clear()
        self.assertTrue(running)
        self.assertTrue(removed)
        self.assertTrue(cancelled.is_set())
        self.assertFalse(self.bot._detached_launch_tasks)
        self.bot.send_text.assert_not_awaited()
        self.assertEqual(await self.bot._list_detached_task_records(), [])

    async def test_stop_during_validation_stops_known_provider_without_registration(self):
        entered = asyncio.Event()

        async def validate(*_args):
            entered.set()
            await asyncio.Event().wait()

        self.backend.get_detached_task_status.side_effect = validate
        with mock.patch.object(agent, "launch_detached", return_value=self.launch):
            command = asyncio.create_task(self.bot.cmd_background(self.ctx))
            await asyncio.wait_for(entered.wait(), 1)
            self.assertTrue(self.bot.has_active_turns())
            self.assertEqual(await asyncio.wait_for(self.bot._cancel_user_work("user"), 1), (False, True))
            await asyncio.wait_for(command, 1)
        self.backend.stop_detached_task.assert_awaited_once_with(self.directory, "fixture-task")
        self.bot.send_text.assert_not_awaited()
        self.bot.start_detached_task_watcher.assert_not_called()
        self.assertEqual(await self.bot._list_detached_task_records(), [])

    async def test_stale_validation_cannot_recreate_ledger_after_stop_clears_it(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def validate(*_args):
            entered.set()
            await release.wait()
            return DetachedTaskStatus("working")

        self.backend.get_detached_task_status.side_effect = validate
        registration = asyncio.create_task(self.bot._register_detached_task(
            uid="user", chat_id="chat", workspace_path=self.directory,
            session_id=None, backend_name="claude_code", task_id="fixture-task", validate=True,
        ))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await self.bot._cancel_user_work("user")
            release.set()
            self.assertFalse(await asyncio.wait_for(registration, 1))
        finally:
            release.set()
            await asyncio.gather(registration, return_exceptions=True)
        self.assertEqual(await self.bot._list_detached_task_records(), [])
        self.bot.start_detached_task_watcher.assert_not_called()

    async def test_new_background_command_waits_for_old_cancellation_to_finish(self):
        validating, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def validate(*_args):
            validating.set()
            await asyncio.Event().wait()

        async def stop(*_args):
            stopping.set()
            await release.wait()
            return True

        self.backend.get_detached_task_status.side_effect = validate
        self.backend.stop_detached_task.side_effect = stop
        with mock.patch.object(agent, "launch_detached", return_value=self.launch) as launch:
            original = asyncio.create_task(self.bot.cmd_background(self.ctx))
            cancellation = None
            try:
                await asyncio.wait_for(validating.wait(), 1)
                cancellation = asyncio.create_task(self.bot._cancel_user_work("user"))
                await asyncio.wait_for(stopping.wait(), 1)
                await asyncio.wait_for(self.bot.cmd_background(self.ctx), 1)
                launch.assert_awaited_once()
                self.assertIn("Cancellation is still finishing", self.bot.send_text.await_args.args[1])
                release.set()
                self.assertEqual(await asyncio.wait_for(cancellation, 1), (False, True))
                await asyncio.wait_for(original, 1)
            finally:
                release.set()
                if cancellation is not None:
                    await asyncio.gather(cancellation, return_exceptions=True)
                await asyncio.gather(original, return_exceptions=True)
        self.assertEqual(await self.bot._list_detached_task_records(), [])

    async def test_stop_during_untracked_launch_ack_stops_known_provider(self):
        acknowledging = asyncio.Event()

        async def send(*_args, **_kwargs):
            acknowledging.set()
            await asyncio.Event().wait()

        self.bot.send_text.side_effect = send
        with (mock.patch.object(agent, "launch_detached", return_value=self.launch),
              mock.patch.object(self.bot, "_register_detached_task", side_effect=OSError("disk full")),
              self.assertLogs(bot_base.logger, "ERROR")):
            command = asyncio.create_task(self.bot.cmd_background(self.ctx))
            try:
                await asyncio.wait_for(acknowledging.wait(), 1)
                await asyncio.wait_for(self.bot._cancel_user_work("user"), 1)
                await asyncio.wait_for(command, 1)
            finally:
                command.cancel()
                await asyncio.gather(command, return_exceptions=True)
        self.backend.stop_detached_task.assert_awaited_once_with(self.directory, "fixture-task")
        self.assertEqual(await self.bot._list_detached_task_records(), [])


if __name__ == "__main__":
    unittest.main()
