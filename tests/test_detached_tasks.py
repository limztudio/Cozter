"""Regression coverage for restart-safe detached task callbacks."""

import json
import os
import tempfile
import unittest
from unittest import mock

from Cozter import agent, backends_agent, workspace
from Cozter.backends_agent.base import (
    AgentResult,
    DetachedTaskRef,
    DetachedTaskRequest,
    DetachedTaskStatus,
)
from Cozter.backends_agent.claude_code import ClaudeCodeBackend
from Cozter.backends_agent import claude_code as claude_code_mod
from Cozter.backends_bot.base import BotContext, BotPlatform


class _DetachedBackend:
    name = "claude_code"
    supports_detached_tasks = True

    def __init__(self) -> None:
        self.status = DetachedTaskStatus("done")
        self.output = "All checks passed."
        self.stopped: list[str] = []

    async def get_detached_task_status(self, _ws: str, _task_id: str):
        return self.status

    async def get_detached_task_output(self, _ws: str, _task_id: str):
        return self.output

    async def stop_detached_task(self, _ws: str, task_id: str):
        self.stopped.append(task_id)
        return True


class _DetachedBot(BotPlatform):
    def __init__(self) -> None:
        super().__init__(["u1"])
        self.sent: list[str] = []
        self.fail_sends = False

    @property
    def platform_id(self) -> str:
        return "test:detached"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def start_detached_task_watcher(self) -> None:
        # Unit tests drive each polling pass explicitly.
        return

    async def send_text(self, _chat_id: str, text: str, *, rich: bool = False):
        if self.fail_sends:
            raise RuntimeError("platform unavailable")
        self.sent.append(text)
        return None

    async def edit_text(self, _handle, _text: str, *, rich: bool = False):
        pass

    async def delete_message(self, _handle) -> None:
        pass

    async def send_file(self, _chat_id: str, _path: str) -> None:
        pass


class ClaudeDetachedTaskTests(unittest.IsolatedAsyncioTestCase):
    def _assert_background_guard_settings(self, cmd: list[str]) -> None:
        self.assertIn("--settings", cmd)
        settings = json.loads(cmd[cmd.index("--settings") + 1])
        self.assertFalse(settings["disableAllHooks"])
        hooks = settings["hooks"]["PreToolUse"]
        self.assertEqual(hooks[0]["matcher"], "Bash")
        hook = hooks[0]["hooks"][0]
        self.assertEqual(hook["type"], "command")
        self.assertTrue(hook["command"])
        self.assertTrue(hook["args"][0].endswith("claude_background_guard.py"))

    async def test_foreground_launch_installs_background_guard(self) -> None:
        backend = ClaudeCodeBackend()
        proc = mock.Mock()
        with mock.patch.object(
            claude_code_mod,
            "create_prompt_subprocess",
            new=mock.AsyncMock(return_value=proc),
        ) as create_process:
            launched = await backend.launch(
                "/work", "run checks", "sonnet", "auto",
            )

        self.assertIs(launched, proc)
        cmd = create_process.await_args.args[0]
        self._assert_background_guard_settings(cmd)

    async def test_launch_uses_background_mode_not_print_mode(self) -> None:
        backend = ClaudeCodeBackend()
        seen: list[str] = []

        async def fake_run(cmd: list[str], *, cwd: str):
            seen.extend(cmd)
            self.assertEqual(cwd, "/work")
            return 0, "Starting background service…\nbackgrounded · 048e1065", ""

        with mock.patch.object(claude_code_mod, "_run_claude_command", fake_run):
            task_id = await backend.launch_detached(
                "/work", "repair the tests", "sonnet", "auto", effort=50,
            )

        self.assertEqual(task_id, "048e1065")
        self.assertIn("--bg", seen)
        self.assertIn("repair the tests", seen)
        self.assertNotIn("--print", seen)
        self.assertNotIn("--output-format", seen)
        self.assertNotIn("--no-session-persistence", seen)
        self._assert_background_guard_settings(seen)

    async def test_status_accepts_only_matching_background_session(self) -> None:
        backend = ClaudeCodeBackend()
        payload = [
            {"id": "048e1065", "kind": "interactive", "cwd": "/work"},
            {
                "id": "048e1065", "kind": "background", "cwd": "/work",
                "state": "blocked", "waitingFor": "input needed",
            },
        ]

        async def fake_run(_cmd: list[str], *, cwd: str):
            self.assertEqual(cwd, "/work")
            return 0, json.dumps(payload), ""

        with mock.patch.object(claude_code_mod, "_run_claude_command", fake_run):
            status = await backend.get_detached_task_status("/work", "048e1065")

        self.assertIsNotNone(status)
        assert status is not None
        self.assertEqual(status.state, "blocked")
        self.assertEqual(status.waiting_for, "input needed")

    async def test_status_accepts_a_workspace_subdirectory(self) -> None:
        backend = ClaudeCodeBackend()
        payload = [{
            "id": "048e1065", "kind": "background",
            "cwd": "/work/packages/api", "state": "working",
        }]

        async def fake_run(_cmd: list[str], *, cwd: str):
            self.assertEqual(cwd, "/work")
            return 0, json.dumps(payload), ""

        with mock.patch.object(claude_code_mod, "_run_claude_command", fake_run):
            status = await backend.get_detached_task_status("/work", "048e1065")

        self.assertIsNotNone(status)
        assert status is not None
        self.assertEqual(status.state, "working")

    def test_background_id_parser_requires_dedicated_launch_line(self) -> None:
        self.assertEqual(
            claude_code_mod._background_task_ids(
                "\x1b[32mbackgrounded · 048e1065\x1b[0m\r\n"
                "  claude logs 048e1065",
            ),
            ["048e1065"],
        )
        self.assertEqual(
            claude_code_mod._background_task_ids("claude logs 048e1065"),
            [],
        )

    async def test_status_falls_back_to_durable_job_state(self) -> None:
        backend = ClaudeCodeBackend()
        task_id = "048e1065"
        with tempfile.TemporaryDirectory() as claude_home:
            state_path = os.path.join(
                claude_home, "jobs", task_id, "state.json",
            )
            os.makedirs(os.path.dirname(state_path))
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({"cwd": "/work", "state": "done"}, f)

            async def fake_run(_cmd: list[str], *, cwd: str):
                self.assertEqual(cwd, "/work")
                return 0, "[]", ""

            with (
                mock.patch.object(
                    claude_code_mod, "_claude_home", return_value=claude_home,
                ),
                mock.patch.object(
                    claude_code_mod, "_run_claude_command", fake_run,
                ),
            ):
                status = await backend.get_detached_task_status("/work", task_id)

        self.assertIsNotNone(status)
        assert status is not None
        self.assertEqual(status.state, "done")

    async def test_output_uses_visible_text_from_durable_transcript(self) -> None:
        backend = ClaudeCodeBackend()
        task_id = "048e1065"
        session_id = "048e1065-aaaa-bbbb-cccc-0123456789ab"
        with tempfile.TemporaryDirectory() as claude_home:
            state_path = os.path.join(
                claude_home, "jobs", task_id, "state.json",
            )
            transcript_path = os.path.join(
                claude_home, "projects", "-work", f"{session_id}.jsonl",
            )
            os.makedirs(os.path.dirname(state_path))
            os.makedirs(os.path.dirname(transcript_path))
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({
                    "cwd": "/work",
                    "state": "done",
                    "sessionId": session_id,
                    "linkScanPath": transcript_path,
                    "output": {"result": "short state summary"},
                }, f)
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "First result."}],
                }}) + "\n")
                f.write(json.dumps({"message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hidden"},
                        {"type": "text", "text": "Final result."},
                    ],
                }}) + "\n")

            with mock.patch.object(
                claude_code_mod, "_claude_home", return_value=claude_home,
            ):
                output = await backend.get_detached_task_output("/work", task_id)

        self.assertEqual(output, "First result.\n\nFinal result.")


class DetachedTaskLedgerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addAsyncCleanup(self._cleanup)
        self.old_config_dir = workspace.CONFIG_DIR
        workspace.CONFIG_DIR = os.path.join(self.temp.name, "config")
        self.bot = _DetachedBot()
        self.backend = _DetachedBackend()
        self.backend_patch = mock.patch.object(
            backends_agent, "get_backend", return_value=self.backend,
        )
        self.backend_patch.start()
        self.addCleanup(self.backend_patch.stop)

    async def _cleanup(self) -> None:
        workspace.CONFIG_DIR = self.old_config_dir
        self.temp.cleanup()

    async def _register(self) -> None:
        tracked = await self.bot._register_detached_task(
            uid="u1",
            chat_id="c1",
            workspace_path="/work",
            session_id=None,
            backend_name="claude_code",
            task_id="048e1065",
            validate=True,
        )
        self.assertTrue(tracked)

    async def test_completion_is_delivered_without_rerunning_an_agent_turn(self) -> None:
        await self._register()

        await self.bot._check_detached_tasks()

        self.assertEqual(
            self.bot.sent,
            ["Background task 048e1065 completed.\n\nAll checks passed."],
        )
        self.assertEqual(await self.bot._list_detached_task_records(), [])

    async def test_completed_payload_survives_failed_delivery_and_retries(self) -> None:
        await self._register()
        self.bot.fail_sends = True

        with self.assertLogs("Cozter.backends_bot.base", level="WARNING"):
            await self.bot._check_detached_tasks()

        records = await self.bot._list_detached_task_records()
        self.assertEqual(len(records), 1)
        self.assertIn("delivery_text", records[0])
        self.bot.fail_sends = False

        await self.bot._check_detached_tasks()

        self.assertEqual(len(self.bot.sent), 1)
        self.assertEqual(await self.bot._list_detached_task_records(), [])

    async def test_discovered_task_is_persisted_only_after_provider_validation(self) -> None:
        result = AgentResult(detached_tasks=[DetachedTaskRef(
            backend_name="claude_code", task_id="048e1065",
        )])

        await self.bot._register_result_detached_tasks(
            "u1", "c1", "/work", result, session_id=None,
        )

        self.assertEqual(len(await self.bot._list_detached_task_records()), 1)

    async def test_agent_request_launches_and_persists_before_acknowledging(self) -> None:
        result = AgentResult(detached_task_requests=[DetachedTaskRequest(
            prompt="Run the full test suite and report the result.",
        )])
        launch = agent.DetachedTaskLaunch(
            backend_name="claude_code", task_id="048e1065",
            session_id="session-1",
        )

        with mock.patch.object(
            agent, "launch_detached", new=mock.AsyncMock(return_value=launch),
        ) as start_task:
            await self.bot._launch_result_detached_tasks(
                "u1",
                "c1",
                "/work",
                result,
                model="sonnet",
                summary_model="haiku",
                approval="auto",
                backend_name="claude_code",
                summary_backend_name="claude_code",
            )

        records = await self.bot._list_detached_task_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["session_id"], "session-1")
        start_task.assert_awaited_once()
        self.assertEqual(
            result.events[-1].content,
            "Started background task 048e1065. I’ll post its final result here.",
        )

    async def test_agent_request_does_not_launch_after_foreground_failure(self) -> None:
        result = AgentResult(
            error="stream disconnected",
            detached_task_requests=[DetachedTaskRequest(prompt="Run checks.")],
        )

        with mock.patch.object(agent, "launch_detached") as start_task:
            await self.bot._launch_result_detached_tasks(
                "u1",
                "c1",
                "/work",
                result,
                model="sonnet",
                summary_model="haiku",
                approval="auto",
                backend_name="claude_code",
                summary_backend_name="claude_code",
            )

        start_task.assert_not_called()
        self.assertIn("foreground agent run failed", result.events[-1].content)

    async def test_background_command_registers_before_acknowledging(self) -> None:
        ctx = BotContext(
            user_id="u1", chat_id="c1", text="/bg run checks",
            command="bg", args="run checks", attachment=None,
            platform=self.bot,
        )
        launch = agent.DetachedTaskLaunch(
            backend_name="claude_code", task_id="048e1065",
            session_id="session-1",
        )
        with (
            mock.patch.object(
                self.bot, "_require_ws",
                new=mock.AsyncMock(return_value="/work"),
            ),
            mock.patch.object(
                workspace, "get_run_config",
                return_value=("claude_code", "sonnet", "haiku", "auto", "claude_code"),
            ),
            mock.patch.object(
                agent, "launch_detached", new=mock.AsyncMock(return_value=launch),
            ) as start_task,
        ):
            await self.bot.cmd_background(ctx)

        records = await self.bot._list_detached_task_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["session_id"], "session-1")
        self.assertEqual(
            self.bot.sent,
            ["Started background task 048e1065. I’ll post its final result here."],
        )
        start_task.assert_awaited_once()

    async def test_background_command_reports_untracked_task_after_ledger_error(self) -> None:
        ctx = BotContext(
            user_id="u1", chat_id="c1", text="/bg run checks",
            command="bg", args="run checks", attachment=None,
            platform=self.bot,
        )
        launch = agent.DetachedTaskLaunch(
            backend_name="claude_code", task_id="048e1065",
            session_id="session-1",
        )
        with (
            mock.patch.object(
                self.bot, "_require_ws",
                new=mock.AsyncMock(return_value="/work"),
            ),
            mock.patch.object(
                workspace, "get_run_config",
                return_value=("claude_code", "sonnet", "haiku", "auto", "claude_code"),
            ),
            mock.patch.object(
                agent, "launch_detached", new=mock.AsyncMock(return_value=launch),
            ),
            mock.patch.object(
                self.bot, "_register_detached_task",
                new=mock.AsyncMock(side_effect=OSError("disk full")),
            ),
            self.assertLogs("Cozter.backends_bot.base", level="ERROR"),
        ):
            await self.bot.cmd_background(ctx)

        self.assertEqual(len(self.bot.sent), 1)
        self.assertIn("could not verify", self.bot.sent[0])

    async def test_cancel_stops_provider_task_and_removes_ledger_record(self) -> None:
        await self._register()

        cancelled = await self.bot._cancel_detached_tasks("u1")

        self.assertEqual(cancelled, 1)
        self.assertEqual(self.backend.stopped, ["048e1065"])
        self.assertEqual(await self.bot._list_detached_task_records(), [])


if __name__ == "__main__":
    unittest.main()
