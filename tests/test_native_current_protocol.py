"""Native adapters require current payloads and workspace-owned job state."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from Cozter.backends_agent import claude_code
from Cozter.backends_agent.base import AgentResult
from Cozter.backends_agent.copilot import CopilotBackend


class CurrentCopilotPayloadTests(unittest.TestCase):
    def test_old_outer_message_wrappers_cannot_supply_agent_text(self):
        backend = CopilotBackend()
        for event in (
            {"type": "assistant_message", "role": "assistant", "text": "retired wrapper"},
            {"type": "assistant.message", "content": "retired wrapper"},
            {"type": "assistant.message", "text": "retired wrapper", "data": None},
            {"type": "assistant.message", "data": {"text": "retired wrapper"}},
            {"type": "assistant.message", "data": {"message": "retired wrapper"}},
            {"type": "assistant.message", "data": {"content": {"text": "retired wrapper"}}},
        ):
            with self.subTest(event=event):
                result = AgentResult()
                backend.parse_event(event, result)
                self.assertEqual(result.text, "")
                self.assertIsNone(backend.extract_agent_text(event))

    def test_current_nested_payload_wins_over_any_outer_text(self):
        backend = CopilotBackend()
        event = {"type": "assistant.message", "content": "retired wrapper",
                 "data": {"text": "retired wrapper", "content": "current visible reply"}}
        result = AgentResult()
        backend.parse_event(event, result)
        self.assertEqual(result.text, "current visible reply")
        self.assertEqual(backend.extract_agent_text(event), "current visible reply")

    def test_retired_assistant_type_and_event_key_aliases_are_ignored(self):
        backend = CopilotBackend()
        events = [
            {"type": alias, "data": {"content": "retired type"}}
            for alias in ("assistant_message", "agent_message", "message", "completion",
                          "response", "text", "", "item.completed")
        ]
        events.extend((
            {"data": {"content": "missing type"}},
            {"event": "assistant.message", "data": {"content": "retired key"}},
        ))
        for event in events:
            with self.subTest(event=event):
                result = AgentResult()
                backend.parse_event(event, result)
                self.assertEqual(result.text, "")
                self.assertIsNone(backend.extract_agent_text(event))

    def test_retired_tool_file_and_error_aliases_cannot_report_activity(self):
        backend = CopilotBackend()
        for alias in ("tool_use", "tool_call", "tool_start", "tool", "tool_result",
                      "tool_output", "tool_end", "file_change", "edit", "file",
                      "turn.failed", "failed"):
            with self.subTest(alias=alias):
                result = AgentResult()
                backend.parse_event({
                    "type": alias, "name": "shell", "message": "retired error",
                    "path": "retired.py", "input": {"command": "retired command"},
                    "data": {"content": "retired nested reply"},
                }, result)
                self.assertEqual(result.events, [])
                self.assertIsNone(result.error)

    def test_current_tool_result_uses_nested_content_and_keeps_failures_recoverable(self):
        backend = CopilotBackend()
        result = AgentResult()
        for payload in (
            {"toolCallId": "first", "success": True, "result": {"content": "current output"}},
            {"toolCallId": "second", "success": False, "error": {"message": "fixable tool failure"}},
            {"toolCallId": "third", "success": True, "result": ["malformed output"]},
        ):
            backend.parse_event({
                "type": "tool.execution_complete", "output": "retired output", "data": payload,
            }, result)
        self.assertEqual([event.content for event in result.events], [
            "tool done\ncurrent output", "tool failed\nfixable tool failure", "tool done",
        ])
        self.assertIsNone(result.error)
        backend.parse_event({"type": "session.error", "message": "retired outer error",
                             "data": {"message": "current session failure"}}, result)
        self.assertEqual(result.error, "current session failure")


class CurrentClaudeJobOutputTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_or_foreign_job_state_never_invokes_legacy_logs(self):
        backend = claude_code.ClaudeCodeBackend()
        with tempfile.TemporaryDirectory() as root:
            home, workspace, foreign = (Path(root, name) for name in ("home", "workspace", "foreign"))
            workspace.mkdir()
            foreign.mkdir()
            state = home / "jobs" / "fixture-task" / "state.json"
            with (
                mock.patch.object(claude_code, "_claude_home", return_value=str(home)),
                mock.patch.object(claude_code, "_run_claude_command", new=mock.AsyncMock()) as run,
            ):
                for contents in (None, "invalid JSON", json.dumps({"cwd": str(foreign), "state": "done"})):
                    if contents is not None:
                        state.parent.mkdir(parents=True, exist_ok=True)
                        state.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "no trusted local job state"):
                        await backend.get_detached_task_output(str(workspace), "fixture-task")
                run.assert_not_awaited()

    async def test_current_trusted_job_can_recover_missing_local_output(self):
        backend = claude_code.ClaudeCodeBackend()
        with tempfile.TemporaryDirectory() as root:
            home, workspace = Path(root, "home"), Path(root, "workspace")
            workspace.mkdir()
            state = home / "jobs" / "fixture-task" / "state.json"
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps({"cwd": str(workspace), "state": "done"}), encoding="utf-8")
            with (
                mock.patch.object(claude_code, "_claude_home", return_value=str(home)),
                mock.patch.object(claude_code, "executable_command", return_value=["fixture-claude"]),
                mock.patch.object(claude_code, "_run_claude_command", new=mock.AsyncMock(
                    return_value=(0, "recovered current output", ""),
                )) as run,
            ):
                output = await backend.get_detached_task_output(str(workspace), "fixture-task")
        self.assertEqual(output, "recovered current output")
        run.assert_awaited_once_with(["fixture-claude", "logs", "fixture-task"], cwd=str(workspace))


if __name__ == "__main__":
    unittest.main()
