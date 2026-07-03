"""Fixture-driven regression tests for each backend's ``parse_event``.

The backends translate a CLI's loosely-typed JSONL events into ChatEvents
with best-effort key probing; a renamed field or reshaped event would
otherwise fail *silently* (a debug log, then a dropped reply). These
fixtures mirror the real streamed shapes documented in each backend
module and lock the event -> ChatEvent contract, so schema drift trips a
test instead.
"""

import unittest

from Cozter.backends_agent.base import AgentResult
from Cozter.backends_agent.claude_code import ClaudeCodeBackend
from Cozter.backends_agent.codex import CodexBackend
from Cozter.backends_agent.copilot import CopilotBackend
from Cozter.backends_agent.llama import LlamaBackend


def _run(backend, events: list[dict]) -> AgentResult:
    result = AgentResult()
    for event in events:
        backend.parse_event(event, result)
    return result


def _kinds(result: AgentResult) -> list[str]:
    return [e.kind for e in result.events]


class CodexParseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = CodexBackend()

    def test_agent_message_becomes_text(self) -> None:
        r = _run(self.backend, [
            {"type": "item.completed",
             "item": {"type": "agent_message", "text": "hello world"}},
        ])
        self.assertEqual(r.text, "hello world")
        self.assertIn("text", _kinds(r))

    def test_command_execution_becomes_tool(self) -> None:
        r = _run(self.backend, [
            {"type": "item.completed", "item": {
                "type": "command_execution",
                "command": "ls -la", "exit_code": 0,
                "aggregated_output": "file.txt",
            }},
        ])
        self.assertIn("tool", _kinds(r))
        self.assertIn("ls -la", r.events[0].content)

    def test_file_change_becomes_file(self) -> None:
        r = _run(self.backend, [
            {"type": "item.completed", "item": {
                "type": "file_change",
                "changes": [{"path": "a.py", "kind": "modified"}],
            }},
        ])
        self.assertIn("file", _kinds(r))
        self.assertIn("a.py", r.events[0].content)

    def test_turn_failed_sets_error(self) -> None:
        r = _run(self.backend, [
            {"type": "turn.failed", "error": {"message": "boom"}},
        ])
        self.assertEqual(r.error, "boom")
        self.assertTrue(
            any(e.kind == "text" and "boom" in e.content for e in r.events)
        )

    def test_turn_completed_captures_usage(self) -> None:
        r = _run(self.backend, [
            {"type": "turn.completed", "usage": {
                "input_tokens": 12470, "cached_input_tokens": 9600,
                "output_tokens": 28,
            }},
        ])
        assert r.usage is not None
        self.assertEqual(r.usage["input_tokens"], 12470)
        self.assertEqual(r.usage["output_tokens"], 28)


class ClaudeCodeParseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = ClaudeCodeBackend()

    def test_assistant_text_block(self) -> None:
        r = _run(self.backend, [
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "hi there"},
            ]}},
        ])
        self.assertEqual(r.text, "hi there")

    def test_assistant_tool_use_bash(self) -> None:
        r = _run(self.backend, [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Bash",
                 "input": {"command": "pytest"}},
            ]}},
        ])
        self.assertIn("tool", _kinds(r))
        self.assertIn("pytest", r.events[0].content)

    def test_assistant_tool_use_file(self) -> None:
        r = _run(self.backend, [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Write",
                 "input": {"file_path": "/ws/x.py"}},
            ]}},
        ])
        self.assertIn("file", _kinds(r))
        self.assertIn("x.py", r.events[0].content)

    def test_result_terminal_text_fallback(self) -> None:
        r = _run(self.backend, [
            {"type": "result", "subtype": "success", "result": "done"},
        ])
        self.assertEqual(r.text, "done")

    def test_result_error_sets_error(self) -> None:
        r = _run(self.backend, [
            {"type": "result", "is_error": True, "error": "nope"},
        ])
        self.assertEqual(r.error, "nope")

    def test_result_captures_usage_and_cost(self) -> None:
        r = _run(self.backend, [
            {"type": "result", "subtype": "success", "result": "done",
             "usage": {"input_tokens": 100, "output_tokens": 50},
             "total_cost_usd": 0.0123},
        ])
        assert r.usage is not None
        self.assertEqual(r.usage["input_tokens"], 100)
        self.assertEqual(r.usage["total_cost_usd"], 0.0123)


class CopilotParseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = CopilotBackend()

    def test_tool_use_event(self) -> None:
        r = _run(self.backend, [
            {"type": "tool_use", "name": "bash",
             "input": {"command": "echo hi"}},
        ])
        self.assertIn("tool", _kinds(r))

    def test_file_change_event(self) -> None:
        r = _run(self.backend, [
            {"type": "file_change", "path": "b.py", "action": "modified"},
        ])
        self.assertIn("file", _kinds(r))
        self.assertIn("b.py", r.events[0].content)

    def test_assistant_text(self) -> None:
        r = _run(self.backend, [
            {"type": "assistant_message", "role": "assistant",
             "text": "answer"},
        ])
        self.assertEqual(r.text, "answer")

    def test_error_sets_error(self) -> None:
        r = _run(self.backend, [
            {"type": "error", "message": "bad"},
        ])
        self.assertEqual(r.error, "bad")


class LlamaParseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = LlamaBackend()

    def test_assistant_text(self) -> None:
        r = _run(self.backend, [
            {"type": "assistant_text", "text": "yo"},
        ])
        self.assertEqual(r.text, "yo")

    def test_tool_use_with_file_action(self) -> None:
        r = _run(self.backend, [
            {"type": "tool_use", "name": "write_file",
             "input": {"path": "c.py"}, "file_action": "write"},
        ])
        self.assertIn("tool", _kinds(r))
        self.assertIn("file", _kinds(r))
        self.assertTrue(any("c.py" in e.content for e in r.events))

    def test_tool_result_is_suppressed(self) -> None:
        r = _run(self.backend, [
            {"type": "tool_result", "name": "read_file", "output": "x"},
        ])
        self.assertEqual(r.events, [])

    def test_error_sets_error(self) -> None:
        r = _run(self.backend, [
            {"type": "error", "message": "explode"},
        ])
        self.assertEqual(r.error, "explode")


if __name__ == "__main__":
    unittest.main()
