"""Fixture-driven regression tests for each backend's ``parse_event``.

The backends translate a CLI's loosely-typed JSONL events into ChatEvents
with best-effort key probing; a renamed field or reshaped event would
otherwise fail *silently* (a debug log, then a dropped reply). These
fixtures mirror the real streamed shapes documented in each backend
module and lock the event -> ChatEvent contract, so schema drift trips a
test instead.
"""

import unittest

from Cozter.backends_agent.base import AgentResult, set_error_result
from Cozter.backends_agent.claude_code import ClaudeCodeBackend
from Cozter.backends_agent.codex import CodexBackend
from Cozter.backends_agent.copilot import CopilotBackend
from Cozter.backends_agent.grok import GrokBackend
from Cozter.backends_agent.llama import LlamaBackend


def _run(backend, events: list[dict]) -> AgentResult:
    result = AgentResult()
    for event in events:
        backend.parse_event(event, result)
    return result


def _kinds(result: AgentResult) -> list[str]:
    return [event.kind for event in result.events]


class ErrorResultTests(unittest.TestCase):
    def test_non_text_or_blank_error_is_normalized(self) -> None:
        for message in (None, "", "   ", ["bad"], {"message": "bad"}):
            with self.subTest(message=message):
                result = AgentResult()
                set_error_result(result, message)
                self.assertEqual(result.error, "Unknown error")
                self.assertEqual(result.text, "Error: Unknown error")


class CodexParseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = CodexBackend()

    def test_agent_message_non_string_text_is_skipped(self) -> None:
        result = _run(self.backend, [{
            "type": "item.completed",
            "item": {"type": "agent_message", "text": {"bad": "shape"}},
        }])
        self.assertEqual(result.text, "")
        self.assertEqual(result.events, [])

    def test_file_change_malformed_changes_are_skipped(self) -> None:
        """Schema drift in ``changes`` must not abort the whole turn."""
        for changes in (None, "unexpected", {"path": "a.py"}):
            with self.subTest(changes=changes):
                result = _run(self.backend, [{
                    "type": "item.completed",
                    "item": {"type": "file_change", "changes": changes},
                }])
                self.assertEqual(result.events, [])

        result = _run(self.backend, [{
            "type": "item.completed",
            "item": {
                "type": "file_change",
                "changes": [None, "unexpected", {
                    "path": "a.py", "kind": "modified",
                }],
            },
        }])
        self.assertEqual(len(result.events), 1)
        self.assertIn("a.py", result.events[0].content)

    def test_late_turn_failed_never_overwrites_the_answer(self) -> None:
        result = _run(self.backend, [
            {"type": "item.completed",
             "item": {"type": "agent_message", "text": "hello world"}},
            {"type": "turn.failed", "error": {"message": "boom"}},
        ])
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.error, "boom")

    def test_extract_agent_text_ignores_non_string_payloads(self) -> None:
        self.assertIsNone(self.backend.extract_agent_text({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": {"detail": "nope"}},
        }))
        self.assertEqual(
            self.backend.extract_agent_text({
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "hello"},
            }),
            "hello",
        )

    def test_a_recovered_stream_error_still_yields_the_answer(self) -> None:
        """A retried error must not bury the reply the model went on to give."""
        result = _run(self.backend, [
            {"type": "error", "message": "transient hiccup"},
            {"type": "item.completed",
             "item": {"type": "agent_message", "text": "hello world"}},
        ])
        self.assertEqual(result.text, "hello world")

    def test_a_late_stream_error_never_overwrites_the_answer(self) -> None:
        """An error trailing the reply is recorded, not shown in its place."""
        result = _run(self.backend, [
            {"type": "item.completed",
             "item": {"type": "agent_message", "text": "hello world"}},
            {"type": "error", "message": "disconnected while closing"},
        ])
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.error, "disconnected while closing")

    def test_late_malformed_stream_error_is_normalized(self) -> None:
        result = _run(self.backend, [
            {"type": "item.completed",
             "item": {"type": "agent_message", "text": "hello world"}},
            {"type": "error", "message": {"detail": "disconnected"}},
        ])
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.error, "Unknown error")

    def test_malformed_item_null_does_not_crash(self) -> None:
        """A present-but-null ``item`` must not raise AttributeError."""
        result = _run(self.backend, [
            {"type": "item.completed", "item": None},
        ])
        self.assertEqual(result.text, "")
        self.assertEqual(result.events, [])

    def test_malformed_item_non_dict_does_not_crash(self) -> None:
        """A non-object ``item`` must be tolerated, not crash the turn."""
        result = _run(self.backend, [
            {"type": "item.completed", "item": "unexpected"},
        ])
        self.assertEqual(result.text, "")
        self.assertEqual(result.events, [])


class ClaudeCodeParseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = ClaudeCodeBackend()

    def test_assistant_text_block_non_string_text_is_skipped(self) -> None:
        result = _run(self.backend, [{
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": []}]},
        }])
        self.assertEqual(result.text, "")
        self.assertEqual(result.events, [])

    def test_assistant_non_dict_block_is_skipped(self) -> None:
        """A bare-string content entry must not raise during parsing."""
        result = _run(self.backend, [
            {"type": "assistant", "message": {"content": [
                "raw string block",
                {"type": "text", "text": "ok"},
            ]}},
        ])
        self.assertEqual(result.text, "ok")

    def test_assistant_malformed_message_shapes_are_skipped(self) -> None:
        """Unexpected assistant envelopes must not abort the stream."""
        malformed_messages: tuple[object, ...] = (
            None,
            "unexpected",
            [],
            {"content": None},
            {"content": {"type": "text"}},
        )
        for message in malformed_messages:
            with self.subTest(message=message):
                result = _run(self.backend, [{
                    "type": "assistant", "message": message,
                }])
                self.assertEqual(result.events, [])
                self.assertIsNone(self.backend.extract_agent_text({
                    "type": "assistant", "message": message,
                }))

    def test_paired_background_bash_result_records_task(self) -> None:
        result = _run(self.backend, [
            {"type": "assistant", "message": {"content": [{
                "type": "tool_use", "id": "tool-bg", "name": "Bash",
                "input": {"command": 'claude --bg "run checks"'},
            }]}},
            {"type": "user", "message": {"content": [{
                "type": "tool_result", "tool_use_id": "tool-bg",
                "content": "backgrounded · 048e1065\n"
                "  claude logs 048e1065",
            }]}},
        ])
        self.assertEqual(
            [(task.backend_name, task.task_id) for task in result.detached_tasks],
            [("claude_code", "048e1065")],
        )

    def test_unpaired_background_marker_is_not_trusted(self) -> None:
        result = _run(self.backend, [
            {"type": "assistant", "message": {"content": [{
                "type": "tool_use", "id": "ordinary", "name": "Bash",
                "input": {"command": "echo backgrounded"},
            }]}},
            {"type": "user", "message": {"content": [{
                "type": "tool_result", "tool_use_id": "ordinary",
                "content": "backgrounded · 048e1065",
            }]}},
        ])
        self.assertEqual(result.detached_tasks, [])

    def test_assistant_file_tool_non_string_path_does_not_crash(self) -> None:
        result = _run(self.backend, [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Write",
                 "input": {"file_path": ["/ws/x.py"]}},
            ]}},
        ])
        self.assertEqual(_kinds(result), ["file"])
        self.assertIn("?", result.events[0].content)

    def test_late_result_error_preserves_streamed_text(self) -> None:
        result = _run(self.backend, [
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "hi there"},
            ]}},
            {"type": "result", "is_error": True, "error": "nope"},
        ])
        self.assertEqual(result.text, "hi there")
        self.assertEqual(result.error, "nope")

    def test_result_error_with_non_text_message_is_normalized(self) -> None:
        result = _run(self.backend, [{
            "type": "result", "is_error": True,
            "error": {"detail": "nope"},
        }])
        self.assertEqual(result.error, "Unknown error")


class CopilotParseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = CopilotBackend()

    def test_current_cli_nested_assistant_message(self) -> None:
        """Copilot CLI 1.0.70 wraps final output under ``data.content``."""
        event = {
            "type": "assistant.message",
            "data": {"content": "answer"},
        }
        result = _run(self.backend, [event])
        self.assertEqual(result.text, "answer")
        self.assertEqual(self.backend.extract_agent_text(event), "answer")

    def test_current_cli_uses_final_message_not_partial_deltas(self) -> None:
        result = _run(self.backend, [
            {"type": "assistant.message_delta",
             "data": {"deltaContent": "ans"}},
            {"type": "assistant.message", "data": {"content": "answer"}},
        ])
        self.assertEqual(result.text, "answer")
        self.assertEqual([event.content for event in result.events], ["answer"])

    def test_late_error_preserves_streamed_text(self) -> None:
        result = _run(self.backend, [
            {"type": "assistant.message", "data": {"content": "answer"}},
            {"type": "error", "message": "bad"},
        ])
        self.assertEqual(result.text, "answer")
        self.assertEqual(result.error, "bad")

    def test_error_with_non_text_message_is_normalized(self) -> None:
        result = _run(self.backend, [
            {"type": "error", "message": ["bad"]},
        ])
        self.assertEqual(result.error, "Unknown error")


class GrokParseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = GrokBackend()

    def test_tool_use_and_file_change(self) -> None:
        result = _run(self.backend, [{
            "type": "assistant", "message": {"content": [{
                "type": "tool_use", "name": "search_replace",
                "input": {"path": "src/main.py"},
            }]},
        }])
        self.assertEqual(_kinds(result), ["tool", "file"])
        self.assertIn("src/main.py", result.events[0].content)
        self.assertIn("src/main.py", result.events[1].content)

    def test_terminal_result_fallback_captures_usage_and_cost(self) -> None:
        event = {
            "type": "result", "subtype": "success", "result": "done",
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "total_cost_usd": 0.0123,
        }
        result = _run(self.backend, [event])
        self.assertEqual(result.text, "done")
        assert result.usage is not None
        self.assertEqual(result.usage["input_tokens"], 100)
        self.assertEqual(result.usage["total_cost_usd"], 0.0123)
        self.assertEqual(self.backend.extract_agent_text(event), "done")

    def test_terminal_errors_list_is_shown_to_the_user(self) -> None:
        """Grok Build uses ``errors``, not ``error``, on turn failures."""
        result = _run(self.backend, [{
            "type": "result",
            "subtype": "error_during_execution",
            "is_error": True,
            "errors": [
                "Couldn't set model 'gpt-5.4': unknown model id.",
            ],
        }])
        self.assertEqual(
            result.error,
            "Couldn't set model 'gpt-5.4': unknown model id.",
        )
        self.assertIn("unknown model id", result.text)

    def test_terminal_errors_without_text_stay_normalized(self) -> None:
        result = _run(self.backend, [{
            "type": "result", "is_error": True,
            "errors": [{"message": "not a supported shape"}],
        }])
        self.assertEqual(result.error, "Unknown error")

    def test_late_error_preserves_streamed_text(self) -> None:
        result = _run(self.backend, [
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "answer"},
            ]}},
            {"type": "error", "message": "stream closed"},
        ])
        self.assertEqual(result.text, "answer")
        self.assertEqual(result.error, "stream closed")

    def test_malformed_assistant_message_is_skipped(self) -> None:
        for message in (None, "bad", [], {"content": None}):
            with self.subTest(message=message):
                result = _run(self.backend, [{
                    "type": "assistant", "message": message,
                }])
                self.assertEqual(result.events, [])
                self.assertIsNone(self.backend.extract_agent_text({
                    "type": "assistant", "message": message,
                }))


class LlamaParseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = LlamaBackend()

    def test_late_error_preserves_streamed_text(self) -> None:
        result = _run(self.backend, [
            {"type": "assistant_text", "text": "yo"},
            {"type": "error", "message": "explode"},
        ])
        self.assertEqual(result.text, "yo")
        self.assertEqual(result.error, "explode")


class NonDictEventTests(unittest.TestCase):
    def test_malformed_type_fields_are_dropped_without_raising(self) -> None:
        for backend in (CopilotBackend(), GrokBackend()):
            for bad in (["assistant"], {"type": "assistant"}, 42, True):
                with self.subTest(backend=backend.name, type=bad):
                    event = {"type": bad, "text": "not an assistant event"}
                    result = AgentResult()
                    backend.parse_event(event, result)
                    self.assertEqual(result.events, [])
                    self.assertIsNone(backend.extract_agent_text(event))

    def test_malformed_claude_tool_name_is_safe_status_text(self) -> None:
        for name in (["Bash"], {"name": "Write"}, 42, True):
            with self.subTest(name=name):
                result = _run(ClaudeCodeBackend(), [{
                    "type": "assistant", "message": {"content": [{
                        "type": "tool_use", "name": name, "input": {},
                    }]},
                }])
                self.assertEqual(_kinds(result), ["tool"])
                self.assertEqual(result.events[0].content, "?")

    def test_malformed_http_text_and_tool_fields_preserve_string_contract(self) -> None:
        for text in (["reply"], {"text": "reply"}, 42, True):
            with self.subTest(text=text):
                result = _run(LlamaBackend(), [{"type": "assistant_text", "text": text}])
                self.assertEqual(result.text, "")
                self.assertEqual(result.events, [])
        for tool_input in (["path"], {"path": "x.py"}, "unexpected", 42):
            with self.subTest(input=tool_input):
                result = _run(LlamaBackend(), [{
                    "type": "tool_use", "name": ["write_file"],
                    "input": tool_input, "file_action": "write",
                }])
                self.assertTrue(all(isinstance(event.content, str) for event in result.events))

    def test_messages_style_reply_keeps_all_text_blocks_per_message(self) -> None:
        for backend in (ClaudeCodeBackend(), GrokBackend()):
            with self.subTest(backend=backend.name):
                result = _run(backend, [
                    {"type": "assistant", "message": {"content": "old commentary"}},
                    {"type": "assistant", "message": {"content": [
                        {"type": "text", "text": "first paragraph"},
                        {"type": "thinking", "thinking": "private"},
                        {"type": "text", "text": "second paragraph"},
                    ]}},
                    {"type": "result", "result": "first paragraph\nsecond paragraph"},
                ])
                self.assertEqual(result.text, "first paragraph\nsecond paragraph")

    def test_non_dict_events_are_dropped_without_raising(self) -> None:
        backends = [
            ClaudeCodeBackend(),
            CodexBackend(),
            CopilotBackend(),
            GrokBackend(),
            LlamaBackend(),
        ]
        for backend in backends:
            with self.subTest(backend=backend.name):
                for bad in ("junk", None, 123, ["x"], True):
                    result = AgentResult()
                    backend.parse_event(bad, result)  # type: ignore[arg-type]
                    self.assertEqual(result.events, [])
                    self.assertEqual(result.text, "")

    def test_non_dict_extract_returns_none_without_raising(self) -> None:
        from Cozter.backends_agent.base import (
            extract_messages_style_agent_text,
            terminal_result_text,
        )
        backends = [
            ClaudeCodeBackend(),
            CodexBackend(),
            CopilotBackend(),
            GrokBackend(),
            LlamaBackend(),
        ]
        for backend in backends:
            with self.subTest(backend=backend.name):
                for bad in ("junk", None, 123, ["x"], True):
                    self.assertIsNone(
                        backend.extract_agent_text(bad),  # type: ignore[arg-type]
                    )
        for helper in (
            extract_messages_style_agent_text, terminal_result_text,
        ):
            with self.subTest(helper=helper.__name__):
                for bad in ("junk", None, 123, ["x"]):
                    self.assertIsNone(helper(bad))  # type: ignore[arg-type]

    def test_non_dict_error_event_returns_false(self) -> None:
        from Cozter.backends_agent.base import record_error_event
        for bad in ("junk", None, 123, ["x"], True):
            with self.subTest(bad=bad):
                self.assertFalse(
                    record_error_event(bad, AgentResult()),  # type: ignore[arg-type]
                )

    def test_non_dict_terminal_event_is_dropped(self) -> None:
        from Cozter.backends_agent.base import apply_terminal_result_event
        for bad in ("junk", None, 123, ["x"], True):
            with self.subTest(bad=bad):
                result = AgentResult()
                self.assertIsNone(
                    apply_terminal_result_event(bad, result),  # type: ignore[arg-type]
                )
                self.assertEqual(result.events, [])

    def test_truncate_status_text_rejects_junk_limits(self) -> None:
        from Cozter.backends_agent.base import truncate_status_text
        self.assertEqual(truncate_status_text("hello", limit=200), "hello")
        self.assertEqual(truncate_status_text("hello world", limit=5), "hell…")
        self.assertEqual(truncate_status_text("hello world", limit=0), "")
        for junk in ("5", None, 1.9, 4.5):
            with self.subTest(junk=junk):
                out = truncate_status_text("hello world", limit=junk)  # type: ignore[arg-type]
                self.assertIsInstance(out, str)
        self.assertEqual(
            truncate_status_text("hello world", limit=4.0),  # type: ignore[arg-type]
            "hel…",
        )


if __name__ == "__main__":
    unittest.main()
