"""Only real Antigravity response/tool work contributes to native progress."""

from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent, flexible, supervision, utils
from Cozter.backends_agent.antigravity_cli import AntigravityCliBackend
from Cozter.backends_agent._http_process import HttpAgentProcess


class AntigravityNativeProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_internal_drain_keeps_corrected_terminal_protocol_without_delta_duplication_or_missing_prefix(self):
        final = ("[OWNER]\ntier: low\nreason: Retain complete requested scope\n"
                 "task: Implement and verify the whole requested change\n[/OWNER]")
        backend = AntigravityCliBackend()
        self.assertFalse(backend.agent_text_is_delta)
        for streamed in (False, True):
            with self.subTest(streamed=streamed):
                proc = HttpAgentProcess("Offline Antigravity terminal response")

                async def produce():
                    proc.emit({"event": "init", "conversation_id": "internal-id", "init": {}})
                    if streamed:
                        for fragment in ("[OWNER]\ntier: high\n", "reason: Incomplete earlier draft"):
                            proc.emit({"event": "step_update", "step_update": {
                                "step_type": "agent_response", "state": "ACTIVE", "text_delta": fragment,
                            }})
                    proc.emit({"event": "result", "result": {"status": "SUCCESS", "response": final}})

                proc.start(produce())
                text = await utils.drain_llm_subprocess(proc, backend, "Offline Antigravity internal call")
                self.assertEqual(text, final)
                self.assertEqual(flexible.parse_plan(text, "Original scope").subtasks[0].tier, "low")
                self.assertEqual(proc.returncode, 0)

    def test_response_fragments_count_fresh_output_without_echo_checkpoint_or_terminal_double_count(self):
        proc = SimpleNamespace(pid=-1)
        with (mock.patch.object(supervision, "note_output") as note,
              mock.patch.object(supervision, "attach_process") as attach):
            for event in (
                {"event": "init", "conversation_id": "native-id", "init": {"cwd": "workspace"}},
                {"event": "step_update", "step_update": {"step_type": "user_input", "state": "DONE", "text_delta": "Echo"}},
                {"event": "step_update", "step_update": {"step_type": "checkpoint", "state": "DONE"}},
                {"event": "result", "result": {"status": "SUCCESS", "response": "Authoritative final response"}},
            ):
                agent._note_native_progress(proc, event)
            note.assert_not_called()
            for text in ("same", "same", " end"):
                agent._note_native_progress(proc, {"event": "step_update", "step_update": {
                    "step_type": "agent_response", "state": "ACTIVE", "text_delta": text,
                }})
            self.assertEqual(note.call_args_list, [mock.call("same"), mock.call("same"), mock.call(" end")])
            attach.assert_not_called()

    def test_step_identity_keeps_identical_tool_results_distinct_and_commandline_attributed(self):
        proc = SimpleNamespace(pid=-1)
        with (mock.patch.object(supervision, "note_output") as note,
              mock.patch.object(supervision, "attach_process") as attach):
            for step_index in (3, 5):
                step = {"conversation_id": "native-id", "step_index": step_index,
                        "step_type": "tool", "tool_name": "run_command", "state": "ACTIVE",
                        "tool_info": {"name": "run_command", "parameters": {"CommandLine": "cmake --build build"}}}
                agent._note_native_progress(proc, {"event": "step_update", "step_update": step})
                attach.assert_called_with(proc, command="cmake --build build")
                step = {**step, "state": "DONE", "tool_info": {**step["tool_info"], "output": "Verified"}}
                agent._note_native_progress(proc, {"event": "step_update", "step_update": step})
                count = note.call_count
                agent._note_native_progress(proc, {"event": "step_update", "step_update": step})
                self.assertEqual(note.call_count, count)
            self.assertEqual(note.call_args_list, [mock.call("Verified"), mock.call("tool completed"),
                                                   mock.call("Verified"), mock.call("tool completed")])
