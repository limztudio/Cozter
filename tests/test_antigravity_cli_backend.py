"""Antigravity stream and launch edges retain permissions and native context."""

import asyncio
import json
import os
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import execution_state
from Cozter.backends_agent import antigravity_cli as agy
from Cozter.backends_agent.base import AgentResult, NativeSessionUnavailableError


class AntigravityCliBackendTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        await self.enterAsyncContext(execution_state.execution_scope(self.ws, "agy-edge-job", "Original work"))
        self.record = await execution_state.record("agy-launch")
        self.enterContext(mock.patch.object(execution_state, "current_run", return_value=self.record))
        self.backend = agy.AntigravityCliBackend()
        self.launch = self.enterContext(mock.patch.object(agy, "create_prompt_subprocess", new_callable=mock.AsyncMock))
        self.enterContext(mock.patch.object(agy, "_command_prefix", return_value=["agy.exe"]))
        self.launch.return_value = SimpleNamespace(pid=42)
        self.piped = self.enterContext(mock.patch.object(agy, "create_piped_subprocess", side_effect=self._create_resume))
        self.reap = self.enterContext(mock.patch.object(agy, "reap_failed_subprocess", new_callable=mock.AsyncMock))
        self.resume_processes = []
        self.resume_override = None
        self.resume_ready = asyncio.Event()

    async def _create_resume(self, command, **_kwargs):
        native_id = command[command.index("--conversation") + 1]
        process = SimpleNamespace(pid=42, stdout=asyncio.StreamReader(), stderr=asyncio.StreamReader(), stdin=mock.Mock())
        process.stdin.drain = mock.AsyncMock()
        self.resume_processes.append(process)
        if self.resume_override is not None:
            self.resume_override(process, native_id)
        else:
            process.stdout.feed_data(json.dumps({"event": "init", "conversation_id": native_id}).encode() + b"\n")
        process.stderr.feed_eof()
        self.resume_ready.set()
        return process

    async def asyncTearDown(self):
        for process in self.resume_processes:
            await self.backend.cleanup_process(process)

    async def test_large_framed_stdin_adopts_native_id_then_resumes_only_that_id(self):
        prompt = 'Quotes " newline\n unicode 한🙂' * 5000
        await self.backend.launch(self.ws, prompt, "gemini-4-argon-medium", "full", effort=99)
        cmd, wire = self.launch.call_args.args
        self.assertEqual(len(wire.splitlines()), 1)
        self.assertEqual(json.loads(wire), {"event": "user", "message": {"content": prompt}})
        self.assertNotIn(prompt, cmd)
        self.assertEqual(cmd[cmd.index("--input-format") + 1], "stream-json")
        self.assertEqual(cmd[cmd.index("--output-format") + 1], "stream-json")
        self.assertEqual(cmd[cmd.index("--print-timeout") + 1], "0")
        self.assertNotIn("--print", cmd)
        self.assertNotIn("--continue", cmd)
        self.assertNotIn("--conversation", cmd)
        self.assertIn("--dangerously-skip-permissions", cmd)
        self.assertIn("--sandbox=false", cmd)
        native_id = "6ca6c4ef-cc6e-4c0b-a11a-5f93bb7708f3"
        self.backend.parse_event({"event": "init", "conversation_id": native_id, "init": {}}, AgentResult())
        await self.backend.launch(self.ws, "Only new guidance", "gemini-4-argon-medium", "auto")
        resumed = self.piped.call_args.args[0]
        await asyncio.sleep(0)
        wire = b"".join(call.args[0] for call in self.resume_processes[-1].stdin.write.call_args_list)
        self.assertEqual(resumed[resumed.index("--conversation") + 1], native_id)
        self.assertEqual(resumed[resumed.index("--mode") + 1], "auto")
        self.assertNotIn("--dangerously-skip-permissions", resumed)
        self.assertEqual(json.loads(wire)["message"]["content"], "Only new guidance")
        self.assertTrue(self.record.get("native_resume_attempted"))

    async def test_unverified_restrictions_fail_before_launch_or_session_changes(self):
        native_id = "6ca6c4ef-cc6e-4c0b-a11a-5f93bb7708f3"
        await self.record.save(native_session_id=native_id, native_tools_started=True)
        for approval, compact in (("confirm", False), ("deny", False), ("other", False), ("full", True)):
            with self.subTest(approval=approval, compact=compact), self.assertRaisesRegex(RuntimeError, "restricted policy"):
                await self.backend.launch(self.ws, "Must not run tools", None, approval, compaction=compact)
        self.launch.assert_not_awaited()
        self.assertEqual(self.record.get("native_session_id"), native_id)
        self.assertTrue(self.record.get("native_tools_started"))
        self.assertFalse(self.backend.supports_read_only_workers)
        self.assertNotIn("restricted", self.backend.permission_arg_sets)

    async def test_interleaved_partial_streams_use_authoritative_terminal_reply(self):
        left, right = AgentResult(), AgentResult()
        for target, text in ((left, "draft "), (right, "other"), (left, "partial")):
            event = {"event": "step_update", "step_update": {"step_type": "agent_response", "state": "ACTIVE", "text_delta": text}}
            self.backend.parse_event(event, target)
            self.assertIsNone(self.backend.extract_agent_text(event))
        tool = {"event": "step_update", "step_update": {
            "step_type": "tool", "state": "DONE", "tool_name": "run_command", "tool_info": {
                "parameters": {"CommandLine": "git status"}, "error": {"message": "benign tool failure"},
            },
        }}
        self.backend.parse_event(tool, left)
        self.assertIsNone(left.error)
        terminal = {"event": "result", "result": {"status": "SUCCESS", "response": "Corrected final reply", "usage": {
            "input_tokens": 100, "output_tokens": 12, "thinking_tokens": 5, "cache_read_tokens": 80,
        }}}
        self.backend.parse_event(terminal, left)
        self.assertEqual(left.text, "Corrected final reply")
        self.assertEqual([event.content for event in left.events if event.kind == "text"], ["Corrected final reply"])
        self.assertTrue(any(event.kind == "tool" and event.content == "benign tool failure" for event in left.events))
        self.assertEqual(right.text, "other")
        self.assertEqual(self.backend.extract_agent_text(terminal), "Corrected final reply")
        self.assertEqual(left.usage["cache_read_tokens"], 80)

    async def test_terminal_errors_keep_partial_text_and_never_replace_pinned_uuid(self):
        native_id = "6ca6c4ef-cc6e-4c0b-a11a-5f93bb7708f3"
        await self.record.save(native_session_id=native_id, native_resume_attempted=True)
        for status in ("ERROR", "CANCELED", "INTERRUPTED", "INVALID", "WAITING", "RUNNING", None):
            with self.subTest(status=status):
                result = AgentResult(text="Partial work retained")
                self.backend.parse_event({"event": "result", "result": {
                    "conversation_id": "another-conversation", "status": status, "response": "", "error": {},
                }}, result)
                self.assertEqual(result.text, "Partial work retained")
                self.assertTrue(result.error)
                self.assertEqual(self.record.get("native_session_id"), native_id)
        malformed = AgentResult()
        self.backend.parse_event({"event": "result", "result": []}, malformed)
        self.assertTrue(malformed.error)

    async def test_exact_missing_uuid_recovery_stops_after_prompt_acceptance(self):
        native_id = "6ca6c4ef-cc6e-4c0b-a11a-5f93bb7708f3"
        for active, error, recover in (
            (False, f"Conversation {native_id} not found", True),
            (False, "Authentication required", False),
            (False, "Conversation another-id not found", False),
            (True, f"Conversation {native_id} not found", False),
        ):
            with self.subTest(active=active, error=error):
                await self.record.save(native_session_id=native_id, native_resume_attempted=True,
                                       native_tools_started=False, native_resume_unavailable=False)
                result = AgentResult()
                if active:
                    self.backend.parse_event({"event": "step_update", "step_update": {
                        "step_type": "user_input", "state": "DONE", "conversation_id": native_id,
                    }}, result)
                self.backend.parse_event({"event": "result", "result": {"status": "ERROR", "error": error}}, result)
                self.assertEqual(self.record.get("native_resume_unavailable"), recover)
                self.assertEqual(self.record.get("native_session_id"), None if recover else native_id)

    async def test_effort_only_uses_levels_published_for_same_model_family(self):
        models = (*agy._MODELS, "private-reasoner-low", "private-reasoner-high", "private-reasoner-max")
        with mock.patch.object(agy, "_MODELS", models):
            for percent, expected in ((0, None), (99, "high"), (100, "max")):
                with self.subTest(percent=percent):
                    await self.backend.launch(self.ws, "Work", "private-reasoner-low", "full", effort=percent)
                    cmd = self.launch.call_args.args[0]
                    if expected is None:
                        self.assertNotIn("--effort", cmd)
                    else:
                        self.assertEqual(cmd[cmd.index("--effort") + 1], expected)
        await self.backend.launch(self.ws, "Work", "unknown-private-model", "full", effort=100)
        self.assertNotIn("--effort", self.launch.call_args.args[0])
        self.assertEqual(self.backend.context_window_tokens("gemini-4-argon-high"), 1_048_576)
        self.assertIsNone(self.backend.context_window_tokens("gemini-4-argon-private"))

    async def test_live_catalog_excludes_fallback_models_and_effort_reads_never_probe(self):
        output = (b"Model\tName\nstartup notice\n"
                  b"gemini-4-argon-low\tGemini 4 Argon (Low)\n"
                  b"gemini-4-argon-high\tGemini 4 Argon (High)\n"
                  b"gemini-4-argon-high\tDuplicate\n"
                  b"private-reasoner-xhigh\tPrivate (Extra high)\n"
                  b"private-reasoner-max\tPrivate (Max)\n")
        with mock.patch.object(agy.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=output)) as run:
            self.assertEqual(self.backend.available_models, (
                "gemini-4-argon-low", "gemini-4-argon-high", "private-reasoner-xhigh", "private-reasoner-max",
            ))
            self.assertEqual(self.backend.effort_levels_for_model("gemini-4-argon-high"), ("low", "high"))
            self.assertEqual(self.backend.describe_effort(99, "private-reasoner-max"), "xhigh")
            self.assertEqual(self.backend.describe_effort(100, "private-reasoner-max"), "max")
            self.assertEqual(run.call_args.kwargs["timeout"], 15)
            self.assertEqual(self.backend.available_models[-1], "private-reasoner-max")
            run.assert_called_once()
        self.backend._catalog_expires_at = time.monotonic() - 1
        with mock.patch.object(agy.subprocess, "run", side_effect=AssertionError("effort must not discover")):
            self.assertEqual(self.backend.effort_levels_for_model("gemini-4-argon-high"), ("low", "medium", "high"))
            self.assertEqual(self.backend.describe_effort(99, "gemini-4-argon-high"), "medium")
            self.assertEqual(self.backend.describe_effort(100, "gemini-4-argon-high"), "high")
            self.assertEqual(self.backend.describe_effort(99, "gpt-oss-120b-medium"), "backend default (no lower effort level for this model)")
            self.assertEqual(self.backend.describe_effort(100, "gpt-oss-120b-medium"), "medium")

    async def test_cancelled_launch_retains_exact_resume_identity_for_next_attempt(self):
        native_id = "6ca6c4ef-cc6e-4c0b-a11a-5f93bb7708f3"
        await self.record.save(native_session_id=native_id)
        self.piped.side_effect = asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError):
            await self.backend.launch(self.ws, "New guidance", None, "full")
        self.assertEqual(self.record.get("native_session_id"), native_id)
        self.assertEqual(self.piped.call_args.args[0][-2:], ["--conversation", native_id])

    async def test_invalid_saved_alias_is_not_resumed_or_allowed_to_block_new_uuid(self):
        native_id = "6ca6c4ef-cc6e-4c0b-a11a-5f93bb7708f3"
        await self.record.save(native_session_id="latest")
        await self.backend.launch(self.ws, "Complete history bootstrap", None, "full")
        self.assertNotIn("--conversation", self.launch.call_args.args[0])
        self.backend.parse_event({"event": "init", "conversation_id": native_id, "init": {}}, AgentResult())
        self.assertEqual(self.record.get("native_session_id"), native_id)

    async def test_terminal_only_activity_prevents_replay_and_missing_result_never_succeeds(self):
        native_id = "6ca6c4ef-cc6e-4c0b-a11a-5f93bb7708f3"
        await self.record.save(native_session_id=native_id, native_resume_attempted=True,
                               native_tools_started=False, native_resume_unavailable=False)
        result = AgentResult()
        self.backend.parse_event({"event": "result", "result": {
            "status": "ERROR", "response": "Partial change completed", "error": f"Conversation {native_id} not found",
        }}, result)
        self.backend.finalize_result(result)
        self.assertTrue(result.terminal_result_received)
        self.assertEqual(result.text, "Partial change completed")
        self.assertTrue(self.record.get("native_tools_started"))
        self.assertFalse(self.record.get("native_resume_unavailable"))
        self.assertEqual(self.record.get("native_session_id"), native_id)
        partial = AgentResult(text="Keep streamed work")
        self.backend.finalize_result(partial)
        self.assertEqual(partial.text, "Keep streamed work")
        self.assertIn("without a final result", partial.error)
        prior_error = AgentResult(error="Actual authentication failure")
        self.backend.finalize_result(prior_error)
        self.assertEqual(prior_error.error, "Actual authentication failure")
        success = AgentResult()
        self.backend.parse_event({"event": "result", "result": {"status": "SUCCESS", "response": "Final answer"}}, success)
        self.backend.finalize_result(success)
        self.assertIsNone(success.error)

    async def test_resume_mismatch_reaps_before_any_prompt_byte_or_activity(self):
        native_id = "6ca6c4ef-cc6e-4c0b-a11a-5f93bb7708f3"
        other_id = "0db27f53-2116-4e99-ae78-99def40b9210"
        await self.record.save(native_session_id=native_id)

        def mismatch(process, _native_id):
            process.stdout.feed_data(json.dumps({"event": "init", "conversation_id": other_id}).encode() + b"\n")

        self.resume_override = mismatch
        with self.assertRaises(NativeSessionUnavailableError):
            await self.backend.launch(self.ws, "Delta that needs existing context", None, "full")
        process = self.resume_processes[-1]
        process.stdin.write.assert_not_called()
        self.reap.assert_awaited_once_with(process)
        self.assertTrue(self.record.get("native_resume_unavailable"))
        self.assertEqual(self.record.get("native_unavailable_session_id"), native_id)
        self.assertIsNone(self.record.get("native_session_id"))
        self.assertFalse(self.record.get("native_tools_started"))
        self.launch.assert_not_awaited()

    async def test_cancel_during_resume_confirmation_reaps_without_delivering_prompt(self):
        native_id = "6ca6c4ef-cc6e-4c0b-a11a-5f93bb7708f3"
        await self.record.save(native_session_id=native_id)
        self.resume_override = lambda _process, _native_id: None
        launch = asyncio.create_task(self.backend.launch(self.ws, "Never sent", None, "full"))
        await self.resume_ready.wait()
        await asyncio.sleep(0)
        launch.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await launch
        process = self.resume_processes[-1]
        process.stdin.write.assert_not_called()
        self.reap.assert_awaited_once_with(process)
        self.assertEqual(self.record.get("native_session_id"), native_id)
        self.assertFalse(self.record.get("native_resume_unavailable"))
        self.assertFalse(self.backend._stdin_tasks)

    async def test_large_resume_writer_yields_to_output_reader_and_cleanup_owns_it(self):
        native_id = "6ca6c4ef-cc6e-4c0b-a11a-5f93bb7708f3"
        await self.record.save(native_session_id=native_id)
        draining = asyncio.Event()

        async def blocked_drain():
            draining.set()
            await asyncio.Event().wait()

        def matching(process, expected):
            process.stdout.feed_data(json.dumps({"event": "init", "conversation_id": expected}).encode() + b"\n")
            process.stdin.drain.side_effect = blocked_drain

        self.resume_override = matching
        process = await asyncio.wait_for(self.backend.launch(self.ws, "한🙂" * 100_000, None, "full"), 1)
        await draining.wait()
        self.assertEqual(len(process.stdin.write.call_args.args[0]), 64 * 1024)
        process.stdin.close.assert_not_called()
        await self.backend.cleanup_process(process)
        process.stdin.close.assert_called_once()
        self.assertFalse(self.backend._stdin_tasks)

    async def test_resume_authentication_eof_preserves_identity_and_never_retries_delta(self):
        native_id = "6ca6c4ef-cc6e-4c0b-a11a-5f93bb7708f3"
        await self.record.save(native_session_id=native_id)

        def failed(process, _expected):
            process.stderr.feed_data(b"Authentication required; sign in with agy\n")
            process.stdout.feed_eof()

        self.resume_override = failed
        with self.assertRaisesRegex(RuntimeError, "Authentication required"):
            await self.backend.launch(self.ws, "Never sent", None, "full")
        process = self.resume_processes[-1]
        process.stdin.write.assert_not_called()
        self.reap.assert_awaited_once_with(process)
        self.assertEqual(self.record.get("native_session_id"), native_id)
        self.assertFalse(self.record.get("native_resume_unavailable"))

    async def test_broken_resume_input_cannot_turn_terminal_success_into_completed_work(self):
        native_id = "6ca6c4ef-cc6e-4c0b-a11a-5f93bb7708f3"
        await self.record.save(native_session_id=native_id)

        def matching(process, expected):
            process.stdout.feed_data(json.dumps({"event": "init", "conversation_id": expected}).encode() + b"\n")
            process.stdin.drain.side_effect = BrokenPipeError

        self.resume_override = matching
        process = await self.backend.launch(self.ws, "Only new guidance", None, "full")
        await asyncio.sleep(0)
        await self.backend.cleanup_process(process)
        result = AgentResult(text="Premature response", terminal_result_received=True)
        self.backend.finalize_result(result)
        self.assertIn("could not receive", result.error)
        self.assertEqual(result.text, "Premature response")
        self.reap.assert_awaited_once_with(process)


class AntigravityExecutableTests(unittest.TestCase):
    def test_canonical_install_fills_stale_path_but_never_overrides_resolved_cli(self):
        with (mock.patch.object(agy, "resolve_executable_prefix", return_value=["custom-agy.exe"]),
              mock.patch.object(agy.os.path, "isfile") as isfile):
            self.assertEqual(agy._command_prefix(), ["custom-agy.exe"])
            isfile.assert_not_called()
        with (mock.patch.object(agy, "resolve_executable_prefix", return_value=None),
              mock.patch.object(agy.sys, "platform", "win32"),
              mock.patch.dict(os.environ, {"ProgramFiles": r"C:\Program Files"}),
              mock.patch.object(agy.os.path, "isfile", return_value=True)):
            path = os.path.join(r"C:\Program Files", "AntigravityCLI", "agy.exe")
            self.assertEqual(agy._command_prefix(), [path])
            ok, detail = agy.AntigravityCliBackend().health_check()
            self.assertTrue(ok)
            self.assertIn(path, detail)
        with (mock.patch.object(agy, "resolve_executable_prefix", return_value=None),
              mock.patch.object(agy.os.path, "isfile", return_value=False)):
            self.assertFalse(agy.AntigravityCliBackend().health_check()[0])
