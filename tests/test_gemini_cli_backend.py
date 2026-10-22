"""Gemini launches retain exact sessions and cannot widen restricted workers."""

import asyncio
import os
from pathlib import Path
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import execution_state
from Cozter.backends_agent import gemini_cli as gemini
from Cozter.backends_agent.base import AgentResult


class GeminiCliBackendTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        await self.enterAsyncContext(execution_state.execution_scope(self.ws, "gemini-edge-job", "Original work"))
        self.record = await execution_state.record("gemini-launch")
        self.enterContext(mock.patch.object(execution_state, "current_run", return_value=self.record))
        self.enterContext(mock.patch.object(gemini, "_system_policy_directory", return_value=self.ws))
        self.backend = gemini.GeminiCliBackend()
        self.launch = self.enterContext(mock.patch.object(gemini, "create_prompt_subprocess", new_callable=mock.AsyncMock))
        self.enterContext(mock.patch.object(gemini, "executable_command", return_value=["node", "gemini.js"]))

    async def test_full_overrides_inherited_sandbox_without_losing_native_home_or_large_stdin(self):
        prompt = "native plugin command and unicode 한🙂\n" * 5000
        process = SimpleNamespace(pid=42)
        self.launch.return_value = process
        with mock.patch.dict(os.environ, {"GEMINI_SANDBOX": "true", "GEMINI_CLI_HOME": self.ws}):
            await self.backend.launch(self.ws, prompt, "gemini-3.8-flash", "full", effort=100)
        cmd, stdin = self.launch.call_args.args
        self.assertEqual(stdin, prompt)
        self.assertNotIn(prompt, cmd)
        self.assertEqual(cmd[cmd.index("--approval-mode") + 1], "yolo")
        self.assertNotIn("--admin-policy", cmd)
        self.assertNotIn("--effort", cmd)
        self.assertEqual(self.launch.call_args.kwargs["env"]["GEMINI_SANDBOX"], "false")
        self.assertEqual(self.launch.call_args.kwargs["env"]["GEMINI_CLI_HOME"], self.ws)
        native_id = self.record.get("native_session_id")
        self.assertEqual(cmd[cmd.index("--session-id") + 1], native_id)
        self.assertFalse(self.record.get("native_resume_attempted"))
        await self.backend.launch(self.ws, "Only the new guidance", "gemini-3.8-flash", "full")
        resumed = self.launch.call_args.args[0]
        self.assertEqual(resumed[resumed.index("--resume") + 1], native_id)
        self.assertNotIn("--session-id", resumed)
        self.assertTrue(self.record.get("native_resume_attempted"))

    async def test_api_model_pin_keeps_native_session_without_alias_or_effort_fallback(self):
        self.launch.return_value = SimpleNamespace(pid=42)
        await self.backend.launch(self.ws, "Original request", "gemini-4-argon", "full")
        native_id = self.record.get("native_session_id")
        fresh = self.launch.call_args.args[0]
        self.assertEqual(fresh[fresh.index("--model") + 1], "gemini-4-argon")
        self.assertEqual(fresh[fresh.index("--session-id") + 1], native_id)
        self.assertNotIn("--resume", fresh)
        await self.backend.launch(self.ws, "Only new guidance", "gemini-4-argon", "full", effort=100)
        cmd, prompt = self.launch.call_args.args
        self.assertIn("gemini-4-argon", self.backend.available_models)
        self.assertEqual(cmd[cmd.index("--model") + 1], "gemini-4-argon")
        self.assertEqual(cmd[cmd.index("--resume") + 1], native_id)
        self.assertEqual(cmd[cmd.index("--approval-mode") + 1], "yolo")
        self.assertNotIn("--session-id", cmd)
        self.assertNotIn("--effort", cmd)
        self.assertEqual(prompt, "Only new guidance")
        self.assertEqual(self.backend.context_window_tokens("gemini-4-argon"), 1_048_576)
        for model in (None, "auto", "pro", "gemini-4-argon-private"):
            with self.subTest(model=model):
                self.assertIsNone(self.backend.context_window_tokens(model))

    async def test_compaction_cannot_reuse_worker_session_or_widen_full_approval(self):
        worker_id = "6ca6c4ef-cc6e-4c0b-a11a-5f93bb7708f3"
        await self.record.save(native_session_id=worker_id)
        process = SimpleNamespace(pid=42)
        self.launch.return_value = process
        await self.backend.launch(self.ws, "Summarize", None, "full", compaction=True, effort=99)
        cmd = self.launch.call_args.args[0]
        policy_path = cmd[cmd.index("--admin-policy") + 1]
        policy = tomllib.loads(Path(policy_path).read_text(encoding="utf-8"))
        try:
            self.assertTrue(all(row["decision"] == "deny" for row in policy["rule"]))
            self.assertEqual(cmd[cmd.index("--approval-mode") + 1], "plan")
            self.assertNotIn("--resume", cmd)
            self.assertNotEqual(cmd[cmd.index("--session-id") + 1], worker_id)
            self.assertEqual(self.record.get("native_session_id"), worker_id)
        finally:
            await self.backend.cleanup_process(process)
        self.assertFalse(Path(policy_path).exists())

    async def test_per_session_trust_flag_preserves_inherited_environment_and_restricted_policy(self):
        for mode in ("full", "auto", "confirm", "deny"):
            with self.subTest(mode=mode), mock.patch.dict(os.environ, {"GEMINI_CLI_TRUST_WORKSPACE": "false"}):
                process = SimpleNamespace(pid=42)
                self.launch.return_value = process
                await self.backend.launch(self.ws, "Authorized workspace task", None, mode)
                cmd = self.launch.call_args.args[0]
                self.assertIn("--skip-trust", cmd)
                self.assertEqual(self.launch.call_args.kwargs["env"]["GEMINI_CLI_TRUST_WORKSPACE"], "false")
                self.assertEqual(os.environ["GEMINI_CLI_TRUST_WORKSPACE"], "false")
                if mode in {"confirm", "deny"}:
                    path = cmd[cmd.index("--admin-policy") + 1]
                    policy = tomllib.loads(Path(path).read_text(encoding="utf-8"))
                    self.assertTrue(any(row["toolName"] == "*" and row["decision"] == "deny"
                                        for row in policy["rule"]))
                await self.backend.cleanup_process(process)

    async def test_same_pid_launches_keep_separate_read_policies_and_cancel_cleans_its_file(self):
        processes = [SimpleNamespace(pid=42), SimpleNamespace(pid=42)]
        self.launch.side_effect = processes
        paths = []
        for process in processes:
            await self.backend.launch(self.ws, "Read the current skill", None, "confirm")
            cmd = self.launch.call_args.args[0]
            paths.append(cmd[cmd.index("--admin-policy") + 1])
        self.assertNotEqual(*paths)
        policy = tomllib.loads(Path(paths[0]).read_text(encoding="utf-8"))
        self.assertEqual(policy["rule"][1]["mcpName"], "*")
        self.assertEqual(policy["rule"][1]["decision"], "deny")
        allowed = next(row["toolName"] for row in policy["rule"] if row["decision"] == "allow")
        self.assertEqual(set(allowed), {"read_file", "list_directory", "glob", "grep_search"})
        await self.backend.cleanup_process(processes[1])
        self.assertTrue(Path(paths[0]).exists())
        self.assertFalse(Path(paths[1]).exists())
        await self.backend.cleanup_process(processes[0])
        self.assertFalse(Path(paths[0]).exists())

        async def cancel(cmd, *_args, **_kwargs):
            paths.append(cmd[cmd.index("--admin-policy") + 1])
            raise asyncio.CancelledError

        self.launch.side_effect = cancel
        with self.assertRaises(asyncio.CancelledError):
            await self.backend.launch(self.ws, "Read only", None, "confirm")
        self.assertFalse(Path(paths[-1]).exists())

    async def test_central_policy_suppression_fails_closed_before_any_launch(self):
        Path(self.ws, "central.toml").write_text('[[rule]]\ntoolName="*"\ndecision="allow"\npriority=999\n')
        with mock.patch.object(gemini, "_write_policy", wraps=gemini._write_policy) as write:
            for mode in ("confirm", "deny"):
                with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, "central admin policies"):
                    await self.backend.launch(self.ws, "Inspect only", None, mode)
            write.assert_not_called()
        self.launch.assert_not_awaited()

    async def test_policy_cleanup_covers_bad_launch_options_and_vision_hint_uses_valid_images(self):
        paths = []
        original = gemini._write_policy

        def write(mode):
            path = original(mode)
            paths.append(path)
            return path

        with (mock.patch.object(gemini, "_write_policy", side_effect=write),
              mock.patch.object(self.backend, "append_launch_options", side_effect=ValueError("bad options"))):
            with self.assertRaisesRegex(ValueError, "bad options"):
                await self.backend.launch(self.ws, "Read only", None, "confirm")
        self.assertFalse(Path(paths[-1]).exists())
        image = Path(self.ws, "attachment.png")
        image.write_bytes(b"image fixture")
        self.launch.return_value = SimpleNamespace(pid=42)
        await self.backend.launch(self.ws, "[image attachment saved to: attachment.png]", None, "full")
        self.assertIn("use your read_file tool", self.launch.call_args.args[1])
        await self.backend.launch(self.ws, "[image attachment saved to: ../private.png]", None, "full")
        self.assertNotIn("[Vision:", self.launch.call_args.args[1])

    async def test_interleaved_deltas_warning_and_terminal_failure_keep_each_visible_reply(self):
        left, right = AgentResult(), AgentResult()
        for output, text in ((left, "Left 한"), (right, "Right"), (left, "🙂 answer"), (right, " answer")):
            self.backend.parse_event({"type": "message", "role": "assistant", "content": text, "delta": True}, output)
        self.backend.parse_event({"type": "message", "role": "user", "content": "Do not echo"}, left)
        self.backend.parse_event({"type": "error", "severity": "warning", "message": "Nonfatal warning"}, left)
        self.assertIsNone(left.error)
        self.backend.parse_event({"type": "result", "status": "error", "error": {"message": "API failure"},
                                 "stats": {"input_tokens": 10, "output_tokens": 5}}, left)
        self.assertEqual(left.text, "Left 한🙂 answer")
        self.assertEqual(right.text, "Right answer")
        self.assertEqual(left.error, "API failure")
        self.assertEqual(left.usage, {"input_tokens": 10, "output_tokens": 5})

    async def test_exact_missing_resume_can_recover_only_before_native_activity(self):
        native_id = "6ca6c4ef-cc6e-4c0b-a11a-5f93bb7708f3"
        error = (f'Invalid session identifier "{native_id}".\n  Searched for sessions in C:\\safe\\chats.\n'
                 '  Use --list-sessions to see available sessions, then use --resume {number}, '
                 '--resume {uuid}, or --resume latest.')
        for active in (False, True):
            with self.subTest(active=active):
                await self.record.save(native_session_id=native_id, native_resume_attempted=True,
                                       native_tools_started=active, native_resume_unavailable=False)
                result = AgentResult()
                self.backend.parse_event({"type": "init", "session_id": "wrong-session"}, result)
                self.assertEqual(self.record.get("native_session_id"), native_id)
                self.backend.parse_event({"type": "error", "severity": "error", "message": error}, result)
                self.assertEqual(self.record.get("native_resume_unavailable"), not active)
                self.assertEqual(self.record.get("native_session_id"), native_id if active else None)

    async def test_stderr_missing_session_requires_exact_startup_failure_and_no_prior_activity(self):
        native_id = "6ca6c4ef-cc6e-4c0b-a11a-5f93bb7708f3"
        missing = "Error resuming session: No previous sessions found for this project."
        cases = [
            (42, False, "Startup warning\n" + missing, True),
            (1, False, missing, False),
            (42, True, missing, False),
            (42, False, "Authentication failed; sign in again", False),
            (42, False, "Permission denied loading session", False),
            (42, False, 'Error resuming session: Invalid session identifier "another-id".', False),
        ]
        for code, active, stderr, recovered in cases:
            with self.subTest(code=code, active=active, stderr=stderr):
                await self.record.save(native_session_id=native_id, native_resume_attempted=True,
                                       native_tools_started=active, native_resume_unavailable=False)
                self.backend.observe_process_exit(code, stderr)
                self.assertEqual(self.record.get("native_resume_unavailable"), recovered)
                self.assertEqual(self.record.get("native_session_id"), None if recovered else native_id)
