"""Native CLI continuity is exact, execution-scoped and entirely offline."""

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from Cozter import execution_state
from Cozter.backends_agent import claude_code, codex, copilot, grok
from Cozter.backends_agent.base import AgentResult, NativeSessionUnavailableError
from Cozter.utils import kill_and_wait


_SID = "01a126fa-e54a-7753-8078-3e31ff8fc962"
_OTHER_SID = "f53ba06b-a880-4e59-93ea-bd623b0b922b"
_CASES = (
    (codex, codex.CodexBackend, "gpt-6.1-sol"),
    (claude_code, claude_code.ClaudeCodeBackend, "opus"),
    (copilot, copilot.CopilotBackend, "fixture-model"),
    (grok, grok.GrokBackend, "grok-4.7"),
)


class NativeCliContinuationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.account_home = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.dict(os.environ, {"COPILOT_HOME": self.account_home}))

    @asynccontextmanager
    async def scope(self, backend, model, *, job="job", worker="worker", approval="full", effort=100):
        async with execution_state.execution_scope(self.ws, job, "synthetic offline task"):
            with execution_state.worker_scope(worker):
                async with execution_state.backend_scope(backend.name, model, approval, effort) as record:
                    yield record

    async def launch(self, module, backend, model, *, compaction=False, approval="full", effort=100):
        factory = "create_captured_subprocess" if module is grok else "create_prompt_subprocess"
        proc = mock.Mock()
        with (
            mock.patch.object(module, "executable_command", return_value=[backend.name]),
            mock.patch.object(module, factory, mock.AsyncMock(return_value=proc)) as create,
        ):
            await backend.launch(self.ws, "portable evidence and remaining scope", model, approval,
                                 compaction=compaction, effort=effort)
        return list(create.await_args.args[0]), create.await_args, proc

    def started_event(self, module, native_id):
        if module is codex:
            return {"type": "thread.started", "thread_id": native_id}
        if module is copilot:
            return {"type": "session.start", "data": {"sessionId": native_id}}
        return {"type": "system", "subtype": "init", "session_id": native_id}

    def error_event(self, module, message):
        if module is claude_code:
            return {"type": "result", "subtype": "error_during_execution", "is_error": True,
                    "result": message}
        if module is copilot:
            return {"type": "session.error", "data": {"message": message}}
        return {"type": "error", "message": message}

    async def confirm_native(self, module, backend, record, native_id=_SID):
        if module is copilot:
            native_id = record.get("native_candidate_session_id")
            # Match the documented CLI persistence layout, not the user home.
            (record.directory / "copilot-home" / "session-state" / native_id).mkdir(parents=True)
        backend.parse_event(self.started_event(module, native_id), AgentResult())
        await record.flush()
        self.assertEqual(record.get("native_session_id"), native_id)
        return native_id

    async def test_no_execution_context_preserves_private_stateless_launches(self):
        for module, cls, model in _CASES:
            with self.subTest(backend=cls.name):
                backend = cls()
                cmd, _, proc = await self.launch(module, backend, model)
                self.assertNotIn("--resume", cmd)
                self.assertNotIn("resume", cmd)
                if module is codex:
                    self.assertIn("--ephemeral", cmd)
                if module is claude_code:
                    self.assertIn("--no-session-persistence", cmd)
                await backend.cleanup_process(proc)

    async def test_all_four_capture_confirmed_ids_and_resume_after_execution_reload(self):
        for module, cls, model in _CASES:
            with self.subTest(backend=cls.name):
                backend = cls()
                async with self.scope(backend, model, job=backend.name) as record:
                    cmd, call, proc = await self.launch(module, backend, model)
                    self.assertNotIn("--ephemeral", cmd)
                    self.assertNotIn("--no-session-persistence", cmd)
                    self.assertNotIn("--resume", cmd)
                    native_id = await self.confirm_native(module, backend, record)
                    home = call.kwargs.get("env", {}).get("COPILOT_HOME")
                    await backend.cleanup_process(proc)
                    if module is copilot:
                        self.assertTrue(Path(home).is_dir())
                        self.assertEqual(record.get("native_home"), "copilot-home")
                # New Python-side Execution objects still recover the exact native ID.
                async with self.scope(backend, model, job=backend.name) as record:
                    self.assertEqual(record.get("native_session_id"), native_id)
                    cmd, call, proc = await self.launch(module, backend, model)
                    flag = "resume" if module is codex else "--resume"
                    self.assertEqual(cmd[cmd.index(flag) + 1], native_id)
                    self.assertFalse(any(arg in ("--last", "--continue", "-c") for arg in cmd
                                         if module is not codex))
                    self.assertNotIn("--fork-session", cmd)
                    self.assertNotIn("--restore-code", cmd)
                    self.assertNotIn("--ephemeral", cmd)
                    if module is copilot:
                        self.assertEqual(call.kwargs["env"]["COPILOT_HOME"], home)
                        self.assertIn("--no-remote-export", cmd)
                        self.assertNotIn("--session-id", cmd)
                    await backend.cleanup_process(proc)

    async def test_compaction_ignores_an_outer_execution_record(self):
        for module, cls, model in _CASES:
            with self.subTest(backend=cls.name):
                backend = cls()
                async with self.scope(backend, model) as record:
                    await record.save(native_session_id=_SID)
                    snapshot = record.get("native_session_id")
                    cmd, _, proc = await self.launch(module, backend, model, compaction=True,
                                                     approval="deny")
                    self.assertNotIn("--resume", cmd)
                    self.assertNotIn("resume", cmd)
                    self.assertNotIn(_SID, cmd)
                    self.assertEqual(record.get("native_session_id"), snapshot)
                    if module is codex:
                        self.assertIn("--ephemeral", cmd)
                    if module is claude_code:
                        self.assertIn("--no-session-persistence", cmd)
                    await backend.cleanup_process(proc)

    async def test_current_model_permission_effort_options_survive_resume(self):
        expected = {
            "codex": ("--dangerously-bypass-approvals-and-sandbox", "model_reasoning_effort=ultra"),
            "claude_code": ("--dangerously-skip-permissions", "max"),
            "copilot": ("--yolo", "max"),
            "grok": ("--always-approve", "xhigh"),
        }
        for module, cls, model in _CASES:
            with self.subTest(backend=cls.name):
                backend = cls()
                async with self.scope(backend, model) as record:
                    await record.save(native_session_id=_SID)
                    if module is copilot:
                        (record.directory / "copilot-home" / "session-state" / _SID).mkdir(parents=True)
                    cmd, _, proc = await self.launch(module, backend, model)
                    self.assertIn(model, cmd)
                    for value in expected[backend.name]:
                        self.assertIn(value, cmd)
                    if module is codex:
                        self.assertLess(cmd.index("-C"), cmd.index("resume"))
                        self.assertLess(cmd.index(expected[backend.name][0]), cmd.index("resume"))
                    await backend.cleanup_process(proc)

    async def test_ids_cannot_be_names_prefixes_paths_flags_or_cross_worker_events(self):
        bad = (None, True, 1, "--last", "previous", "../../session", _SID[:8], _SID + "\n", [], {})
        for module, cls, model in _CASES:
            with self.subTest(backend=cls.name):
                backend = cls()
                async with self.scope(backend, model) as record:
                    await record.save(native_session_id=_SID)
                    for value in bad:
                        backend.parse_event(self.started_event(module, value), AgentResult())
                    backend.parse_event(self.started_event(module, _OTHER_SID), AgentResult())
                    self.assertEqual(record.get("native_session_id"), _SID)

    async def test_invalid_persisted_id_is_cleared_before_a_fresh_launch(self):
        for module, cls, model in _CASES:
            with self.subTest(backend=cls.name):
                backend = cls()
                async with self.scope(backend, model) as record:
                    await record.save(native_session_id="--last")
                    cmd, _, proc = await self.launch(module, backend, model)
                    self.assertNotIn("--last", cmd)
                    self.assertNotIn("resume", cmd)
                    self.assertNotIn("--resume", cmd)
                    self.assertIsNone(record.get("native_session_id"))
                    await backend.cleanup_process(proc)

    async def test_native_ids_do_not_cross_job_worker_model_permission_or_effort(self):
        backend = codex.CodexBackend()
        async with self.scope(backend, "gpt-6.1-sol") as record:
            await record.save(native_session_id=_SID)
        boundaries = (
            {"job": "other"}, {"worker": "other"}, {"approval": "auto"}, {"effort": 50},
        )
        for kwargs in boundaries:
            async with self.scope(backend, "gpt-6.1-sol", **kwargs) as record:
                self.assertIsNone(record.get("native_session_id"))
        async with self.scope(backend, "gpt-6-luna") as record:
            self.assertIsNone(record.get("native_session_id"))
        async with self.scope(backend, "gpt-6.1-sol") as record:
            self.assertEqual(record.get("native_session_id"), _SID)

    async def test_copilot_unconfirmed_candidate_never_becomes_a_resume_target(self):
        backend = copilot.CopilotBackend()
        async with self.scope(backend, "fixture-model") as record:
            first, _, proc = await self.launch(copilot, backend, "fixture-model")
            candidate = first[first.index("--session-id") + 1]
            backend.parse_event(self.started_event(copilot, _OTHER_SID), AgentResult())
            self.assertIsNone(record.get("native_session_id"))
            await backend.cleanup_process(proc)
        async with self.scope(backend, "fixture-model") as record:
            second, _, proc = await self.launch(copilot, backend, "fixture-model")
            self.assertNotIn("--resume", second)
            self.assertNotEqual(second[second.index("--session-id") + 1], candidate)
            await backend.cleanup_process(proc)

    async def test_copilot_missing_owned_native_state_requests_bootstrap_before_process_launch(self):
        backend = copilot.CopilotBackend()
        async with self.scope(backend, "fixture-model") as record:
            await record.save(native_session_id=_SID)
            with (
                mock.patch.object(copilot, "create_prompt_subprocess", new=mock.AsyncMock()) as create,
                mock.patch.object(copilot, "executable_command") as executable,
            ):
                with self.assertRaises(NativeSessionUnavailableError) as raised:
                    await backend.launch(
                        self.ws, "incremental guidance only", "fixture-model", "full", effort=100,
                    )
                create.assert_not_awaited()
                executable.assert_not_called()
            self.assertEqual(raised.exception.backend_name, backend.name)
            self.assertEqual(raised.exception.native_session_id, _SID)
            self.assertIsNone(record.get("native_session_id"))
            self.assertEqual(record.get("native_unavailable_session_id"), _SID)
            self.assertTrue(record.get("native_resume_unavailable"))
            self.assertFalse(record.get("native_resume_attempted"))
            self.assertFalse(record.get("native_tools_started"))

            cmd, _, proc = await self.launch(copilot, backend, "fixture-model")
            self.assertNotIn("--resume", cmd)
            self.assertIn("--session-id", cmd)
            self.assertIn("fixture-model", cmd)
            self.assertIn("--yolo", cmd)
            self.assertIn("max", cmd)
            self.assertIsNone(record.get("native_session_id"))
            self.assertEqual(record.get("native_unavailable_session_id"), _SID)
            await backend.cleanup_process(proc)

    async def test_copilot_private_home_rejects_links_outside_execution_state(self):
        backend = copilot.CopilotBackend()
        async with self.scope(backend, "fixture-model") as record:
            home = record.directory / "copilot-home"
            home.mkdir()
            target = Path(self.account_home) / "other-session-state"
            target.mkdir()
            try:
                (home / "session-state").symlink_to(target, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks unavailable: {exc}")
            with self.assertRaisesRegex(RuntimeError, "inside its execution home"):
                await self.launch(copilot, backend, "fixture-model")
            self.assertEqual(list(target.iterdir()), [])

    async def test_exact_missing_session_flag_is_only_set_before_native_activity(self):
        for module, cls, model in _CASES:
            with self.subTest(backend=cls.name):
                backend = cls()
                async with self.scope(backend, model) as record:
                    await record.save(native_session_id=_SID, native_resume_attempted=True,
                                      native_tools_started=False)
                    # A wrong ID, quoted text, generic provider error or authentication failure
                    # must not authorize a fresh run.
                    for message in (f"No session found with ID: {_OTHER_SID}",
                                    f"Tool said: No session found with ID: {_SID}",
                                    "Provider unavailable", "Authentication failed"):
                        backend.parse_event(self.error_event(module, message), AgentResult())
                        self.assertFalse(record.get("native_resume_unavailable"))
                    backend.parse_event(self.error_event(module, f"No session found with ID: {_SID}"),
                                        AgentResult())
                    self.assertTrue(record.get("native_resume_unavailable"))
                    self.assertIsNone(record.get("native_session_id"))
                    for provider_missing in (
                        f"Error: thread/resume: thread/resume failed: no rollout found for thread id {_SID} (code -32600)",
                        f"no rollout found for conversation id {_SID}",
                        f"Error: No session, task, or name matched '{_SID}'.",
                        f"No conversation found with session ID: {_SID}",
                        f'warning: conversation "{_SID}" not found',
                    ):
                        await record.save(native_session_id=_SID, native_resume_attempted=True,
                                          native_resume_unavailable=False, native_tools_started=False)
                        backend.parse_event(self.error_event(module, provider_missing), AgentResult())
                        self.assertTrue(record.get("native_resume_unavailable"))
                        self.assertIsNone(record.get("native_session_id"))
                    await record.save(native_session_id=_SID, native_resume_attempted=True,
                                      native_resume_unavailable=False, native_tools_started=False)
                    activity = ({"type": "turn.started"} if module is codex else
                                {"type": "assistant.message", "data": {"content": "started"}}
                                if module is copilot else
                                {"type": "assistant", "message": {"content": []}})
                    backend.parse_event(activity, AgentResult())
                    backend.parse_event(self.error_event(module, f"No session found with ID: {_SID}"),
                                        AgentResult())
                    self.assertFalse(record.get("native_resume_unavailable"))
                    self.assertEqual(record.get("native_session_id"), _SID)

    async def test_resumed_cli_processes_remain_owned_and_cancellable(self):
        script = "import json,sys,time; print(json.dumps(sys.argv[1:]),flush=True); time.sleep(3600)"
        for module, cls, model in _CASES:
            with self.subTest(backend=cls.name):
                backend = cls()
                async with self.scope(backend, model) as record:
                    await record.save(native_session_id=_SID)
                    if module is copilot:
                        (record.directory / "copilot-home" / "session-state" / _SID).mkdir(parents=True)
                    launched = asyncio.Event()
                    procs = []

                    async def worker():
                        proc = await backend.launch(self.ws, "offline test", model, "full", effort=100)
                        procs.append(proc)
                        try:
                            args = json.loads(await proc.stdout.readline())
                            flag = "resume" if module is codex else "--resume"
                            self.assertEqual(args[args.index(flag) + 1], _SID)
                            launched.set()
                            await proc.wait()
                        finally:
                            await kill_and_wait(proc)
                            await backend.cleanup_process(proc)

                    with mock.patch.object(module, "executable_command",
                                           return_value=[sys.executable, "-c", script]):
                        task = asyncio.create_task(worker())
                        try:
                            await asyncio.wait_for(launched.wait(), 5)
                            task.cancel()
                            with self.assertRaises(asyncio.CancelledError):
                                await asyncio.wait_for(task, 5)
                        finally:
                            if not task.done():
                                task.cancel()
                                await asyncio.gather(task, return_exceptions=True)
                    self.assertIsNotNone(procs[0].returncode)
                    self.assertEqual(record.get("native_session_id"), _SID)
