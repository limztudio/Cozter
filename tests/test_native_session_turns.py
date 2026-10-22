"""Offline native CLI launches across completed Cozter conversation turns."""

import asyncio
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import uuid

from Cozter import agent, execution_state, flexible, session, skills
from Cozter.backends_agent import claude_code, codex, copilot, grok
from Cozter.backends_agent.base import SHELL_INPUT_RULE, AgentResult


_CASES = (
    (codex, codex.CodexBackend, "gpt-6.1-sol"),
    (claude_code, claude_code.ClaudeCodeBackend, "opus"),
    (copilot, copilot.CopilotBackend, "fixture-model"),
    (grok, grok.GrokBackend, "grok-4.7"),
)


class NativeSessionTurnsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        account_home = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.dict(os.environ, {"COPILOT_HOME": account_home}))
        self.backends = {cls.name: cls() for _, cls, _ in _CASES}
        self.models = {cls.name: model for _, cls, model in _CASES}
        self.modules = {cls.name: module for module, cls, _ in _CASES}
        self.attempts: list[dict] = []
        self.launches: list[dict] = []
        self.serial = 0
        self.effort = 0
        self.missing_provider = False
        self.fail_after_activity = False
        self.enterContext(mock.patch.object(
            agent.backends_agent, "get_backend", side_effect=lambda name=None: self.backends[name or "codex"],
        ))
        self.enterContext(mock.patch.object(agent, "_compaction_context_targets", return_value=()))
        self.enterContext(mock.patch.object(agent, "_snapshot_attachment_images", return_value={}))
        self.enterContext(mock.patch.object(agent, "_append_new_attachment_images", new=mock.AsyncMock()))
        self.enterContext(mock.patch.object(agent, "_run_post_turn_maintenance", new=mock.AsyncMock()))
        self.enterContext(mock.patch.object(agent, "_build_backend_prompt", side_effect=(
            lambda _backend, prompt, **_kwargs: "POLICY-PRELUDE-MARKER\n" + prompt
        )))
        self.enterContext(mock.patch.object(
            agent.workspace_mod, "get_reasoning_effort", side_effect=lambda _ws: self.effort,
        ))
        self.judge = self.enterContext(mock.patch.object(
            agent, "_judge_draft", new=mock.AsyncMock(return_value=(flexible.JudgeVerdict(False), False)),
        ))
        self.enterContext(mock.patch.object(agent, "_drive_backend_impl", side_effect=self._drive))
        for module, cls, _ in _CASES:
            self.enterContext(mock.patch.object(module, "executable_command", return_value=[cls.name]))
            factory = "create_captured_subprocess" if module is grok else "create_prompt_subprocess"
            self.enterContext(mock.patch.object(module, factory, side_effect=self._create_process))

    def _conversation(self, workspace_path: str | None = None) -> str:
        ws = workspace_path or self.ws
        data = session.create_session(ws, name="Offline native conversation")
        session.append_messages(ws, data["id"], [
            {"role": "user", "content": "HISTORICAL-REQUEST-MARKER"},
            {"role": "assistant", "content": "HISTORICAL-ANSWER-MARKER"},
        ])
        session.set_last_session(ws, 7, data["id"])
        return data["id"]

    async def _create_process(self, command, *args, **kwargs):
        command = list(command)
        if command[0] == "grok":
            prompt = Path(command[command.index("--prompt-file") + 1]).read_text(encoding="utf-8")
        else:
            prompt = args[0]
        self.launches.append({
            "backend": command[0], "command": command, "prompt": prompt,
            "home": kwargs.get("env", {}).get("COPILOT_HOME"),
        })
        return mock.Mock()

    @staticmethod
    def _started(module, native_id):
        if module is codex:
            return {"type": "thread.started", "thread_id": native_id}
        if module is copilot:
            return {"type": "session.start", "data": {"sessionId": native_id}}
        return {"type": "system", "subtype": "init", "session_id": native_id}

    @staticmethod
    def _answer(module, text):
        if module is codex:
            return {"type": "item.completed", "item": {"type": "agent_message", "text": text}}
        if module is copilot:
            return {"type": "assistant.message", "data": {"content": text}}
        return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}

    @staticmethod
    def _missing(module, native_id):
        message = f"No session found with ID: {native_id}"
        if module is claude_code:
            return {"type": "result", "subtype": "error_during_execution", "is_error": True, "result": message}
        if module is copilot:
            return {"type": "session.error", "data": {"message": message}}
        return {"type": "error", "message": message}

    async def _drive(self, backend, workspace_path, prompt, model, approval, *, effort, **_kwargs):
        module = self.modules[backend.name]
        record = execution_state.current_run()
        self.assertIsNotNone(record)
        previous_id = record.get("native_session_id")
        attempt = {"backend": backend.name, "prompt": prompt, "previous_id": previous_id}
        self.attempts.append(attempt)
        proc = await backend.launch(workspace_path, prompt, model, approval, effort=effort)
        result = AgentResult()
        try:
            if previous_id and self.missing_provider:
                self.missing_provider = False
                if self.fail_after_activity:
                    backend.parse_event(self._answer(module, "ACTIVITY-BEFORE-ERROR-MARKER"), result)
                backend.parse_event(self._missing(module, previous_id), result)
                return result, False
            native_id = previous_id or record.get("native_candidate_session_id")
            if native_id is None:
                native_id = str(uuid.uuid4())
            if module is copilot:
                (record.directory / "copilot-home" / "session-state" / native_id).mkdir(parents=True, exist_ok=True)
            backend.parse_event(self._started(module, native_id), result)
            answer = f"ANSWER-MARKER-{len(self.attempts)}"
            backend.parse_event(self._answer(module, answer), result)
            self.assertEqual(result.text, answer)
            self.assertEqual(record.get("native_session_id"), native_id)
            attempt["native_id"] = native_id
            attempt["home"] = self.launches[-1]["home"]
            return result, False
        finally:
            await backend.cleanup_process(proc)

    async def _turn(self, name, request, *, sid=None, model=None, approval="full", workspace_path=None):
        self.serial += 1
        ws = workspace_path or self.ws
        job = f"offline-completed-job-{self.serial}"
        result = await agent.run(
            request, ws, 7, model=model or self.models[name], approval=approval,
            summary_model="offline-summary", backend_name=name, summary_backend_name="codex",
            session_id=sid, execution_id=job,
        )
        # The bot removes only the completed queue job; native state must survive.
        await execution_state.discard_execution(ws, job)
        await asyncio.sleep(0)
        return result

    async def test_all_four_resume_across_completed_jobs_without_repeating_own_context(self):
        for module, cls, _model in _CASES:
            with self.subTest(backend=cls.name):
                sid = self._conversation()
                first = await self._turn(cls.name, "FIRST-REQUEST-MARKER")
                initial_attempt = self.attempts[-1]
                initial_launch = self.launches[-1]
                self.assertEqual(first.session_id, sid)
                for marker in ("POLICY-PRELUDE-MARKER", "HISTORICAL-REQUEST-MARKER",
                               "HISTORICAL-ANSWER-MARKER", "FIRST-REQUEST-MARKER"):
                    self.assertIn(marker, initial_launch["prompt"])
                second = await self._turn(cls.name, "SECOND-REQUEST-MARKER")
                later_attempt = self.attempts[-1]
                later_launch = self.launches[-1]
                self.assertEqual(second.session_id, sid)
                self.assertEqual(later_attempt["previous_id"], initial_attempt["native_id"])
                flag = "resume" if module is codex else "--resume"
                self.assertEqual(later_launch["command"][later_launch["command"].index(flag) + 1],
                                 initial_attempt["native_id"])
                self.assertEqual(later_launch["prompt"], SHELL_INPUT_RULE + "\n\n" + skills.refresh_instructions(cli=True)
                                 + "\n\nNew message:\nSECOND-REQUEST-MARKER")
                if module is copilot:
                    self.assertEqual(initial_launch["home"], later_launch["home"])
                    self.assertTrue(Path(later_launch["home"]).is_dir())

    async def test_switching_backend_supplies_intervening_messages_before_resuming_its_own_id(self):
        sid = self._conversation()
        await self._turn("codex", "CODEX-ORIGINAL-REQUEST", sid=sid)
        original_id = self.attempts[-1]["native_id"]
        await self._turn("claude_code", "CLAUDE-INTERVENING-REQUEST", sid=sid)
        intervening_answer = self.attempts[-1]["native_id"]
        await self._turn("codex", "CODEX-RETURN-REQUEST", sid=sid)

        returned = self.attempts[-1]
        self.assertEqual(returned["previous_id"], original_id)
        self.assertNotEqual(original_id, intervening_answer)
        self.assertIn("CLAUDE-INTERVENING-REQUEST", returned["prompt"])
        self.assertIn("ANSWER-MARKER-2", returned["prompt"])
        self.assertNotIn("CODEX-ORIGINAL-REQUEST", returned["prompt"])
        self.assertNotIn("HISTORICAL-REQUEST-MARKER", returned["prompt"])
        self.assertIn("CODEX-RETURN-REQUEST", returned["prompt"])

    async def test_detached_completion_after_logging_lock_release_is_not_acknowledged_as_own_pair(self):
        sid = self._conversation()
        persist = agent._persist_session_write
        appended = False

        async def persist_then_complete_detached(workspace_path, write):
            nonlocal appended
            outcome = await persist(workspace_path, write)
            if not appended:
                appended = True
                lock = agent.workspace_mod.get_lock(workspace_path)
                self.assertFalse(lock.locked())
                # A detached provider can finish after the foreground append
                # releases the file lock, before its native acknowledgment.
                async with lock:
                    session.append_messages(workspace_path, sid, [
                        {"role": "user", "content": "DETACHED-COMPLETION-REQUEST-MARKER"},
                        {"role": "assistant", "content": "DETACHED-COMPLETION-ANSWER-MARKER"},
                    ])
            return outcome

        with mock.patch.object(agent, "_persist_session_write", side_effect=persist_then_complete_detached):
            await self._turn("codex", "OWN-FOREGROUND-REQUEST-MARKER", sid=sid)
        native_id = self.attempts[-1]["native_id"]
        self.assertTrue(appended)

        await self._turn("codex", "AFTER-DETACHED-COMPLETION-MARKER", sid=sid)

        returned = self.attempts[-1]
        self.assertEqual(returned["previous_id"], native_id)
        self.assertIn("DETACHED-COMPLETION-REQUEST-MARKER", returned["prompt"])
        self.assertIn("DETACHED-COMPLETION-ANSWER-MARKER", returned["prompt"])
        self.assertIn("AFTER-DETACHED-COMPLETION-MARKER", returned["prompt"])
        self.assertNotIn("OWN-FOREGROUND-REQUEST-MARKER", returned["prompt"])
        self.assertNotIn("ANSWER-MARKER-1", returned["prompt"])
        self.assertNotIn("HISTORICAL-REQUEST-MARKER", returned["prompt"])
        self.assertNotIn("POLICY-PRELUDE-MARKER", returned["prompt"])

    async def test_new_cozter_session_model_permission_effort_and_workspace_start_fresh(self):
        sid = self._conversation()
        await self._turn("codex", "ORIGINAL-IDENTITY-REQUEST", sid=sid)
        original_id = self.attempts[-1]["native_id"]
        other_sid = self._conversation()
        for kwargs in ({"sid": other_sid}, {"sid": sid, "model": "gpt-6-luna"},
                       {"sid": sid, "approval": "auto"}):
            with self.subTest(kwargs=kwargs):
                await self._turn("codex", "SEPARATE-IDENTITY-REQUEST", **kwargs)
                self.assertIsNone(self.attempts[-1]["previous_id"])
                self.assertNotEqual(self.attempts[-1]["native_id"], original_id)
                self.assertIn("POLICY-PRELUDE-MARKER", self.attempts[-1]["prompt"])
        self.effort = 50
        await self._turn("codex", "SEPARATE-EFFORT-REQUEST", sid=sid)
        self.assertIsNone(self.attempts[-1]["previous_id"])
        self.effort = 0
        other_ws = self.enterContext(tempfile.TemporaryDirectory())
        # Even an identical Cozter session ID is distinct in another workspace.
        session.save_session(other_ws, sid, session.load_session(self.ws, sid))
        await self._turn("codex", "SEPARATE-WORKSPACE-REQUEST", sid=sid, workspace_path=other_ws)
        self.assertIsNone(self.attempts[-1]["previous_id"])
        await self._turn("codex", "RETURN-ORIGINAL-IDENTITY", sid=sid)
        self.assertEqual(self.attempts[-1]["previous_id"], original_id)

    async def test_confirmed_missing_provider_retries_once_with_full_bootstrap_for_all_four(self):
        for _module, cls, _model in _CASES:
            with self.subTest(backend=cls.name):
                sid = self._conversation()
                await self._turn(cls.name, "BEFORE-MISSING-REQUEST", sid=sid)
                original_id = self.attempts[-1]["native_id"]
                start = len(self.attempts)
                self.missing_provider = True
                await self._turn(cls.name, "RECOVERED-REQUEST", sid=sid)
                attempts = self.attempts[start:]
                self.assertEqual(len(attempts), 2)
                self.assertEqual(attempts[0]["previous_id"], original_id)
                self.assertEqual(attempts[0]["prompt"], SHELL_INPUT_RULE + "\n\n" + skills.refresh_instructions(cli=True)
                                 + "\n\nNew message:\nRECOVERED-REQUEST")
                self.assertIsNone(attempts[1]["previous_id"])
                for marker in ("HISTORICAL-REQUEST-MARKER", "HISTORICAL-ANSWER-MARKER",
                               "BEFORE-MISSING-REQUEST", "RECOVERED-REQUEST", "POLICY-PRELUDE-MARKER"):
                    self.assertIn(marker, attempts[1]["prompt"])
                self.assertNotEqual(attempts[1]["native_id"], original_id)

    async def test_missing_copilot_home_bootstraps_before_new_launch(self):
        sid = self._conversation()
        await self._turn("copilot", "COPILOT-INITIAL-REQUEST", sid=sid)
        original_id = self.attempts[-1]["native_id"]
        owned_home = Path(self.launches[-1]["home"])
        self.assertTrue(str(owned_home.resolve()).startswith(str(Path(self.ws).resolve()) + os.sep))
        shutil.rmtree(owned_home)
        attempts_before, launches_before = len(self.attempts), len(self.launches)

        await self._turn("copilot", "COPILOT-RECOVERY-REQUEST", sid=sid)

        attempts = self.attempts[attempts_before:]
        launches = self.launches[launches_before:]
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(launches), 1)
        self.assertEqual(attempts[0]["previous_id"], original_id)
        self.assertEqual(attempts[0]["prompt"], SHELL_INPUT_RULE + "\n\n" + skills.refresh_instructions(cli=True)
                         + "\n\nNew message:\nCOPILOT-RECOVERY-REQUEST")
        self.assertIsNone(attempts[1]["previous_id"])
        self.assertNotIn("--resume", launches[0]["command"])
        self.assertIn("HISTORICAL-REQUEST-MARKER", launches[0]["prompt"])
        self.assertIn("COPILOT-INITIAL-REQUEST", launches[0]["prompt"])
        self.assertNotEqual(attempts[1]["native_id"], original_id)

    async def test_missing_session_error_after_native_activity_never_replays_for_all_four(self):
        for _module, cls, _model in _CASES:
            with self.subTest(backend=cls.name):
                sid = self._conversation()
                await self._turn(cls.name, "COMPLETED-PRIOR-REQUEST", sid=sid)
                native_id = self.attempts[-1]["native_id"]
                attempts_before, launches_before = len(self.attempts), len(self.launches)
                self.missing_provider = True
                self.fail_after_activity = True

                await self._turn(cls.name, "REQUEST-WITH-AMBIGUOUS-OUTCOME", sid=sid)

                self.assertEqual(len(self.attempts) - attempts_before, 1)
                self.assertEqual(len(self.launches) - launches_before, 1)
                self.assertEqual(self.attempts[-1]["previous_id"], native_id)
                self.assertEqual(self.attempts[-1]["prompt"], SHELL_INPUT_RULE + "\n\n" + skills.refresh_instructions(cli=True)
                                 + "\n\nNew message:\nREQUEST-WITH-AMBIGUOUS-OUTCOME")

    async def test_judge_followup_sends_only_new_instruction_but_missing_session_restores_draft(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                sid = self._conversation()
                start = len(self.attempts)
                judge_calls = 0

                async def judge(*_args, **_kwargs):
                    nonlocal judge_calls
                    judge_calls += 1
                    if judge_calls == 1:
                        self.missing_provider = missing
                        return flexible.JudgeVerdict(True, "final proof", "VERIFY-REMAINING-MARKER"), False
                    return flexible.JudgeVerdict(False), False

                with mock.patch.object(agent, "_judge_draft", side_effect=judge):
                    await self._turn("codex", "FULL-ORIGINAL-SCOPE-MARKER", sid=sid)

                attempts = self.attempts[start:]
                self.assertEqual(len(attempts), 3 if missing else 2)
                incremental = attempts[1]["prompt"]
                self.assertIn("VERIFY-REMAINING-MARKER", incremental)
                self.assertNotIn("FULL-ORIGINAL-SCOPE-MARKER", incremental)
                self.assertNotIn("ANSWER-MARKER", incremental)
                self.assertNotIn("POLICY-PRELUDE-MARKER", incremental)
                if missing:
                    self.assertIn("FULL-ORIGINAL-SCOPE-MARKER", attempts[2]["prompt"])
                    self.assertIn(f"ANSWER-MARKER-{start + 1}", attempts[2]["prompt"])
                    self.assertIn("VERIFY-REMAINING-MARKER", attempts[2]["prompt"])
                    self.assertIn("HISTORICAL-REQUEST-MARKER", attempts[2]["prompt"])


if __name__ == "__main__":
    unittest.main()
