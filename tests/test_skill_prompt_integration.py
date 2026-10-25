"""Workspace skill metadata follows execution roles without replaying bodies."""

import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent, agent_tools, execution_state, skills, utils, workspace
from Cozter.backends_agent.base import SHELL_INPUT_RULE, AgentResult, ChatEvent
from Cozter.tests import test_native_session_turns as native_turns


_NATIVE_ID = "01a126fa-e54a-7753-8078-3e31ff8fc962"
_BODY = "FULL-SKILL-BODY-MUST-BE-LOADED-PROGRESSIVELY"


class SkillPromptIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.skill_path = Path(self.ws, ".cozter", "skills", "proof-workflow.md")
        self.skill_path.parent.mkdir(parents=True)
        self._write_skill("SKILL-CATALOG-DESCRIPTION")
        self.native = SimpleNamespace(
            name="codex", supports_typed_plugins=False, supports_plugin_prelude=True,
            supports_detached_tasks=False,
        )
        self.enterContext(mock.patch.object(agent_tools, "cli_plugin_prelude", return_value="PLUGIN-PRELUDE"))

    def _write_skill(self, description: str) -> None:
        self.skill_path.write_text(
            "---\nname: proof-workflow\ndescription: " + description
            + "\n---\n\n# Proof workflow\n\n" + _BODY + "\n",
            encoding="utf-8",
        )

    def _native_prompt(self, catalog: str, request="current request"):
        return agent._native_turn_prompt(
            self.native, request, {"messages": []}, [], budget=100_000,
            collaborative=True, is_scheduled=False, skills_catalog=catalog,
        )

    async def _flexible(self, context="historical context\ncurrent request"):
        return await agent._run_flexible(
            context, "current request", self.ws, approval="full", effort=0,
            collaborative=False, summary_backend_name="codex", summary_model="offline-summary",
            on_event=None, inject_queue=None, injected=[],
        )

    async def test_public_run_binds_one_workspace_catalog_after_acquiring_run_lock(self) -> None:
        async def turn(*_args, **_kwargs):
            self.assertTrue(workspace.get_run_lock(self.ws).locked())
            catalog = skills.current_catalog(self.ws)
            self.assertTrue(catalog.entries)
            self.assertIn("SKILL-CATALOG-DESCRIPTION", skills.catalog_prompt(catalog, cli=True))
            self.assertIsNotNone(execution_state.current_execution())
            return AgentResult(text="offline result")

        with mock.patch.object(agent, "_run_turn", side_effect=turn):
            result = await agent.run("current request", self.ws, 7, execution_id="offline-job")

        self.assertEqual(result.text, "offline result")

    async def test_native_catalog_is_separate_and_only_changed_metadata_is_resent(self) -> None:
        async with skills.scope(self.ws) as catalog:
            index = skills.catalog_prompt(catalog, cli=True)
        execution = execution_state.Execution(self.ws, "offline-native-prompt", "request")
        record = await execution.record({"backend": "codex", "worker": "main"})
        try:
            first = self._native_prompt(index)
            self.assertIn("skills catalog", first.sections)
            self.assertNotIn("SKILL-CATALOG-DESCRIPTION", first.sections["policy and plugins"])
            self.assertIn("SKILL-CATALOG-DESCRIPTION", first.render(None))
            self.assertNotIn(_BODY, first.render(None))
            await record.save(native_session_id=_NATIVE_ID, native_tools_started=True)
            await first.acknowledge(record)

            self.assertEqual(
                self._native_prompt(index, "next request").render(record),
                SHELL_INPUT_RULE + "\n\n" + skills.refresh_instructions(cli=True) + "\n\nNew message:\nnext request",
            )
            self._write_skill("UPDATED-SKILL-CATALOG-DESCRIPTION")
            async with skills.scope(self.ws) as updated_catalog:
                updated = self._native_prompt(skills.catalog_prompt(updated_catalog, cli=True), "new request")
            rendered = updated.render(record)
            self.assertIn("[Updated skills catalog; replaces the previous version]", rendered)
            self.assertIn("UPDATED-SKILL-CATALOG-DESCRIPTION", rendered)
            self.assertNotIn("PLUGIN-PRELUDE", rendered)
            self.assertNotIn(_BODY, rendered)
            self.assertTrue(rendered.endswith("New message:\nnew request"))
        finally:
            await execution.flush()

    async def test_http_user_prompt_does_not_duplicate_its_system_skill_catalog(self) -> None:
        backend = SimpleNamespace(
            name="llama", supports_typed_plugins=True, supports_plugin_prelude=False,
            supports_detached_tasks=False,
        )
        async with skills.scope(self.ws) as catalog:
            prompt = agent._build_backend_prompt(
                backend, "current request", collaborative=False,
                skills_catalog=skills.catalog_prompt(catalog),
            )

        self.assertNotIn("SKILL-CATALOG-DESCRIPTION", prompt)
        self.assertNotIn(_BODY, prompt)
        self.assertIn("current request", prompt)

    async def test_planner_catalog_survives_history_cap_and_worker_gets_the_same_metadata(self) -> None:
        internal_calls = []
        worker_calls = []

        async def internal(_backend, _ws, prompt, _model, *, label, **_kwargs):
            internal_calls.append((label, prompt, agent_tools.skill_planning_active()))
            if label == "Flexible planner":
                return "1. [low] use proof-workflow to inspect logs\n2. [high] complete repairs"
            if label == "Flexible merge":
                return "verified result"
            return "[DONE]"

        async def drive(backend, _ws, prompt, _model, _approval, **kwargs):
            worker_calls.append((backend.name, prompt, kwargs["native_context"], agent_tools.skill_planning_active()))
            return AgentResult(text="worker proof", events=[ChatEvent("text", "worker proof")]), False

        context = "OLD-HISTORY-MARKER " * 5_000 + "current request"
        async with skills.scope(self.ws), execution_state.execution_scope(self.ws, "planner-job", "current request"):
            with (mock.patch.object(agent, "run_internal_backend", side_effect=internal),
                  mock.patch.object(agent, "_drive_backend", side_effect=drive)):
                result, restarting = await self._flexible(context)
                await agent._judge_draft(
                    "current request", "verified result", self.ws,
                    summary_backend_name="codex", summary_model="offline-summary",
                    inject_queue=None, injected=[],
                )

        self.assertFalse(restarting)
        self.assertEqual(result.text, "verified result")
        planner = internal_calls[0]
        self.assertEqual(planner[0], "Flexible planner")
        self.assertTrue(planner[2])
        self.assertIn("SKILL-CATALOG-DESCRIPTION", planner[1])
        self.assertLess(len(planner[1]), len(context))
        self.assertIn("current request", planner[1])
        self.assertNotIn(_BODY, planner[1])
        self.assertIn("If a skill-reading interface is available", planner[1])
        self.assertIn("read the selected instructions before choosing a detailed workflow", planner[1])
        self.assertIn("Without read tools, select skills from catalog metadata only", planner[1])
        self.assertIn("do not assume hidden instructions", planner[1])
        self.assertIn("assign that worker to load the full skill before choosing or executing", planner[1])
        for label, prompt, planning in internal_calls[1:]:
            with self.subTest(label=label):
                self.assertFalse(planning)
                self.assertNotIn("SKILL-CATALOG-DESCRIPTION", prompt)
                self.assertNotIn(_BODY, prompt)
        for backend, prompt, native_context, planning in worker_calls:
            with self.subTest(backend=backend):
                self.assertFalse(planning)
                self.assertIn("SKILL-CATALOG-DESCRIPTION", prompt)
                self.assertIn("SKILL-CATALOG-DESCRIPTION", native_context.sections["skills catalog"])
                self.assertNotIn("SKILL-CATALOG-DESCRIPTION", native_context.sections["policy and plugins"])
                self.assertNotIn(_BODY, native_context.render(None))

    async def test_skill_change_keeps_saved_flexible_cursor_and_does_not_replay_completed_worker(self) -> None:
        planner_calls = 0
        worker_calls = []
        interrupt = True

        async def internal(_backend, _ws, _prompt, _model, *, label, **_kwargs):
            nonlocal planner_calls
            if label == "Flexible planner":
                planner_calls += 1
                return "1. [low] inspect completed scope\n2. [high] finish remaining scope"
            return "verified merged result"

        async def drive(_backend, _ws, prompt, _model, _approval, **kwargs):
            nonlocal interrupt
            worker_calls.append((prompt, kwargs["native_context"].sections["skills catalog"]))
            if len(worker_calls) == 2 and interrupt:
                interrupt = False
                raise asyncio.CancelledError
            return AgentResult(text="completed proof", events=[ChatEvent("text", "completed proof")]), False

        with (mock.patch.object(agent, "run_internal_backend", side_effect=internal),
              mock.patch.object(agent, "_drive_backend", side_effect=drive)):
            with self.assertRaises(asyncio.CancelledError):
                async with skills.scope(self.ws), execution_state.execution_scope(self.ws, "same-job", "current request"):
                    await self._flexible()
            self._write_skill("UPDATED-PENDING-WORKER-SKILL-DESCRIPTION")
            async with skills.scope(self.ws), execution_state.execution_scope(self.ws, "same-job", "current request"):
                result, restarting = await self._flexible()

        self.assertFalse(restarting)
        self.assertEqual(result.text, "verified merged result")
        self.assertEqual(planner_calls, 1)
        self.assertEqual(len(worker_calls), 3)
        self.assertIn("Your task: inspect completed scope", worker_calls[0][0])
        self.assertIn("Your task: finish remaining scope", worker_calls[1][0])
        self.assertIn("Your task: finish remaining scope", worker_calls[2][0])
        self.assertIn("UPDATED-PENDING-WORKER-SKILL-DESCRIPTION", worker_calls[2][1])

    async def test_later_flexible_worker_refreshes_skills_changed_by_earlier_worker(self) -> None:
        worker_catalogs = []

        async def internal(_backend, _ws, _prompt, _model, *, label, **_kwargs):
            if label == "Flexible planner":
                return "1. [low] inspect first scope\n2. [high] finish remaining scope"
            return "verified result"

        async def drive(_backend, _ws, _prompt, _model, _approval, **kwargs):
            context = kwargs["native_context"]
            worker_catalogs.append(context.sections["skills catalog"])
            self.assertIn(skills.refresh_instructions(cli=True), context.render(None))
            if len(worker_catalogs) == 1:
                self._write_skill("CHANGED-DURING-EARLIER-WORKER")
            return AgentResult(text="completed proof", events=[ChatEvent("text", "completed proof")]), False

        with (mock.patch.object(agent, "run_internal_backend", side_effect=internal),
              mock.patch.object(agent, "_drive_backend", side_effect=drive)):
            async with skills.scope(self.ws), execution_state.execution_scope(self.ws, "live-skills", "current request"):
                _, restarting = await self._flexible()
        self.assertFalse(restarting)
        self.assertEqual(len(worker_catalogs), 2)
        self.assertNotIn("CHANGED-DURING-EARLIER-WORKER", worker_catalogs[0])
        self.assertIn("CHANGED-DURING-EARLIER-WORKER", worker_catalogs[1])

    async def test_unrelated_internal_roles_remain_stateless_without_skill_prompt_injection(self) -> None:
        launches = []

        async def launch(_ws, prompt, _model, **kwargs):
            launches.append((prompt, kwargs, agent_tools.skill_planning_active(), execution_state.current_run()))
            return SimpleNamespace()

        backend = SimpleNamespace(launch=launch, executable="offline")
        async with skills.scope(self.ws):
            with mock.patch.object(utils, "drain_llm_subprocess", new=mock.AsyncMock(return_value="offline result")):
                for label in ("Session router", "Session title", "Continue judge", "Flexible merge", "Compaction", "Colony"):
                    result = await utils.run_internal_backend(
                        backend, self.ws, f"{label}: existing input", "offline", label=label,
                        log=agent.logger, missing_executable_message="offline missing",
                    )
                    self.assertEqual(result, "offline result")

        for prompt, kwargs, planning, native_record in launches:
            self.assertNotIn("SKILL-CATALOG-DESCRIPTION", prompt)
            self.assertNotIn(_BODY, prompt)
            self.assertEqual(kwargs, {"approval": "deny", "compaction": True})
            self.assertFalse(planning)
            self.assertIsNone(native_record)


class NativeSkillRefreshTests(unittest.IsolatedAsyncioTestCase):
    _started = staticmethod(native_turns.NativeSessionTurnsTests._started)
    _answer = staticmethod(native_turns.NativeSessionTurnsTests._answer)
    _missing = staticmethod(native_turns.NativeSessionTurnsTests._missing)
    _conversation = native_turns.NativeSessionTurnsTests._conversation
    _turn = native_turns.NativeSessionTurnsTests._turn
    _drive = native_turns.NativeSessionTurnsTests._drive
    _create_process = native_turns.NativeSessionTurnsTests._create_process

    async def asyncSetUp(self):
        await native_turns.NativeSessionTurnsTests.asyncSetUp(self)
        self.enterContext(mock.patch.object(skills, "_native_roots", return_value=[]))
        self.skill_path = Path(self.ws, ".cozter", "skills", "checks.md")
        self.skill_path.parent.mkdir(parents=True)
        self.skill_path.write_text("# Checks\nCurrent check workflow.", encoding="utf-8")

    async def test_all_native_backends_reread_instruction_survives_exact_session_resume(self):
        for _module, backend, _model in native_turns.CASES:
            with self.subTest(backend=backend.name):
                sid = self._conversation()
                await self._turn(backend.name, "FIRST-SKILL-REQUEST", sid=sid)
                first = self.attempts[-1]
                await self._turn(backend.name, "NEXT-SKILL-REQUEST", sid=sid)
                second = self.attempts[-1]
                self.assertEqual(second["previous_id"], first["native_id"])
                self.assertIn(skills.refresh_instructions(cli=True), first["prompt"])
                self.assertIn(skills.refresh_instructions(cli=True), second["prompt"])
                self.assertNotIn("[Available workspace and shared native skills]", second["prompt"])
                self.assertNotIn("FIRST-SKILL-REQUEST", second["prompt"])
                self.assertNotIn("HISTORICAL-REQUEST-MARKER", second["prompt"])

    async def test_injected_user_request_refreshes_skills_without_replacing_native_session(self):
        sid = self._conversation()

        async def drive(*args, **kwargs):
            result, restarting = await self._drive(*args, **kwargs)
            if len(self.attempts) == 1:
                self.skill_path.write_text("# Checks\nUPDATED-WORKFLOW-BODY", encoding="utf-8")
                Path(self.skill_path.parent, "new.md").write_text("# NEW-INJECTED-SKILL", encoding="utf-8")
                kwargs["injected"].append("Continue with the updated skills.")
                return result, True
            return result, restarting

        with mock.patch.object(agent, "_drive_backend_impl", side_effect=drive):
            await self._turn("codex", "INJECT-SKILL-REQUEST", sid=sid)
        self.assertEqual(len(self.attempts), 2)
        self.assertEqual(self.attempts[1]["previous_id"], self.attempts[0]["native_id"])
        self.assertIn("NEW-INJECTED-SKILL", self.attempts[1]["prompt"])
        self.assertIn("[Updated skills catalog; replaces the previous version]", self.attempts[1]["prompt"])
        self.assertIn(skills.refresh_instructions(cli=True), self.attempts[1]["prompt"])
        self.assertIn("UPDATED-WORKFLOW-BODY", json.loads(skills.read_skill(self.ws, "Checks"))["content"])

    async def test_native_only_skill_refresh_instruction_is_present_with_empty_shared_catalog(self):
        self.skill_path.unlink()
        sid = self._conversation()
        await self._turn("claude_code", "NATIVE-ONLY-SKILL-REQUEST", sid=sid)
        self.assertNotIn("[Available workspace and shared native skills]", self.attempts[-1]["prompt"])
        self.assertIn("native CLI SKILL.md", self.attempts[-1]["prompt"])
        self.assertIn(skills.refresh_instructions(cli=True), self.attempts[-1]["prompt"])


if __name__ == "__main__":
    unittest.main()
