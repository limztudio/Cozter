"""Offline provider launches share bounded Cozter history across Flexible roles."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid

from Cozter import agent, execution_state, flexible, session
from Cozter.backends_agent import codex, copilot
from Cozter.backends_agent.base import AgentResult
from Cozter.tests import test_native_session_turns as native_turns


_CASES = native_turns._CASES
_SERIAL_PLAN = "[PLAN]\n1. [low] Inspect the requested target.\n2. [high] Repair and verify it.\n[/PLAN]"


class FlexibleNativeSessionsTests(unittest.IsolatedAsyncioTestCase):
    _started = staticmethod(native_turns.NativeSessionTurnsTests._started)
    _answer = staticmethod(native_turns.NativeSessionTurnsTests._answer)
    _missing = staticmethod(native_turns.NativeSessionTurnsTests._missing)
    _turn = native_turns.NativeSessionTurnsTests._turn

    async def asyncSetUp(self):
        # Reuse the direct-turn fixture's offline boundaries, not its tests.
        # Real agent.run, native record selection, launch argv and events stay active.
        await native_turns.NativeSessionTurnsTests.asyncSetUp(self)
        self.backends[flexible.BACKEND_NAME] = SimpleNamespace(name=flexible.BACKEND_NAME)
        self.models[flexible.BACKEND_NAME] = "unused-flexible-model"
        self.tiers = {tier: ("codex", self.models["codex"]) for tier in flexible.TIERS}
        self.plan = _SERIAL_PLAN
        self.planner_prompts = []
        self.merge_prompts = []
        self.parallel_barrier = None
        self.parallel_started = 0
        self.enterContext(mock.patch.object(
            agent.workspace_mod, "get_flexible_run_config", side_effect=lambda _ws: self.tiers,
        ))
        self.enterContext(mock.patch.object(
            agent.config, "get_flexible_max_parallel_workers", return_value=3,
        ))
        self.enterContext(mock.patch.object(agent, "run_internal_backend", side_effect=self._internal))
        self.enterContext(mock.patch.object(agent.colony, "get_items", return_value=["COLONY-CONTEXT-MARKER"]))

    def _conversation(self):
        sid = native_turns.NativeSessionTurnsTests._conversation(self)
        data = session.load_session(self.ws, sid)
        data["summary"] = "SUMMARY-CONTEXT-MARKER"
        data["long_term"] = ["MEMORY-CONTEXT-MARKER"]
        session.save_session(self.ws, sid, data)
        return sid

    def _select_workers(self, name):
        self.tiers = {tier: (name, self.models[name]) for tier in flexible.TIERS}

    async def _internal(self, _backend, _ws, prompt, _model, *, label, **_kwargs):
        self.assertIsNone(execution_state.current_run())
        if label == "Flexible planner":
            self.planner_prompts.append(prompt)
            return self.plan
        self.assertEqual(label, "Flexible merge")
        self.merge_prompts.append(prompt)
        return f"MERGED-ANSWER-MARKER-{len(self.merge_prompts)}"

    async def _create_process(self, command, *args, **kwargs):
        command = list(command)
        prompt = (Path(command[command.index("--prompt-file") + 1]).read_text(encoding="utf-8")
                  if command[0] == "grok" else args[0])
        launch = {"backend": command[0], "command": command, "prompt": prompt,
                  "home": kwargs.get("env", {}).get("COPILOT_HOME")}
        self.launches.append(launch)
        process = mock.Mock()
        process.fixture_launch = launch
        return process

    async def _drive(self, backend, workspace_path, prompt, model, approval, *, effort, **_kwargs):
        module = self.modules[backend.name]
        record = execution_state.current_run()
        self.assertIsNotNone(record)
        previous_id = record.get("native_session_id")
        attempt = {"backend": backend.name, "prompt": prompt, "previous_id": previous_id,
                   "record_key": record.key, "directory": str(record.directory), "approval": approval}
        self.attempts.append(attempt)
        number = len(self.attempts)
        process = await backend.launch(workspace_path, prompt, model, approval, effort=effort)
        result = AgentResult()
        try:
            if previous_id and self.missing_provider:
                self.missing_provider = False
                backend.parse_event(self._missing(module, previous_id), result)
                return result, False
            native_id = previous_id or record.get("native_candidate_session_id") or str(uuid.uuid4())
            if module is copilot:
                (record.directory / "copilot-home" / "session-state" / native_id).mkdir(parents=True, exist_ok=True)
            backend.parse_event(self._started(module, native_id), result)
            attempt["native_id"] = native_id
            attempt["home"] = process.fixture_launch["home"]
            if self.parallel_barrier is not None and "Inspection only:" in prompt:
                self.parallel_started += 1
                if self.parallel_started == 2:
                    self.parallel_barrier.set()
                await asyncio.wait_for(self.parallel_barrier.wait(), 5)
            answer = f"WORKER-ANSWER-MARKER-{number}"
            backend.parse_event(self._answer(module, answer), result)
            self.assertEqual(result.text, answer)
            return result, False
        finally:
            await backend.cleanup_process(process)

    async def test_all_four_reuse_native_writer_across_roles_and_completed_flexible_jobs(self):
        for module, cls, _model in _CASES:
            with self.subTest(backend=cls.name):
                self._select_workers(cls.name)
                self.plan = _SERIAL_PLAN
                sid = self._conversation()
                start = len(self.attempts)
                first = await self._turn("flexible", "FIRST-FLEXIBLE-REQUEST", sid=sid)
                first_attempt, second_attempt = self.attempts[start:]
                native_id = first_attempt["native_id"]
                self.assertEqual(second_attempt["previous_id"], native_id)
                for marker in ("HISTORICAL-REQUEST-MARKER", "HISTORICAL-ANSWER-MARKER",
                               "SUMMARY-CONTEXT-MARKER", "MEMORY-CONTEXT-MARKER", "COLONY-CONTEXT-MARKER"):
                    self.assertIn(marker, first_attempt["prompt"])
                    self.assertNotIn(marker, second_attempt["prompt"])
                self.assertIn(first.text, session.load_session(self.ws, sid)["messages"][-1]["content"])
                self.plan = "[PLAN]\n1. [mid] Handle a different follow-up role and verify.\n[/PLAN]"
                second = await self._turn("flexible", "SECOND-FLEXIBLE-REQUEST", sid=sid)
                returned = self.attempts[-1]
                self.assertEqual(second.session_id, sid)
                self.assertEqual(returned["previous_id"], native_id)
                self.assertEqual(returned["prompt"].count(first.text), 1)
                self.assertIn("SECOND-FLEXIBLE-REQUEST", returned["prompt"])
                self.assertNotIn("HISTORICAL-REQUEST-MARKER", returned["prompt"])
                self.assertNotIn("SUMMARY-CONTEXT-MARKER", returned["prompt"])
                flag = "resume" if module is codex else "--resume"
                command = next(launch["command"] for launch in reversed(self.launches)
                               if launch["backend"] == cls.name)
                self.assertEqual(command[command.index(flag) + 1], native_id)
                if module is copilot:
                    self.assertEqual(first_attempt["home"], returned["home"])

    async def test_direct_flexible_and_backend_switches_deliver_merged_context_once(self):
        sid = self._conversation()
        await self._turn("codex", "DIRECT-INITIAL-REQUEST", sid=sid)
        native_id = self.attempts[-1]["native_id"]
        merged = await self._turn("flexible", "FLEXIBLE-MERGED-REQUEST", sid=sid)
        self.assertTrue(all(attempt["previous_id"] == native_id for attempt in self.attempts[1:]))
        claude = await self._turn("claude_code", "OTHER-BACKEND-REQUEST", sid=sid)
        self.plan = "[PLAN]\n1. [high] Continue the shared conversation.\n[/PLAN]"
        await self._turn("flexible", "FLEXIBLE-RETURN-REQUEST", sid=sid)
        returned = self.attempts[-1]
        self.assertEqual(returned["previous_id"], native_id)
        for marker in (merged.text, "OTHER-BACKEND-REQUEST", claude.text):
            self.assertEqual(returned["prompt"].count(marker), 1)
            self.assertIn(marker, self.planner_prompts[-1])
        self.assertNotIn("HISTORICAL-REQUEST-MARKER", returned["prompt"])
        self.assertNotIn("WORKER-ANSWER-MARKER-2", self.planner_prompts[-1])
        await self._turn("flexible", "NEXT-FLEXIBLE-REQUEST", sid=sid)
        later = self.attempts[-1]
        for marker in (merged.text, "OTHER-BACKEND-REQUEST", claude.text):
            self.assertNotIn(marker, later["prompt"])
        last_flexible_answer = later["native_id"], f"WORKER-ANSWER-MARKER-{len(self.attempts)}"
        await self._turn("codex", "DIRECT-RETURN-REQUEST", sid=sid)
        direct = self.attempts[-1]
        self.assertEqual(direct["previous_id"], last_flexible_answer[0])
        self.assertNotIn(last_flexible_answer[1], direct["prompt"])
        self.assertNotIn("HISTORICAL-REQUEST-MARKER", direct["prompt"])

    async def test_parallel_readers_keep_distinct_native_histories_and_writer_barrier(self):
        sid = self._conversation()
        self.plan = ("[PLAN]\n1. [low] [parallel-read] Inspect target A.\n"
                     "2. [low] [parallel-read] Inspect target B.\n"
                     "3. [high] Repair and verify the combined findings.\n[/PLAN]")
        self.parallel_barrier = asyncio.Event()
        await self._turn("flexible", "PARALLEL-REQUEST", sid=sid)
        first, second, writer = self.attempts
        self.assertEqual(self.parallel_started, 2)
        self.assertNotEqual(first["native_id"], second["native_id"])
        self.assertNotEqual(first["directory"], second["directory"])
        self.assertNotIn(writer["native_id"], {first["native_id"], second["native_id"]})
        for attempt in (first, second):
            self.assertEqual(attempt["approval"], "confirm")
            self.assertIn("HISTORICAL-REQUEST-MARKER", attempt["prompt"])
            self.assertIn("SUMMARY-CONTEXT-MARKER", attempt["prompt"])
            self.assertNotIn("WORKER-ANSWER-MARKER", attempt["prompt"])
        self.assertIn("WORKER-ANSWER-MARKER-1", writer["prompt"])
        self.assertIn("WORKER-ANSWER-MARKER-2", writer["prompt"])
        self.parallel_started = 0
        self.parallel_barrier = asyncio.Event()
        await self._turn("flexible", "NEXT-PARALLEL-REQUEST", sid=sid)
        # Readers may start in either order, so match each one by its lane.
        lanes = {attempt["directory"]: attempt for attempt in (first, second, writer)}
        later_attempts = self.attempts[3:]
        self.assertEqual(len(later_attempts), 3)
        self.assertEqual(later_attempts[-1]["directory"], writer["directory"])
        self.assertEqual({attempt["directory"] for attempt in later_attempts}, set(lanes))
        for later in later_attempts:
            self.assertEqual(later["previous_id"], lanes[later["directory"]]["native_id"])
            self.assertNotIn("HISTORICAL-REQUEST-MARKER", later["prompt"])
            self.assertEqual(later["prompt"].count("MERGED-ANSWER-MARKER-1"), 1)

    async def test_confirmed_missing_writer_restores_shared_context_for_all_four(self):
        for _module, cls, _model in _CASES:
            with self.subTest(backend=cls.name):
                self._select_workers(cls.name)
                self.plan = _SERIAL_PLAN
                sid = self._conversation()
                merged = await self._turn("flexible", "BEFORE-MISSING-REQUEST", sid=sid)
                native_id = self.attempts[-1]["native_id"]
                self.plan = "[PLAN]\n1. [mid] Finish recovery and verify.\n[/PLAN]"
                self.missing_provider = True
                start = len(self.attempts)
                await self._turn("flexible", "AFTER-MISSING-REQUEST", sid=sid)
                attempts = self.attempts[start:]
                self.assertEqual(len(attempts), 2)
                self.assertEqual(attempts[0]["previous_id"], native_id)
                self.assertNotIn("HISTORICAL-REQUEST-MARKER", attempts[0]["prompt"])
                self.assertIsNone(attempts[1]["previous_id"])
                self.assertNotEqual(attempts[1]["native_id"], native_id)
                for marker in ("HISTORICAL-REQUEST-MARKER", "HISTORICAL-ANSWER-MARKER",
                               "SUMMARY-CONTEXT-MARKER", "MEMORY-CONTEXT-MARKER",
                               "COLONY-CONTEXT-MARKER", merged.text, "AFTER-MISSING-REQUEST"):
                    self.assertIn(marker, attempts[1]["prompt"])


if __name__ == "__main__":
    unittest.main()
