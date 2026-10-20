"""A leased native helper keeps its lane while its parent resumes in place."""

import unittest
import uuid

from Cozter import execution_state
from Cozter.backends_agent import codex, copilot
from Cozter.backends_agent.base import AgentResult
from Cozter.tests import test_flexible_native_sessions as native_flexible


def lease(question):
    return (
        "Parent retains the repair and verification obligation.\n"
        "[LEASE]\n"
        "tier: low\n"
        "reason: An independent bounded source inspection can answer this question.\n"
        f"task: Inspect existing source only and report evidence for {question}.\n"
        "[/LEASE]"
    )


class FlexibleLeaseNativeContinuityTests(unittest.IsolatedAsyncioTestCase):
    _started = staticmethod(native_flexible.FlexibleNativeSessionsTests._started)
    _answer = staticmethod(native_flexible.FlexibleNativeSessionsTests._answer)
    _missing = staticmethod(native_flexible.FlexibleNativeSessionsTests._missing)
    _turn = native_flexible.FlexibleNativeSessionsTests._turn
    _conversation = native_flexible.FlexibleNativeSessionsTests._conversation
    _select_workers = native_flexible.FlexibleNativeSessionsTests._select_workers
    _create_process = native_flexible.FlexibleNativeSessionsTests._create_process
    _internal = native_flexible.FlexibleNativeSessionsTests._internal

    async def asyncSetUp(self):
        await native_flexible.FlexibleNativeSessionsTests.asyncSetUp(self)
        self.plan = "[PLAN]\n1. [high] Repair the cross-thread cache ordering and verify it.\n[/PLAN]"
        self.parent_calls = 0
        self.helper_calls = 0

    async def _drive(self, backend, workspace_path, prompt, model, approval, *, effort, **_kwargs):
        module = self.modules[backend.name]
        record = execution_state.current_run()
        self.assertIsNotNone(record)
        previous_id = record.get("native_session_id")
        scope = execution_state._WORKER.get()
        is_helper = scope.startswith("flexible:lease:")
        attempt = {
            "backend": backend.name, "prompt": prompt, "previous_id": previous_id,
            "record_key": record.key, "directory": str(record.directory),
            "approval": approval, "worker_scope": scope,
            "helper": is_helper,
        }
        self.attempts.append(attempt)
        process = await backend.launch(workspace_path, prompt, model, approval, effort=effort)
        result = AgentResult()
        try:
            native_id = previous_id or record.get("native_candidate_session_id") or str(uuid.uuid4())
            if module is copilot:
                (record.directory / "copilot-home" / "session-state" / native_id).mkdir(parents=True, exist_ok=True)
            backend.parse_event(self._started(module, native_id), result)
            attempt["native_id"] = native_id
            attempt["home"] = process.fixture_launch["home"]
            if is_helper:
                self.helper_calls += 1
                question = "QUESTION-A-MARKER" if self.helper_calls == 1 else "QUESTION-B-MARKER"
                self.assertIn(question, prompt)
                answer = f"HELPER-ANSWER-{'A' if self.helper_calls == 1 else 'B'}-MARKER: source evidence verified."
            else:
                self.parent_calls += 1
                if self.parent_calls <= 2:
                    answer = lease("QUESTION-A-MARKER" if self.parent_calls == 1 else "QUESTION-B-MARKER")
                else:
                    answer = "Parent repair and regression verification are complete."
            backend.parse_event(self._answer(module, answer), result)
            self.assertEqual(result.text, answer)
            return result, False
        finally:
            await backend.cleanup_process(process)

    async def test_all_four_keep_parent_and_helper_uuid_with_only_unseen_context(self):
        history = (
            "HISTORICAL-REQUEST-MARKER", "HISTORICAL-ANSWER-MARKER",
            "SUMMARY-CONTEXT-MARKER", "MEMORY-CONTEXT-MARKER", "COLONY-CONTEXT-MARKER",
        )
        cases = ((case, approval) for case in native_flexible._CASES for approval in ("full", "confirm"))
        for (module, cls, _model), approval in cases:
            with self.subTest(backend=cls.name, approval=approval):
                self._select_workers(cls.name)
                self.parent_calls = self.helper_calls = 0
                sid = self._conversation()
                start = len(self.attempts)
                launch_start = len(self.launches)
                result = await self._turn("flexible", "PARENT-REQUEST-MARKER", sid=sid, approval=approval)
                attempts = self.attempts[start:]
                self.assertEqual([attempt["helper"] for attempt in attempts], [False, True, False, True, False])
                parent, helper, resumed, helper_again, finished = attempts
                self.assertEqual(self.parent_calls, 3)
                self.assertEqual(self.helper_calls, 2)
                self.assertFalse(result.error)
                self.assertEqual(result.session_id, sid)
                self.assertNotEqual(parent["native_id"], helper["native_id"])
                for later in (resumed, finished):
                    self.assertEqual(later["previous_id"], parent["native_id"])
                    self.assertEqual(later["record_key"], parent["record_key"])
                    self.assertEqual(later["directory"], parent["directory"])
                    self.assertEqual(later["worker_scope"], parent["worker_scope"])
                    self.assertEqual(later["approval"], approval)
                self.assertEqual(helper_again["previous_id"], helper["native_id"])
                self.assertEqual(helper_again["record_key"], helper["record_key"])
                self.assertEqual(helper_again["directory"], helper["directory"])
                self.assertNotEqual(helper_again["worker_scope"], helper["worker_scope"])
                self.assertEqual(helper_again["approval"], "confirm")
                self.assertNotEqual(helper["directory"], parent["directory"])
                for marker in history:
                    self.assertIn(marker, parent["prompt"])
                    self.assertIn(marker, helper["prompt"])
                    for later in (resumed, helper_again, finished):
                        self.assertNotIn(marker, later["prompt"])
                self.assertIn("HELPER-ANSWER-A-MARKER", resumed["prompt"])
                self.assertIn("HELPER-ANSWER-B-MARKER", finished["prompt"])
                self.assertNotIn("HELPER-ANSWER-A-MARKER", finished["prompt"])
                self.assertNotIn("QUESTION-A-MARKER", helper_again["prompt"])
                self.assertNotIn("PARENT-REQUEST-MARKER", resumed["prompt"])
                self.assertNotIn("PARENT-REQUEST-MARKER", finished["prompt"])
                launches = self.launches[launch_start:]
                self.assertEqual(len(launches), 5)
                flag = "resume" if module is codex else "--resume"
                for launch, expected in zip(launches[2:], (parent, helper, parent)):
                    command = launch["command"]
                    self.assertEqual(command[command.index(flag) + 1], expected["native_id"])
                if module is copilot:
                    self.assertEqual(parent["home"], resumed["home"])
                    self.assertEqual(parent["home"], finished["home"])
                    self.assertEqual(helper["home"], helper_again["home"])
                    self.assertNotEqual(parent["home"], helper["home"])


if __name__ == "__main__":
    unittest.main()
