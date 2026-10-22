"""Owned native hang recovery preserves exact sessions and Flexible work."""

import asyncio
from contextlib import nullcontext
import logging
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent, execution_state, supervision, utils
from Cozter.backends_agent import codex
from Cozter.backends_agent.base import SHELL_INPUT_RULE
from Cozter.tests import test_native_delta_sync as delta
from Cozter.tests import test_native_session_turns as turns


class NativeHangRecoveryTests(unittest.IsolatedAsyncioTestCase):
    _started = staticmethod(turns.NativeSessionTurnsTests._started)
    _answer = staticmethod(turns.NativeSessionTurnsTests._answer)
    _missing = staticmethod(turns.NativeSessionTurnsTests._missing)
    _activity = staticmethod(delta.NativeDeltaSyncTests._activity)
    _disk_values = staticmethod(delta.NativeDeltaSyncTests._disk_values)
    _conversation = turns.NativeSessionTurnsTests._conversation
    _turn = turns.NativeSessionTurnsTests._turn
    _drive = turns.NativeSessionTurnsTests._drive
    _create_process = delta.NativeDeltaSyncTests._create_process
    _events = delta.NativeDeltaSyncTests._events
    _kill = delta.NativeDeltaSyncTests._kill
    _internal = delta.NativeDeltaSyncTests._internal

    async def asyncSetUp(self):
        await delta.NativeDeltaSyncTests.asyncSetUp(self)
        self.enterContext(mock.patch.object(agent.skills, "catalog_prompt", return_value="OFFLINE-SKILLS-CATALOG"))
        self.operations = self.enterContext(mock.patch.object(
            supervision, "new_operation", side_effect=lambda _ws, kind, details="":
            SimpleNamespace(kind=kind, details=details),
        ))
        self.enterContext(mock.patch.object(supervision, "operation_scope", side_effect=lambda _ticket: nullcontext()))
        self.enterContext(mock.patch.object(supervision, "attach_process"))
        self.enterContext(mock.patch.object(supervision, "note_output"))

    async def _wait(self, task, _ticket):
        await asyncio.wait((task,))
        return None

    async def asyncTearDown(self):
        await agent._wait_for_pending_session_writes(self.ws)

    async def test_all_native_backends_recover_in_same_job_without_history_replay(self):
        for module, cls, _model in turns._CASES:
            with self.subTest(backend=cls.name):
                sid = self._conversation()
                self.mode = "hold-after-activity"
                self.receipt_ready = asyncio.Event()
                calls = 0

                async def diagnose(task, ticket):
                    nonlocal calls
                    calls += 1
                    if calls != 1:
                        return await self._wait(task, ticket)
                    await self.receipt_ready.wait()
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    self.assertIsNotNone(self.processes[-1].returncode)
                    self.mode = "complete"
                    return "Test fixture found an unchanged test output cursor; inspect and repair its barrier."

                start = len(self.launches)
                with mock.patch.object(supervision, "wait_owned", side_effect=diagnose):
                    result = await self._turn(cls.name, "PRIVATE-CURRENT-REQUEST", sid=sid)
                first, recovered = self.launches[start:]
                self.assertEqual(result.text, "COMPLETED-NATIVE-ANSWER")
                self.assertIsNone(result.error)
                self.assertEqual(first["native_id"], recovered["native_id"])
                self.assertIs(first["record"], recovered["record"])
                self.assertEqual(first["home"], recovered["home"])
                flag = "resume" if module is codex else "--resume"
                self.assertEqual(recovered["command"][recovered["command"].index(flag) + 1], first["native_id"])
                self.assertIn("Automatic hang recovery", recovered["prompt"])
                self.assertIn("Continue the current job from this native conversation", recovered["prompt"])
                for marker in ("PRIVATE-CURRENT-REQUEST", "HISTORICAL-REQUEST-MARKER",
                               "HISTORICAL-ANSWER-MARKER", "POLICY-PRELUDE-MARKER"):
                    self.assertNotIn(marker, recovered["prompt"])
                for call in self.operations.call_args_list:
                    self.assertNotIn("PRIVATE-CURRENT-REQUEST", call.kwargs["details"])

    async def test_flexible_recovery_keeps_completed_worker_and_plan_cursor(self):
        self._conversation()
        calls = 0

        async def diagnose(task, ticket):
            nonlocal calls
            calls += 1
            if calls != 2:
                return await self._wait(task, ticket)
            self.mode = "hold-after-activity"
            self.receipt_ready.clear()
            await self.receipt_ready.wait()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self.mode = "complete"
            return "Second worker test stalled; inspect existing evidence and repair it."

        with mock.patch.object(supervision, "wait_owned", side_effect=diagnose):
            result = await self._turn("flexible", "Finish both assigned tasks.")
        self.assertEqual(result.text, "UNSEEN-MERGED-ANSWER-1")
        self.assertEqual(len(self.launches), 3)
        self.assertIn("[Sub-task 1/2:", self.launches[0]["prompt"])
        self.assertIn("[Sub-task 2/2:", self.launches[1]["prompt"])
        self.assertIs(self.launches[1]["record"], self.launches[2]["record"])
        self.assertEqual(self.launches[1]["native_id"], self.launches[2]["native_id"])
        for launch in self.launches:
            self.assertIn(SHELL_INPUT_RULE, launch["prompt"])
        planner_calls = [call for call in agent.run_internal_backend.call_args_list
                         if call.kwargs["label"] == "Flexible planner"]
        self.assertEqual(len(planner_calls), 1)

    async def test_repeated_hangs_back_off_automatically_without_changing_native_id(self):
        sid = self._conversation()
        self.mode = "hold-after-activity"

        async def diagnose(task, _ticket):
            self.receipt_ready.clear()
            await self.receipt_ready.wait()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return "The same test still has no useful progress."

        with mock.patch.object(supervision, "wait_owned", side_effect=diagnose):
            result = await agent.run(
                "Finish the requested work", self.ws, 7, model=self.models["codex"], approval="full",
                backend_name="codex", summary_backend_name="codex", summary_model="offline-summary",
                session_id=sid, execution_id="same-durable-job",
            )
        self.assertEqual(len(self.launches), 3)
        self.assertEqual(len({launch["native_id"] for launch in self.launches}), 1)
        self.assertEqual(result.continuation_reason, "prerequisite")
        self.assertIn("diagnose and repair the cause", result.continue_instruction)
        self.assertTrue(result.error)
        async with execution_state.execution_scope(self.ws, "same-durable-job", "Finish the requested work"):
            async with execution_state.backend_scope("codex", self.models["codex"], "full", 0, native=True) as record:
                self.assertEqual(record.get("native_record_id"), self.launches[-1]["record"].get("native_record_id"))

    async def test_external_cancel_reaps_native_worker_and_does_not_retry(self):
        sid = self._conversation()
        self.mode = "hold-after-activity"
        with mock.patch.object(supervision, "wait_owned", side_effect=self._wait):
            task = asyncio.create_task(agent.run(
                "Finish the requested work", self.ws, 7, model=self.models["codex"], approval="full",
                backend_name="codex", summary_backend_name="codex", summary_model="offline-summary",
                session_id=sid, execution_id="cancel-durable-job",
            ))
            await self.receipt_ready.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(len(self.launches), 1)
        self.assertIsNotNone(self.processes[0].returncode)
        self.assertTrue(self._disk_values(self.launches[0]["record"]).get("native_prompt_ledger"))

    async def test_repeated_cancel_waits_for_owned_native_cleanup(self):
        sid = self._conversation()
        self.mode = "hold-after-activity"
        cleanup_entered = asyncio.Event()
        release_cleanup = asyncio.Event()

        async def cleanup(process):
            cleanup_entered.set()
            await release_cleanup.wait()
            process.returncode = -9

        with (
            mock.patch.object(supervision, "wait_owned", side_effect=self._wait),
            mock.patch.object(agent, "kill_and_wait", side_effect=cleanup),
        ):
            task = asyncio.create_task(agent.run(
                "Keep repairing", self.ws, 7, model=self.models["codex"], approval="full",
                backend_name="codex", summary_backend_name="codex", summary_model="offline-summary",
                session_id=sid, execution_id="repeated-cancel-job",
            ))
            await self.receipt_ready.wait()
            task.cancel()
            await cleanup_entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            release_cleanup.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(len(self.launches), 1)
        self.assertIsNotNone(self.processes[0].returncode)

    async def test_recovery_of_one_parallel_reader_does_not_cancel_healthy_sibling(self):
        self._conversation()
        self.plan = ("[PLAN]\n1. [low] [parallel-read] inspect A\n"
                     "2. [mid] [parallel-read] inspect B\n[/PLAN]")
        backend = self.backends["codex"]
        both_started = asyncio.Event()
        release_healthy = asyncio.Event()
        started = set()
        role_by_record = {}
        attempts = {}
        healthy_task = None
        calls = 0

        async def events(process, **_kwargs):
            launch = process.fixture_launch
            record = launch["record"]
            key = record.key
            if key not in role_by_record:
                role_by_record[key] = 1 if "[Sub-task 1/2:" in launch["prompt"] else 2
            role = role_by_record[key]
            attempts[key] = attempts.get(key, 0) + 1
            yield self._started(codex, launch["native_id"])
            yield self._activity(codex)
            started.add(role)
            if len(started) == 2:
                both_started.set()
            if role == 1 and attempts[key] == 1:
                await asyncio.Event().wait()
            if role == 2:
                await release_healthy.wait()
            yield self._answer(codex, f"Verified reader {role}")
            process.returncode = 0

        async def diagnose(task, ticket):
            nonlocal calls, healthy_task
            calls += 1
            index = calls
            if index == 2:
                healthy_task = task
            if index != 1:
                return await self._wait(task, ticket)
            await both_started.wait()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self.assertIsNotNone(healthy_task)
            self.assertFalse(healthy_task.done())
            release_healthy.set()
            return "Only reader A stalled; repair its read operation."

        with (
            mock.patch.object(backend, "supports_read_only_workers", True),
            mock.patch.object(agent, "iter_process_json_events", side_effect=events),
            mock.patch.object(supervision, "wait_owned", side_effect=diagnose),
        ):
            result = await self._turn("flexible", "Inspect both independent targets")
        self.assertEqual(result.text, "UNSEEN-MERGED-ANSWER-1")
        self.assertEqual(sorted(attempts.values()), [1, 2])
        self.assertEqual(len(self.launches), 3)

    async def test_flexible_recurring_hang_keeps_cursor_and_resumes_remaining_worker(self):
        sid = self._conversation()
        calls = 0

        async def diagnose(task, ticket):
            nonlocal calls
            calls += 1
            if calls == 1 or calls > 4:
                self.mode = "complete"
                return await self._wait(task, ticket)
            self.mode = "hold-after-activity"
            self.receipt_ready.clear()
            await self.receipt_ready.wait()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return "Repair remaining worker's repeated test hang."

        async def run_job():
            return await agent.run(
                "Finish both assigned tasks", self.ws, 7, model="unused-flexible-model", approval="full",
                backend_name="flexible", summary_backend_name="codex", summary_model="offline-summary",
                session_id=sid, execution_id="flexible-recurring-job",
            )

        with mock.patch.object(supervision, "wait_owned", side_effect=diagnose):
            partial = await run_job()
            self.assertEqual(partial.continuation_reason, "prerequisite")
            self.assertEqual(self.merges, 0)
            self.assertEqual(len(self.launches), 4)
            completed = self.launches[0]["record"]
            async with execution_state.execution_scope(self.ws, "flexible-recurring-job", "Finish both assigned tasks"):
                plan = await execution_state.record("flexible_plan")
                self.assertTrue(plan.get("active"))
                self.assertEqual(plan.get("cursor"), 1)
            result = await run_job()
        self.assertEqual(result.text, "UNSEEN-MERGED-ANSWER-1")
        self.assertEqual(len(self.launches), 5)
        self.assertIsNot(self.launches[-1]["record"], completed)
        planners = [call for call in agent.run_internal_backend.call_args_list
                    if call.kwargs["label"] == "Flexible planner"]
        self.assertEqual(len(planners), 1)

    def test_raw_native_tool_starts_are_visible_without_counting_heartbeats(self):
        proc = SimpleNamespace(pid=123)
        agent._note_native_progress(proc, {"type": "item.started", "item": {
            "type": "command_execution", "command": "fixture-tests", "id": "cmd-1",
        }})
        supervision.attach_process.assert_called_with(proc, command="fixture-tests")
        supervision.note_output.assert_not_called()
        agent._note_native_progress(proc, {"type": "turn.started"})
        agent._note_native_progress(proc, {"type": "heartbeat"})
        agent._note_native_progress(proc, {"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": "Still thinking"},
        ]}})
        supervision.note_output.assert_not_called()
        agent._note_native_progress(proc, {"type": "item.completed", "item": {
            "type": "command_execution", "command": "fixture-tests", "id": "cmd-1", "aggregated_output": "OK",
        }})
        self.assertIn(mock.call("OK"), supervision.note_output.call_args_list)
        count = supervision.note_output.call_count
        for _ in range(3):
            agent._note_native_progress(proc, {"type": "item.updated", "item": {
                "type": "command_execution", "command": "fixture-tests", "id": "cmd-1", "aggregated_output": "OK",
            }})
        self.assertEqual(supervision.note_output.call_count, count)
        agent._note_native_progress(proc, {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "x", "content": "verified"},
        ]}})
        supervision.note_output.assert_called_with("verified")


class InternalHangRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_internal_worker_settles_before_single_retry_and_retains_deny_policy(self):
        entered = asyncio.Event()
        settled = asyncio.Event()
        calls = 0
        backend = SimpleNamespace(name="offline-summary", executable="offline", launch=mock.AsyncMock())
        proc = SimpleNamespace(pid=-1)
        backend.launch.return_value = proc

        async def drain(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                try:
                    entered.set()
                    await asyncio.Event().wait()
                finally:
                    settled.set()
            self.assertTrue(settled.is_set())
            return "VERIFIED-SUMMARY"

        async def diagnose(task, _ticket):
            if calls == 0:
                await entered.wait()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                return "Summary stream stalled."
            await asyncio.wait((task,))
            return None

        with (
            mock.patch.object(supervision, "new_operation", return_value=object()),
            mock.patch.object(supervision, "operation_scope", side_effect=lambda _ticket: nullcontext()),
            mock.patch.object(supervision, "attach_process"),
            mock.patch.object(supervision, "wait_owned", side_effect=diagnose),
            mock.patch.object(utils, "drain_llm_subprocess", side_effect=drain),
        ):
            result = await utils.run_internal_backend(
                backend, "fixture", "PRIVATE-INTERNAL-PROMPT", "fixture-model", label="Continue judge",
                log=logging.getLogger(__name__), missing_executable_message="missing %s",
            )
        self.assertEqual(result, "VERIFIED-SUMMARY")
        self.assertEqual(backend.launch.await_count, 2)
        for call in backend.launch.await_args_list:
            self.assertEqual(call.kwargs["approval"], "deny")
            self.assertTrue(call.kwargs["compaction"])


if __name__ == "__main__":
    unittest.main()
