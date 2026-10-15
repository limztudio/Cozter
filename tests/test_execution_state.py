"""Job continuation survives cancellation without crossing execution policy."""

import asyncio
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent, agent_tools, execution_state, utils
from Cozter.backends_agent.base import AgentResult, ChatEvent
from Cozter.backends_agent.codex import CodexBackend


class ExecutionStateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = self.temp.name

    async def test_persists_across_attempts_and_context_is_cleared(self):
        async with execution_state.execution_scope(self.workspace, "queue-a", "original"):
            async with execution_state.backend_scope("codex", "chosen", "full", 100):
                record = execution_state.current_run()
                record.set(native_session_id="exact-session", nested={"steps": [1, 2]})
                copy = record.get("nested")
                copy["steps"].append(3)
                self.assertEqual(record.get("nested"), {"steps": [1, 2]})
        self.assertIsNone(execution_state.current_run())
        self.assertIsNone(execution_state.current_execution())
        async with execution_state.execution_scope(self.workspace, "queue-a", "original"):
            async with execution_state.backend_scope("codex", "chosen", "full", 100):
                self.assertEqual(execution_state.current_run().get("native_session_id"),
                                 "exact-session")

    async def test_isolates_job_backend_model_permission_effort_and_worker(self):
        async with execution_state.execution_scope(self.workspace, "job", "request"):
            async with execution_state.backend_scope("codex", "chosen", "full", 100):
                await execution_state.current_run().save(secret="owned")
            variants = [("claude_code", "chosen", "full", 100),
                        ("codex", "other", "full", 100),
                        ("codex", "chosen", "confirm", 100),
                        ("codex", "chosen", "full", 50)]
            for settings in variants:
                async with execution_state.backend_scope(*settings):
                    self.assertIsNone(execution_state.current_run().get("secret"))
            with execution_state.worker_scope("another worker"):
                async with execution_state.backend_scope("codex", "chosen", "full", 100):
                    self.assertIsNone(execution_state.current_run().get("secret"))
        for job, request in (("other", "request"),):
            async with execution_state.execution_scope(self.workspace, job, request):
                async with execution_state.backend_scope("codex", "chosen", "full", 100):
                    self.assertIsNone(execution_state.current_run().get("secret"))
        async with execution_state.execution_scope(self.workspace, "job", "request plus guidance"):
            async with execution_state.backend_scope("codex", "chosen", "full", 100):
                self.assertEqual(execution_state.current_run().get("secret"), "owned")

    async def test_isolates_workspaces_and_calls_without_job_id(self):
        second = Path(self.workspace, "second")
        second.mkdir()
        async with execution_state.execution_scope(self.workspace, None, "request"):
            record = await execution_state.record("plan")
            await record.save(done=True)
            temporary_path = record.path
        self.assertFalse(temporary_path.exists())
        for ws in (self.workspace, str(second)):
            async with execution_state.execution_scope(ws, None, "request"):
                self.assertIsNone((await execution_state.record("plan")).get("done"))

    async def test_corrupt_or_foreign_record_does_not_silently_restart_tools(self):
        async with execution_state.execution_scope(self.workspace, "job", "request"):
            record = await execution_state.record("plan")
            await record.save(done=True)
            path = record.path
        for payload in ("{broken", json.dumps({"version": 1, "identity": "foreign",
                                              "values": {"done": True}})):
            path.write_text(payload, encoding="utf-8")
            async with execution_state.execution_scope(self.workspace, "job", "request"):
                with self.assertRaises(RuntimeError):
                    await execution_state.record("plan")

    async def test_cancelled_save_settles_owned_write_before_returning(self):
        entered = threading.Event()
        release = threading.Event()
        original = execution_state.save_json_object

        def slow_write(path, snapshot):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test failed to release write")
            original(path, snapshot)

        async with execution_state.execution_scope(self.workspace, "job", "request"):
            record = await execution_state.record("plan")
            with mock.patch.object(execution_state, "save_json_object", slow_write):
                save = asyncio.create_task(record.save(completed=["first"]))
                for _ in range(100):
                    if entered.is_set():
                        break
                    await asyncio.sleep(.01)
                self.assertTrue(entered.is_set())
                save.cancel()
                await asyncio.sleep(.01)
                self.assertFalse(save.done())
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await save
            self.assertEqual(json.loads(record.path.read_bytes())["values"]["completed"],
                             ["first"])

    async def test_non_json_and_oversized_state_fail_without_overwriting_valid_state(self):
        async with execution_state.execution_scope(self.workspace, "job", "request"):
            record = await execution_state.record("plan")
            await record.save(valid=True)
            with self.assertRaises(ValueError):
                record.set(invalid=float("nan"))
            with mock.patch.object(execution_state, "_MAX_STATE_BYTES", 32):
                with self.assertRaises(ValueError):
                    await record.save(oversized="x" * 50)
            self.assertTrue(json.loads(record.path.read_bytes())["values"]["valid"])
            # Restore the bound and make the owned writer healthy before exit.
            await record.save(oversized=None)

    async def test_native_home_cannot_follow_link_outside_workspace(self):
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        async with execution_state.execution_scope(self.workspace, "job", "request"):
            record = await execution_state.record("plan")
            path = record.directory
            path.rmdir()
            try:
                path.symlink_to(outside.name, target_is_directory=True)
            except OSError:
                self.skipTest("symlinks unavailable")
            with self.assertRaises(ValueError):
                _ = record.directory

    async def test_worker_binding_is_owned_and_internal_calls_have_no_record(self):
        self.assertIsNone(await execution_state.record("plan"))
        async def execute(*_args, **_kwargs):
            state = execution_state.current_run()
            self.assertIsNotNone(state)
            state.set(native_session_id="saved")
            return AgentResult(text="done"), False
        async with execution_state.execution_scope(self.workspace, "job", "request"):
            with mock.patch.object(agent, "_drive_backend_impl", side_effect=execute):
                result, _ = await agent._drive_backend(
                    SimpleNamespace(name="test", default_model="chosen"),
                    self.workspace, "wrapped", None, "full", effort=100,
                )
            self.assertEqual(result.text, "done")
            self.assertIsNone(execution_state.current_run())

    async def test_agent_uses_canonical_request_and_stable_job_id(self):
        async def run_turn(*_args, **_kwargs):
            current = execution_state.current_execution()
            self.assertEqual(current.execution_id, "queue-id")
            self.assertEqual(current.original_request, "canonical")
            return AgentResult(text="done")
        with (mock.patch.object(agent, "_wait_for_pending_session_writes", new=mock.AsyncMock()),
              mock.patch.object(agent, "_run_turn", side_effect=run_turn)):
            await agent.run("transport wrapper", self.workspace, 1,
                            execution_id="queue-id", original_request="canonical")
        self.assertIsNone(execution_state.current_execution())

    async def test_completed_cleanup_only_removes_selected_job(self):
        paths = []
        for job in ("completed", "still-running"):
            async with execution_state.execution_scope(self.workspace, job, "request"):
                record = await execution_state.record("plan")
                await record.save(evidence="keep until complete")
                paths.append(record.path)
        schedule = Path(self.workspace, ".cozter", "schedules.json")
        schedule.write_bytes(b"unchanged-schedule-fixture")
        await execution_state.discard_execution(self.workspace, "completed")
        self.assertFalse(paths[0].exists())
        self.assertTrue(paths[1].exists())
        self.assertEqual(schedule.read_bytes(), b"unchanged-schedule-fixture")

    async def test_missing_native_session_retries_once_before_activity_and_keeps_inject_open(self):
        calls = []
        injected = []
        queue = agent.asyncio.Queue()
        async def execute(*args, **kwargs):
            calls.append(args[2])
            self.assertFalse(kwargs["close_inject_on_completion"])
            if len(calls) == 1:
                state = execution_state.current_run()
                native_id = "01a126fa-e54a-7753-8078-3e31ff8fc962"
                await state.save(native_session_id=native_id, native_resume_attempted=True,
                                 native_tools_started=False)
                result = AgentResult()
                CodexBackend().parse_event({"type": "error",
                                           "message": "No session found with ID: " + native_id}, result)
                return result, False
            # A second unavailable result is returned, never looped forever.
            return AgentResult(text="recovered"), False
        async with execution_state.execution_scope(self.workspace, "job", "request"):
            with mock.patch.object(agent, "_drive_backend_impl", side_effect=execute):
                result, restarting = await agent._drive_backend(
                    SimpleNamespace(name="test", default_model="chosen"),
                    self.workspace, "original prompt", None, "full", effort=100,
                    inject_queue=queue, injected=injected,
                )
        self.assertEqual(len(calls), 2)
        self.assertIn("saved evidence", calls[1])
        self.assertFalse(restarting)
        self.assertEqual(result.text, "recovered")

    async def test_native_missing_session_does_not_replay_ambiguous_work(self):
        async def execute(*_args, **_kwargs):
            await execution_state.current_run().save(
                native_resume_unavailable=True, native_tools_started=True,
            )
            return AgentResult(error="failed after work", events=[ChatEvent(kind="tool", content="write")]), False
        async with execution_state.execution_scope(self.workspace, "job", "request"):
            with mock.patch.object(agent, "_drive_backend_impl", side_effect=execute) as drive:
                await agent._drive_backend(SimpleNamespace(name="test", default_model="chosen"),
                                           self.workspace, "prompt", None, "full", effort=100)
            self.assertEqual(drive.await_count, 1)

    async def test_internal_call_does_not_capture_worker_session_even_when_nested(self):
        async def launch(*_args, **kwargs):
            self.assertIsNone(execution_state.current_run())
            self.assertTrue(kwargs["compaction"])
            return object()
        async def drain(*_args, **_kwargs):
            self.assertIsNone(execution_state.current_run())
            return "summary"
        async with execution_state.execution_scope(self.workspace, "job", "request"):
            async with execution_state.backend_scope("codex", "chosen", "full", 100):
                worker = execution_state.current_run()
                with mock.patch.object(utils, "drain_llm_subprocess", side_effect=drain):
                    text = await utils.run_internal_backend(
                        SimpleNamespace(launch=launch), self.workspace, "internal", "chosen",
                        label="summary", log=execution_state.logger, missing_executable_message="missing %s",
                    )
                self.assertEqual(text, "summary")
                self.assertIs(execution_state.current_run(), worker)

    async def test_unicode_size_limit_matches_actual_disk_encoding(self):
        async with execution_state.execution_scope(self.workspace, "job", "request"):
            record = await execution_state.record("plan")
            await record.save(valid=True)
            before = record.path.read_bytes()
            with mock.patch.object(execution_state, "_MAX_STATE_BYTES", 350):
                with self.assertRaises(ValueError):
                    await record.save(unicode="한" * 50)
            self.assertEqual(record.path.read_bytes(), before)
            await record.save(unicode=None)


class FileReadResultBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_builtin_read_has_larger_bounded_result_without_enlarging_plugins(self):
        with tempfile.TemporaryDirectory() as ws:
            Path(ws, "large.txt").write_text("x" * 40_000, encoding="utf-8")
            result = await agent_tools.execute_tool(
                "read_file", {"path": "large.txt"}, ws, "confirm", lambda _event: None,
            )
            self.assertGreater(len(result), 4_000)
            self.assertLessEqual(len(result), 32 * 1024)
            self.assertIn("truncated", result)


if __name__ == "__main__":
    unittest.main()
