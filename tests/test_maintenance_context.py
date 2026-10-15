"""Post-turn maintenance must not retain completed execution contexts."""

import asyncio
import contextvars
import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import weakref

from Cozter import agent, execution_state, flexible, session
from Cozter.backends_agent.base import AgentResult


class _Backend:
    name = "maintenance-context-test"
    supports_typed_plugins = True
    default_summary_model = "summary-model"


class MaintenanceContextTests(unittest.IsolatedAsyncioTestCase):
    async def _exercise(self, outcome):
        started = asyncio.Event()
        release = asyncio.Event()
        tasks = []
        execution_refs = []
        budget_refs = []
        foreground = []
        maintenance = []
        probe = contextvars.ContextVar("maintenance_fixture", default="default")
        token = probe.set("foreground")
        original_background = agent.create_background_task

        def capture(coro, *, name, log=None):
            task = original_background(coro, name=name, log=log)
            # Retaining the normal session-write Task would itself retain
            # that task's deliberately inherited foreground context.
            if name.startswith("post-turn-maintenance:"):
                tasks.append(task)
            return task

        async def drive(*args, **kwargs):
            execution = execution_state.current_execution()
            budget = agent._TURN_BUDGET.get()
            self.assertIsNotNone(execution)
            self.assertIsNotNone(budget)
            self.assertEqual(execution.execution_id, "durable-job")
            self.assertEqual(probe.get(), "foreground")
            execution_refs.append(weakref.ref(execution))
            budget_refs.append(weakref.ref(budget))
            async with execution_state.backend_scope(_Backend.name, "worker-model", "full", 50) as record:
                self.assertIs(execution_state.current_run(), record)
                await record.save(native_session_id="saved-provider-session")
                # The fixture mimics a large retained HTTP conversation,
                # without provider calls or an unnecessarily large disk file.
                payload = "x" * (8 * 1024 * 1024)
                record._values["http_messages"] = [payload]
                budget.latest_draft = payload
                foreground.append(str(record.path))
            return AgentResult(text="finished"), False

        async def judge(*args, **kwargs):
            self.assertIsNotNone(execution_state.current_execution())
            self.assertIsNotNone(agent._TURN_BUDGET.get())
            return flexible.JudgeVerdict(should_continue=False), False

        async def compact(path, session_id, summary_model, **kwargs):
            maintenance.append((
                execution_state.current_execution(), execution_state.current_run(),
                agent._TURN_BUDGET.get(), probe.get(),
                path, session_id, summary_model, kwargs,
            ))
            started.set()
            await release.wait()
            if outcome == "failure":
                raise RuntimeError("maintenance fixture failure")

        async def title(*args, **kwargs):
            self.assertIsNone(execution_state.current_execution())
            self.assertIsNone(agent._TURN_BUDGET.get())

        try:
            with tempfile.TemporaryDirectory() as workspace:
                data = session.create_session(workspace, name="Manual")
                with (
                    mock.patch.object(agent.backends_agent, "get_backend", return_value=_Backend()),
                    # Plain functions avoid Mock call-history retaining the
                    # streamed-event closure and its normal-write Task.
                    mock.patch.object(agent, "_drive_backend", new=drive),
                    mock.patch.object(agent, "_judge_draft", new=judge),
                    mock.patch.object(agent, "create_background_task", new=capture),
                    mock.patch.object(agent.compaction, "maybe_compact", new=compact),
                    mock.patch.object(agent.titling, "maybe_auto_title", new=title),
                ):
                    try:
                        result = await agent.run(
                            "do the requested work", workspace, 1,
                            backend_name=_Backend.name, session_id=data["id"],
                            execution_id="durable-job", original_request="canonical request",
                        )
                        await asyncio.wait_for(started.wait(), 2)
                        self.assertEqual(result.text, "finished")
                        self.assertEqual(len(tasks), 1)
                        self.assertFalse(tasks[0].done())
                        self.assertEqual(maintenance[0][:4], (None, None, None, "default"))
                        self.assertEqual(maintenance[0][4:7], (workspace, data["id"], "summary-model"))
                        self.assertEqual(maintenance[0][7]["backend_name"], _Backend.name)
                        self.assertEqual(maintenance[0][7]["context_targets"], ((_Backend.name, None),))
                        self.assertIsNone(execution_state.current_execution())
                        self.assertIsNone(execution_state.current_run())
                        self.assertIsNone(agent._TURN_BUDGET.get())
                        self.assertEqual(probe.get(), "foreground")
                        disk = json.loads(Path(foreground[0]).read_bytes())
                        self.assertEqual(disk["values"]["native_session_id"], "saved-provider-session")
                        await asyncio.sleep(0)
                        gc.collect()
                        # The maintenance task is still blocked, but both
                        # heavy foreground owners must already be collectable.
                        self.assertTrue(all(reference() is None for reference in execution_refs))
                        self.assertTrue(all(reference() is None for reference in budget_refs))
                        if outcome == "cancel":
                            tasks[0].cancel()
                        elif outcome == "failure":
                            with self.assertLogs(agent.logger.name, level="ERROR") as logs:
                                release.set()
                                results = await asyncio.gather(*tasks, return_exceptions=True)
                                await asyncio.sleep(0)
                            self.assertIsInstance(results[0], RuntimeError)
                            self.assertIn("maintenance fixture failure", "\n".join(logs.output))
                        else:
                            release.set()
                            await asyncio.gather(*tasks)
                    finally:
                        release.set()
                        await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            probe.reset(token)

    async def test_blocked_maintenance_has_clean_context_and_releases_completed_owners(self):
        await self._exercise("success")

    async def test_failed_maintenance_does_not_retain_completed_owners(self):
        await self._exercise("failure")

    async def test_cancelled_maintenance_does_not_retain_completed_owners(self):
        await self._exercise("cancel")
