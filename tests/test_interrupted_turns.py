"""Stopped turns must leave durable memory for a later resume.

``/stop``, shutdown, and mid-turn restarts cancel the agent task. Session
logging previously ran only after a normal completion, so a stopped turn
vanished from history even though its workspace side effects persisted.
These tests pin the interrupted-turn record: the prompt and whatever partial
output the attempt produced are appended to the session before the
cancellation propagates.
"""

import asyncio
import tempfile
import unittest
from unittest import mock

from Cozter import agent, session
from Cozter.backends_agent.base import AgentResult, ChatEvent

LOG_TASK_PREFIX = "log-interrupted-turn"


class _RecordingBackend:
    name = "interrupted-turn-test"
    supports_typed_plugins = True
    default_summary_model = "summary-default"


class InterruptedTurnMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def _run_until_cancelled(
        self,
        workspace_path: str,
        session_id: str,
        drive: object,
    ) -> list[asyncio.Task]:
        """Start a turn, cancel it once the fake backend is driving it.

        Returns the detached background tasks the cancellation scheduled
        (awaited by the caller so no task outlives the test).
        """
        interrupted_log_tasks: list[asyncio.Task] = []
        original_background_task = agent.create_background_task

        def capture_background_task(coro, *, name, log=None):
            task = original_background_task(coro, name=name, log=log)
            if name.startswith(LOG_TASK_PREFIX):
                interrupted_log_tasks.append(task)
            return task

        entered = asyncio.Event()

        async def drive_wrapper(*args, **kwargs):
            entered.set()
            await drive(*args, **kwargs)

        with (
            mock.patch.object(
                agent.backends_agent, "get_backend",
                return_value=_RecordingBackend(),
            ),
            mock.patch.object(
                agent, "_drive_backend", side_effect=drive_wrapper,
            ),
            mock.patch.object(
                agent, "create_background_task",
                side_effect=capture_background_task,
            ),
        ):
            run_task = asyncio.create_task(agent.run(
                "continue the refactor",
                workspace_path,
                1,
                backend_name=_RecordingBackend.name,
                session_id=session_id,
            ))
            await asyncio.wait_for(entered.wait(), timeout=5)
            run_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await run_task
            # The interrupted-turn record is written by a detached task so a
            # second /stop cannot abort it; give it the loop to finish.
            self.assertEqual(len(interrupted_log_tasks), 1)
            await asyncio.gather(*interrupted_log_tasks)
        return interrupted_log_tasks

    async def _load_messages(self, workspace_path: str, session_id: str):
        data = session.load_session(workspace_path, session_id)
        self.assertIsNotNone(data)
        return data["messages"]

    async def test_stop_persists_prompt_and_partial_output(self) -> None:
        async def drive(_backend, _ws, _prompt, _model, _approval, **kwargs):
            await kwargs["on_event"](
                ChatEvent(kind="text", content="Refactored router.py;"),
            )
            await asyncio.Event().wait()  # hang until /stop cancels us

        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path, name="Work")
            await self._run_until_cancelled(workspace_path, data["id"], drive)
            messages = await self._load_messages(workspace_path, data["id"])

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "continue the refactor")
        self.assertEqual(messages[1]["role"], "assistant")
        note = messages[1]["content"]
        self.assertIn("[Interrupted turn", note)
        self.assertIn("Partial output before the stop:", note)
        self.assertIn("Refactored router.py;", note)

    async def test_stop_without_text_records_last_activity(self) -> None:
        async def drive(_backend, _ws, _prompt, _model, _approval, **kwargs):
            await kwargs["on_event"](
                ChatEvent(kind="tool", content="Running pytest -q\n"),
            )
            await asyncio.Event().wait()

        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path, name="Work")
            await self._run_until_cancelled(workspace_path, data["id"], drive)
            messages = await self._load_messages(workspace_path, data["id"])

        note = messages[1]["content"]
        self.assertIn("[Interrupted turn", note)
        self.assertIn("Last activity before the stop:", note)
        self.assertIn("Running pytest -q", note)
        self.assertNotIn("Partial output before the stop:", note)

    async def test_completed_turn_is_not_marked_interrupted(self) -> None:
        """A normal completion logs exactly one prompt/response pair."""
        async def fast_drive(*_args, **_kwargs):
            return AgentResult(text="all done"), False

        maintenance_tasks: list[asyncio.Task] = []
        original_background_task = agent.create_background_task

        def capture_background_task(coro, *, name, log=None):
            task = original_background_task(coro, name=name, log=log)
            if not name.startswith(LOG_TASK_PREFIX):
                maintenance_tasks.append(task)
            return task

        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path, name="Work")
            with (
                mock.patch.object(
                    agent.backends_agent, "get_backend",
                    return_value=_RecordingBackend(),
                ),
                mock.patch.object(
                    agent, "_drive_backend", side_effect=fast_drive,
                ),
                mock.patch.object(
                    agent.compaction, "maybe_compact",
                    new_callable=mock.AsyncMock,
                ),
                mock.patch.object(
                    agent.titling, "maybe_auto_title",
                    new_callable=mock.AsyncMock,
                ),
                mock.patch.object(
                    agent, "create_background_task",
                    side_effect=capture_background_task,
                ),
            ):
                result = await agent.run(
                    "continue the refactor",
                    workspace_path,
                    1,
                    backend_name=_RecordingBackend.name,
                    session_id=data["id"],
                )
                await asyncio.gather(*maintenance_tasks)

            self.assertEqual(result.text, "all done")
            messages = await self._load_messages(workspace_path, data["id"])
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0]["content"], "continue the refactor")
            self.assertEqual(messages[1]["content"], "all done")


if __name__ == "__main__":
    unittest.main()
