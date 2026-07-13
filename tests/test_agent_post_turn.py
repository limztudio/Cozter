"""Regression coverage for latency-sensitive post-turn maintenance."""

import asyncio
import tempfile
import unittest
from unittest import mock

from Cozter import agent, session
from Cozter.backends_agent.base import AgentResult


class _ImmediateBackend:
    name = "post-turn-test"
    supports_typed_plugins = True


class AgentPostTurnMaintenanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_compaction_does_not_block_the_agent_result(self) -> None:
        """A slow compaction must not delay delivery to a chat platform."""
        started = asyncio.Event()
        release = asyncio.Event()
        maintenance_tasks: list[asyncio.Task] = []
        backend = _ImmediateBackend()

        async def slow_compaction(*_args, **_kwargs) -> None:
            started.set()
            await release.wait()

        async def fast_backend(*_args, **_kwargs):
            return AgentResult(text="reply"), False

        original_background_task = agent.create_background_task

        def capture_background_task(coro, *, name, log=None):
            task = original_background_task(coro, name=name, log=log)
            maintenance_tasks.append(task)
            return task

        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path, name="Manual")
            with (
                mock.patch.object(
                    agent.backends_agent, "get_backend", return_value=backend,
                ),
                mock.patch.object(
                    agent, "_drive_backend", side_effect=fast_backend,
                ),
                mock.patch.object(
                    agent.compaction, "maybe_compact", side_effect=slow_compaction,
                ),
                mock.patch.object(
                    agent, "create_background_task",
                    side_effect=capture_background_task,
                ),
            ):
                turn = asyncio.create_task(agent.run(
                    "hello", workspace_path, 1,
                    backend_name=backend.name, session_id=data["id"],
                ))
                try:
                    await asyncio.wait_for(started.wait(), timeout=1)
                    result = await asyncio.wait_for(asyncio.shield(turn), 0.1)
                    self.assertEqual(result.text, "reply")
                    self.assertEqual(len(maintenance_tasks), 1)
                    self.assertFalse(maintenance_tasks[0].done())
                finally:
                    release.set()
                    await turn
                    await asyncio.gather(*maintenance_tasks)


if __name__ == "__main__":
    unittest.main()
