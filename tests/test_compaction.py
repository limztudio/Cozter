"""Concurrency coverage for automatic session compaction."""

import asyncio
import os
import tempfile
import unittest
from unittest import mock

from Cozter import colony, compaction, session


class CompactionConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_session_compacts_only_once_at_a_time(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def slow_compact(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return ("x" * 100, [], None)

        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path, name="Manual")
            session.append_messages(workspace_path, data["id"], [{
                "role": "user", "content": "hello",
            }])
            with (
                mock.patch.object(
                    compaction.workspace_mod, "get_compact_interval", return_value=1,
                ),
                mock.patch.object(
                    compaction, "compact_session", side_effect=slow_compact,
                ),
                mock.patch.object(
                    compaction.colony, "bump_compact_count", return_value=1,
                ),
                mock.patch.object(compaction.colony, "maybe_trigger"),
            ):
                first = asyncio.create_task(compaction.maybe_compact(
                    workspace_path, data["id"], "model", backend_name="backend",
                ))
                await asyncio.wait_for(started.wait(), timeout=1)
                await compaction.maybe_compact(
                    os.path.join(workspace_path, "."),
                    data["id"], "model", backend_name="backend",
                )
                self.assertEqual(calls, 1)

                release.set()
                await first

                # The guard is released after the first compaction finishes.
                await compaction.maybe_compact(
                    workspace_path, data["id"], "model", backend_name="backend",
                )
                self.assertEqual(calls, 2)

    async def test_colony_waits_for_a_running_compaction(self) -> None:
        compaction_started = asyncio.Event()
        release_compaction = asyncio.Event()
        colony_started = asyncio.Event()

        async def slow_compact(*_args, **_kwargs):
            compaction_started.set()
            await release_compaction.wait()
            return ("x" * 100, [], None)

        async def fake_consolidate(*_args, **_kwargs) -> bool:
            colony_started.set()
            return False

        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path, name="Manual")
            session.append_messages(workspace_path, data["id"], [{
                "role": "user", "content": "hello",
            }])
            with (
                mock.patch.object(
                    compaction.workspace_mod, "get_compact_interval", return_value=1,
                ),
                mock.patch.object(
                    compaction, "compact_session", side_effect=slow_compact,
                ),
                mock.patch.object(
                    compaction.colony, "bump_compact_count", return_value=1,
                ),
                mock.patch.object(compaction.colony, "maybe_trigger"),
                mock.patch.object(
                    colony, "_consolidate_inner", side_effect=fake_consolidate,
                ),
            ):
                compact_task = asyncio.create_task(compaction.maybe_compact(
                    workspace_path, data["id"], "model", backend_name="backend",
                ))
                await asyncio.wait_for(compaction_started.wait(), timeout=1)
                colony_task = asyncio.create_task(colony.consolidate(
                    workspace_path, "model", backend_name="backend",
                ))
                await asyncio.sleep(0)
                self.assertFalse(colony_started.is_set())

                release_compaction.set()
                await compact_task
                await asyncio.wait_for(colony_started.wait(), timeout=1)
                await colony_task

    async def test_compaction_waits_for_a_running_colony_pass(self) -> None:
        colony_started = asyncio.Event()
        release_colony = asyncio.Event()
        compaction_started = asyncio.Event()
        release_compaction = asyncio.Event()

        async def slow_consolidate(*_args, **_kwargs) -> bool:
            colony_started.set()
            await release_colony.wait()
            return False

        async def slow_compact(*_args, **_kwargs):
            compaction_started.set()
            await release_compaction.wait()
            return ("x" * 100, [], None)

        with tempfile.TemporaryDirectory() as workspace_path:
            data = session.create_session(workspace_path, name="Manual")
            session.append_messages(workspace_path, data["id"], [{
                "role": "user", "content": "hello",
            }])
            with (
                mock.patch.object(
                    colony, "_consolidate_inner", side_effect=slow_consolidate,
                ),
                mock.patch.object(
                    compaction.workspace_mod, "get_compact_interval", return_value=1,
                ),
                mock.patch.object(
                    compaction, "compact_session", side_effect=slow_compact,
                ),
                mock.patch.object(
                    compaction.colony, "bump_compact_count", return_value=1,
                ),
                mock.patch.object(compaction.colony, "maybe_trigger"),
            ):
                colony_task = asyncio.create_task(colony.consolidate(
                    workspace_path, "model", backend_name="backend",
                ))
                await asyncio.wait_for(colony_started.wait(), timeout=1)
                compact_task = asyncio.create_task(compaction.maybe_compact(
                    workspace_path, data["id"], "model", backend_name="backend",
                ))
                await asyncio.sleep(0)
                self.assertFalse(compaction_started.is_set())

                release_colony.set()
                await colony_task
                await asyncio.wait_for(compaction_started.wait(), timeout=1)
                release_compaction.set()
                await compact_task


if __name__ == "__main__":
    unittest.main()
