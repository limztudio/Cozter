"""Concurrency coverage for automatic session compaction."""

import asyncio
import tempfile
import unittest
from unittest import mock

from Cozter import compaction, session


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
                    workspace_path, data["id"], "model", backend_name="backend",
                )
                self.assertEqual(calls, 1)

                release.set()
                await first

                # The guard is released after the first compaction finishes.
                await compaction.maybe_compact(
                    workspace_path, data["id"], "model", backend_name="backend",
                )
                self.assertEqual(calls, 2)


if __name__ == "__main__":
    unittest.main()
