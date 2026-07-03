"""Tests for the per-workspace run lock that serializes agent turns."""

import asyncio
import unittest

from Cozter import workspace


class RunLockTests(unittest.TestCase):
    def test_run_lock_is_per_workspace(self) -> None:
        a = workspace.get_run_lock("/ws/a")
        self.assertIs(a, workspace.get_run_lock("/ws/a"))
        self.assertIsNot(a, workspace.get_run_lock("/ws/b"))
        self.assertIsInstance(a, asyncio.Lock)
        # Distinct from the file lock so a turn can hold it without
        # deadlocking on the reentrant-unsafe file lock.
        self.assertIsNot(
            workspace.get_run_lock("/ws/a"), workspace.get_lock("/ws/a"),
        )

    def test_run_lock_serializes_turns(self) -> None:
        order: list[str] = []

        async def worker(lock: asyncio.Lock, tag: str) -> None:
            async with lock:
                order.append(f"{tag}-start")
                await asyncio.sleep(0.01)
                order.append(f"{tag}-end")

        async def scenario() -> None:
            lock = workspace.get_run_lock("/ws/serialize")
            await asyncio.gather(worker(lock, "A"), worker(lock, "B"))

        asyncio.run(scenario())
        # Whichever acquires first fully completes before the other starts:
        # no interleaving.
        self.assertIn(order, [
            ["A-start", "A-end", "B-start", "B-end"],
            ["B-start", "B-end", "A-start", "A-end"],
        ])


if __name__ == "__main__":
    unittest.main()
