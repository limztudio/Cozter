"""Ensure filesystem discovery scans do not block the asyncio event loop."""

import asyncio
import os
import tempfile
import threading
import unittest
from unittest import mock

from Cozter.agent_tools.builtin.glob import GlobTool
from Cozter.agent_tools.builtin.list_dir import ListDirTool
from Cozter.agent_tools.builtin.tree import TreeTool


async def _wait_for_start(start: threading.Event) -> None:
    for _ in range(100):
        if start.is_set():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("filesystem scan did not start")


class DiscoveryToolAsyncTests(unittest.TestCase):
    def test_list_dir_scan_runs_off_the_event_loop(self) -> None:
        async def run() -> None:
            started = threading.Event()
            release = threading.Event()
            tool = ListDirTool()

            def slow_list_entries(
                _path: str, _max_results: int,
            ) -> tuple[list[str], int]:
                started.set()
                release.wait(timeout=1)
                return ["directory/", "file.txt"], 2

            with tempfile.TemporaryDirectory() as workspace:
                os.mkdir(os.path.join(workspace, "directory"))
                with open(
                    os.path.join(workspace, "file.txt"), "w",
                    encoding="utf-8",
                ):
                    pass
                with mock.patch.object(
                    tool, "_list_entries", side_effect=slow_list_entries,
                ):
                    task = asyncio.create_task(tool.run(workspace, {}))
                    try:
                        await _wait_for_start(started)
                        self.assertFalse(task.done())
                    finally:
                        release.set()
                    self.assertEqual(
                        await task,
                        "directory/\nfile.txt",
                    )

        asyncio.run(run())

    def test_glob_scan_runs_off_the_event_loop(self) -> None:
        async def run() -> None:
            started = threading.Event()
            release = threading.Event()

            def slow_iter_workspace_files(*args):
                started.set()
                release.wait(timeout=1)
                return []

            with mock.patch(
                "Cozter.agent_tools.builtin.glob.iter_workspace_files",
                side_effect=slow_iter_workspace_files,
            ):
                task = asyncio.create_task(
                    GlobTool().run("/workspace", {"pattern": "**/*"}),
                )
                try:
                    await _wait_for_start(started)
                    self.assertFalse(task.done())
                finally:
                    release.set()
                self.assertEqual(await task, "No files matched: **/*")

        asyncio.run(run())

    def test_tree_scan_runs_off_the_event_loop(self) -> None:
        async def run() -> None:
            started = threading.Event()
            release = threading.Event()
            tool = TreeTool()

            def slow_walk(*args) -> bool:
                started.set()
                release.wait(timeout=1)
                args[4].append("entry")
                return False

            with tempfile.TemporaryDirectory() as workspace:
                with mock.patch.object(tool, "_walk", side_effect=slow_walk):
                    task = asyncio.create_task(tool.run(workspace, {}))
                    try:
                        await _wait_for_start(started)
                        self.assertFalse(task.done())
                    finally:
                        release.set()
                    self.assertEqual(await task, "entry")

        asyncio.run(run())
