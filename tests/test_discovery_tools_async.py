"""Ensure filesystem discovery scans do not block the asyncio event loop."""

import asyncio
import os
import re
import tempfile
import threading
import unittest
from unittest import mock

from Cozter import agent_tools
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
    def test_dispatch_pages_long_names_without_losing_entries(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as workspace:
                names = [f"{index:04d}" + "x" * 175 for index in range(200)]
                for name in names:
                    with open(os.path.join(workspace, name), "w", encoding="utf-8"):
                        pass
                for tool_name, limit in (("list_dir", "max_results"), ("tree", "max_entries")):
                    offset = 0
                    seen = []
                    while True:
                        result = await agent_tools.execute_tool(
                            tool_name, {limit: 1000, "offset": offset}, workspace,
                            "full", lambda _event: None,
                        )
                        rows = result.splitlines()
                        marker = re.search(r"offset=(\d+)", rows[-1])
                        seen.extend(rows[:-1] if marker else rows)
                        if not marker:
                            break
                        next_offset = int(marker.group(1))
                        self.assertGreater(next_offset, offset)
                        offset = next_offset
                    self.assertEqual(seen, names)

        asyncio.run(run())

    def test_paging_reaches_entries_beyond_both_limits(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as workspace:
                for index in range(2001):
                    with open(os.path.join(workspace, f"{index:04}.txt"), "w", encoding="utf-8"):
                        pass
                listing = ListDirTool()
                first = await listing.run(workspace, {"max_results": 1000})
                last = await listing.run(workspace, {"max_results": 1000, "offset": 2000})
                self.assertIn("offset=1000", first)
                self.assertEqual(last, "2000.txt")
                tree = TreeTool()
                first = await tree.run(workspace, {"max_entries": 2000})
                last = await tree.run(workspace, {"max_entries": 2000, "offset": 2000})
                self.assertIn("offset=2000", first)
                self.assertEqual(last, "2000.txt")

        asyncio.run(run())

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

    def test_list_dir_continues_after_one_entry_stat_error(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as workspace:
                os.mkdir(os.path.join(workspace, "directory"))
                with open(
                    os.path.join(workspace, "file.txt"), "w", encoding="utf-8",
                ):
                    pass

                real_scandir = os.scandir

                class FlakyEntry:
                    def __init__(self, entry: os.DirEntry) -> None:
                        self.name = entry.name
                        self._entry = entry

                    def is_dir(self, *args, **kwargs):
                        if self.name == "directory":
                            raise OSError("vanished")
                        return self._entry.is_dir(*args, **kwargs)

                class FlakyScan:
                    def __init__(self, path: str) -> None:
                        self._scan = real_scandir(path)

                    def __enter__(self):
                        return (
                            FlakyEntry(entry) for entry in self._scan.__enter__()
                        )

                    def __exit__(self, *args):
                        return self._scan.__exit__(*args)

                with mock.patch(
                    "Cozter.agent_tools.builtin.list_dir.os.scandir",
                    FlakyScan,
                ):
                    result = await ListDirTool().run(workspace, {})

                self.assertIn("file.txt", result)
                self.assertIn("directory", result)
                self.assertNotIn("directory/", result)

        asyncio.run(run())

    def test_glob_scan_runs_off_the_event_loop(self) -> None:
        async def run() -> None:
            started = threading.Event()
            release = threading.Event()

            def slow_iter_workspace_files(*args, **kwargs):
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

            def slow_walk(*args, **_kwargs) -> bool:
                started.set()
                release.wait(timeout=1)
                args[4].append("entry")
                return False

            with tempfile.TemporaryDirectory() as workspace, mock.patch.object(
                tool, "_walk", side_effect=slow_walk,
            ):
                task = asyncio.create_task(tool.run(workspace, {}))
                try:
                    await _wait_for_start(started)
                    self.assertFalse(task.done())
                finally:
                    release.set()
                self.assertEqual(await task, "entry")

        asyncio.run(run())
