"""Root discovery avoids ignored trees without closing explicit paths."""

import asyncio
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from Cozter.agent_tools import base
from Cozter.agent_tools.builtin.glob import GlobTool
from Cozter.agent_tools.builtin.grep import GrepTool


@unittest.skipUnless(shutil.which("git"), "Git is required for ignore fixtures")
class IgnoreAwareDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name, "workspace")
        self.workspace.mkdir()
        subprocess.run(["git", "init", "-q", str(self.workspace)], check=True)
        self._write(".gitignore", "archive/\n*.ignored\n")
        self._write("root.cpp")
        self._write("src/deep/current.cpp")
        self._write("src/new α.cpp")
        self._write("archive/old.cpp")
        self._write("kept.ignored")
        self._write("skip.ignored")
        subprocess.run(
            ["git", "-C", str(self.workspace), "add", "-f", "root.cpp", "kept.ignored"],
            check=True,
        )

    def _write(self, relative: str, text: str = "needle\n") -> Path:
        target = self.workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target

    async def test_inventory_keeps_tracked_ignored_and_untracked_source_with_stable_pages(self) -> None:
        inventory = await base.git_discovery_files(str(self.workspace), str(self.workspace), "**/*")
        self.assertIsNotNone(inventory)
        self.assertTrue({"root.cpp", "kept.ignored", "src/new α.cpp"}.issubset(inventory))
        self.assertNotIn("skip.ignored", inventory)
        self.assertNotIn("archive/old.cpp", inventory)
        first = await GlobTool().run(str(self.workspace), {"pattern": "**/*.cpp", "max_results": 1})
        second = await GlobTool().run(
            str(self.workspace), {"pattern": "**/*.cpp", "max_results": 1, "offset": 1},
        )
        third = await GlobTool().run(
            str(self.workspace), {"pattern": "**/*.cpp", "max_results": 1, "offset": 2},
        )
        self.assertIn("root.cpp", first)
        self.assertIn("src/new α.cpp", second)
        self.assertIn("src/deep/current.cpp", third)
        self.assertNotIn("archive/", first + second + third)

    async def test_root_grep_does_not_walk_ignored_trees_but_explicit_targets_remain_searchable(self) -> None:
        result = await GrepTool().run(str(self.workspace), {"pattern": "needle", "glob": "**/*.cpp"})
        self.assertIn("root.cpp:1: needle", result)
        self.assertIn("src/new α.cpp:1: needle", result)
        self.assertNotIn("archive/", result)
        for args in (
            {"path": "archive"},
            {"glob": "archive/**/*.cpp"},
            {"glob": "**/archive/*.cpp"},
        ):
            result = await GrepTool().run(str(self.workspace), {"pattern": "needle", **args})
            self.assertIn("archive/old.cpp:1: needle", result)
        explicit = await GlobTool().run(str(self.workspace), {"pattern": "archive/**/*.cpp"})
        self.assertEqual(explicit, "archive/old.cpp")
        literal = await GlobTool().run(str(self.workspace), {"pattern": "skip.ignored"})
        self.assertEqual(literal, "skip.ignored")

    async def test_root_inventory_retains_glob_semantics_and_symlink_authority(self) -> None:
        inventory = await base.git_discovery_files(str(self.workspace), str(self.workspace), "**/*.cpp")
        for pattern in ("*.cpp", "**/*.cpp", "*/new*.cpp", "**/current.cpp"):
            with self.subTest(pattern=pattern):
                expected = GrepTool._scan(
                    str(self.workspace), str(self.workspace), pattern, re.compile("needle"), 200,
                    source_files=inventory, sort_entries=True,
                )
                actual = await GrepTool().run(
                    str(self.workspace), {"pattern": "needle", "glob": pattern},
                )
                self.assertEqual(actual, "\n".join(expected))
        alias = self._write("escape.cpp")
        outside = Path(self.workspace.parent, "outside.cpp")
        outside.write_text("needle\n", encoding="utf-8")
        original = os.path.realpath

        def resolve(path, *args, **kwargs):
            return str(outside) if os.fspath(path) == str(alias) else original(path, *args, **kwargs)

        with mock.patch.object(base.os.path, "realpath", side_effect=resolve):
            rows = list(base.iter_workspace_files(
                str(self.workspace), str(self.workspace), "**/*.cpp", source_files=["escape.cpp"],
            ))
            self.assertEqual(rows, [])
            with base.tool_permission_scope("full"):
                self.assertEqual(list(base.iter_workspace_files(
                    str(self.workspace), str(self.workspace), "**/*.cpp", source_files=["escape.cpp"],
                ))[0][1], "escape.cpp")

    async def test_cached_deleted_and_directory_entries_are_skipped_without_dropping_broken_file_links(self) -> None:
        deleted = self.workspace / "root.cpp"
        deleted.unlink()
        broken = self.workspace / "broken.cpp"
        original = os.path.lexists

        def lexists(path):
            return os.fspath(path) == str(broken) or original(path)

        with mock.patch.object(base.os.path, "lexists", side_effect=lexists):
            matches = list(base.iter_workspace_files(
                str(self.workspace), str(self.workspace), "**/*",
                source_files=["root.cpp", "src", "broken.cpp", "src/new α.cpp"],
            ))
        self.assertEqual([row[1] for row in matches], ["broken.cpp", "src/new α.cpp"])

    async def test_inventory_timeout_and_cancel_reap_owned_child_without_blocking_loop(self) -> None:
        processes = []
        create = asyncio.create_subprocess_exec
        started = asyncio.Event()

        async def slow_git(*_args, **kwargs):
            proc = await create(sys.executable, "-c", "import time; time.sleep(30)", **kwargs)
            processes.append(proc)
            started.set()
            return proc

        with mock.patch.object(base.asyncio, "create_subprocess_exec", side_effect=slow_git):
            with mock.patch.object(base, "_GIT_DISCOVERY_TIMEOUT_SECONDS", 0.01):
                self.assertIsNone(await base.git_discovery_files(
                    str(self.workspace), str(self.workspace), "**/*",
                ))
            self.assertIsNotNone(processes[-1].returncode)
            started.clear()
            owner = asyncio.create_task(base.git_discovery_files(
                str(self.workspace), str(self.workspace), "**/*",
            ))
            await asyncio.wait_for(started.wait(), 2)
            owner.cancel("stop root search")
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(owner, 2)
            self.assertIsNotNone(processes[-1].returncode)

    async def test_failed_or_oversized_inventory_falls_back_without_partial_results(self) -> None:
        create = asyncio.create_subprocess_exec

        async def oversized_git(*_args, **kwargs):
            return await create(sys.executable, "-c", "print('x' * 2048)", **kwargs)

        with (
            mock.patch.object(base.asyncio, "create_subprocess_exec", side_effect=oversized_git),
            mock.patch.object(base, "_GIT_DISCOVERY_MAX_BYTES", 128),
        ):
            self.assertIsNone(await base.git_discovery_files(
                str(self.workspace), str(self.workspace), "**/*",
            ))

        async def failed_git(*_args, **kwargs):
            return await create(
                sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'root.cpp\\0'); sys.exit(2)",
                **kwargs,
            )

        with mock.patch.object(base.asyncio, "create_subprocess_exec", side_effect=failed_git):
            self.assertIsNone(await base.git_discovery_files(
                str(self.workspace), str(self.workspace), "**/*",
            ))
        self.assertIsNone(await base.git_discovery_files(
            str(self.workspace), str(self.workspace / "src"), "**/*",
        ))

    async def test_cancel_during_git_creation_retains_ownership_until_spawn_is_reaped(self) -> None:
        create = asyncio.create_subprocess_exec
        entered = asyncio.Event()
        release = asyncio.Event()
        processes = []

        async def delayed_git(*_args, **kwargs):
            entered.set()
            await release.wait()
            proc = await create(sys.executable, "-c", "import time; time.sleep(30)", **kwargs)
            processes.append(proc)
            return proc

        with mock.patch.object(base.asyncio, "create_subprocess_exec", side_effect=delayed_git):
            owner = asyncio.create_task(base.git_discovery_files(
                str(self.workspace), str(self.workspace), "**/*",
            ))
            try:
                await asyncio.wait_for(entered.wait(), 2)
                owner.cancel("first stop")
                await asyncio.sleep(0)
                owner.cancel("repeated stop")
                await asyncio.sleep(0)
                self.assertFalse(owner.done())
                release.set()
                with self.assertRaises(asyncio.CancelledError) as caught:
                    await asyncio.wait_for(owner, 2)
                self.assertEqual(caught.exception.args, ("first stop",))
                self.assertEqual(len(processes), 1)
                self.assertIsNotNone(processes[0].returncode)
            finally:
                release.set()
                if not owner.done():
                    owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
