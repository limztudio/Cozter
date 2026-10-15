"""Tree recursion keeps bounded permissions across Windows directory junctions."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from Cozter.agent_tools import base
from Cozter.agent_tools.builtin.tree import TreeTool


class TreeJunctionPermissionTests(unittest.IsolatedAsyncioTestCase):
    async def scan(self, workspace, permission, **args):
        with base.tool_permission_scope(permission):
            return await TreeTool().run(str(workspace), args)

    async def test_simulated_escaping_directory_is_visible_but_not_recursed_in_bounded_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory, "workspace")
            blocked = workspace / "blocked"
            blocked.mkdir(parents=True)
            (blocked / "external.txt").touch()
            allowed = workspace / "allowed"
            allowed.mkdir()
            (allowed / "internal.txt").touch()
            outside = Path(directory, "outside")
            outside.mkdir()
            original = os.path.realpath

            def resolve(path, *args, **kwargs):
                if os.fspath(path) == str(blocked):
                    return str(outside)
                return original(path, *args, **kwargs)

            with mock.patch.object(os.path, "realpath", side_effect=resolve):
                for permission in ("auto", "confirm"):
                    result = await self.scan(workspace, permission, depth=3)
                    self.assertIn("blocked/", result)
                    self.assertNotIn("external.txt", result)
                    self.assertIn("  internal.txt", result)
                full = await self.scan(workspace, "full", depth=3)
                self.assertIn("  external.txt", full)

    async def test_ordinary_order_depth_and_entry_limits_equal_unchanged_full_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            for name in ("z-last.txt", "a-first.txt", "b-dir/deep.txt", "a-dir/z.txt", "a-dir/child/end.txt"):
                path = workspace / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            for depth in (1, 2, 5):
                for maximum in (1, 2, 4, 8, 200):
                    expected = await self.scan(workspace, "full", depth=depth, max_entries=maximum)
                    for permission in ("auto", "confirm"):
                        self.assertEqual(await self.scan(workspace, permission, depth=depth, max_entries=maximum),
                                         expected, (permission, depth, maximum))

    async def test_containment_resolves_recursed_directories_without_resolving_every_file(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            for index in range(80):
                path = workspace / f"dir-{index % 4}" / f"file-{index}.txt"
                path.parent.mkdir(exist_ok=True)
                path.touch()
            original = os.path.realpath
            calls = {}
            results = {}
            for permission in ("auto", "confirm", "full"):
                with mock.patch.object(os.path, "realpath", wraps=original) as resolve:
                    results[permission] = await self.scan(workspace, permission)
                calls[permission] = [os.fspath(call.args[0]) for call in resolve.call_args_list]
                self.assertFalse(any(path.endswith(".txt") for path in calls[permission]))
            self.assertEqual(results["auto"], results["full"])
            self.assertEqual(results["confirm"], results["full"])
            for permission in ("auto", "confirm"):
                for index in range(4):
                    target = str(workspace / f"dir-{index}")
                    self.assertEqual(calls[permission].count(target), 1)
                    self.assertNotIn(target, calls["full"])

    @unittest.skipUnless(sys.platform == "win32", "Windows directory junctions")
    async def test_actual_windows_root_inside_and_outside_junctions_preserve_permission_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            workspace = parent / "workspace"
            source = workspace / "source"
            source.mkdir(parents=True)
            (source / "internal.txt").touch()
            outside = parent / "outside"
            outside.mkdir()
            (outside / "external.txt").touch()
            selected_subdir = workspace / "selected"
            selected_subdir.mkdir()
            links = []
            try:
                for link, target in (
                    (parent / "workspace-alias", workspace),
                    (workspace / "inside-alias", source),
                    (workspace / "outside-alias", outside),
                    (selected_subdir / "sibling-alias", source),
                    (selected_subdir / "outside-alias", outside),
                ):
                    result = subprocess.run(
                        ["cmd.exe", "/c", "mklink", "/J", str(link), str(target)],
                        capture_output=True, text=True, check=False,
                    )
                    if result.returncode:
                        self.skipTest("temporary junctions unavailable")
                    links.append(link)
                with os.scandir(workspace) as entries:
                    recurses_junction = next(entry for entry in entries if entry.name == "outside-alias").is_dir(
                        follow_symlinks=False,
                    )
                for selected in (workspace, parent / "workspace-alias"):
                    for permission in ("auto", "confirm"):
                        bounded = await self.scan(selected, permission, depth=3)
                        self.assertIn("outside-alias", bounded)
                        self.assertNotIn("external.txt", bounded)
                        self.assertIn("internal.txt", bounded)
                    full = await self.scan(selected, "full", depth=3)
                    self.assertEqual("external.txt" in full, recurses_junction)
                    self.assertIn("internal.txt", full)
                for permission in ("auto", "confirm", "full"):
                    nested = await self.scan(workspace, permission, path="selected", depth=3)
                    self.assertIn("sibling-alias", nested)
                    self.assertEqual("internal.txt" in nested, recurses_junction)
                    self.assertEqual("external.txt" in nested,
                                     permission == "full" and recurses_junction)
            finally:
                # Remove only the junction entries, before recursive cleanup
                # of the fixture parent. Never remove their target contents.
                for link in reversed(links):
                    if os.path.lexists(link):
                        os.rmdir(link)
                self.assertTrue((outside / "external.txt").exists())
                self.assertTrue((source / "internal.txt").exists())
