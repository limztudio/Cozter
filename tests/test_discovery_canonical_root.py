"""Bounded discovery reuses its canonical root while resolving every target."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from Cozter.agent_tools import base


def _scan(workspace, root=None, approval="auto", pattern="**/*"):
    with base.tool_permission_scope(approval):
        return list(base.iter_workspace_files(
            str(workspace), str(root or workspace), pattern, sort_entries=True,
        ))


def _legacy_bounded_selection(workspace, root=None, pattern="**/*"):
    # Full mode's matching, traversal, and lexical labels are unchanged. The
    # previous bounded branch filtered these same rows with is_path_within.
    canonical_root = os.path.realpath(workspace)
    return [row for row in _scan(workspace, root, "full", pattern)
            if base.is_path_within(row[0], canonical_root)]


class DiscoveryCanonicalRootTests(unittest.TestCase):
    def test_auto_confirm_and_full_keep_exact_rows_with_only_per_target_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            for index in range(40):
                path = workspace / f"src-{index % 4}" / f"file-{index:02}.py"
                path.parent.mkdir(exist_ok=True)
                path.touch()
            expected = _legacy_bounded_selection(workspace)
            for approval in ("auto", "confirm", "full"):
                with self.subTest(approval=approval):
                    with mock.patch.object(base.os.path, "realpath", wraps=os.path.realpath) as resolve:
                        actual = _scan(workspace, approval=approval)
                    self.assertEqual(actual, expected)
                    self.assertEqual(resolve.call_count, 2 if approval == "full" else 2 + len(expected))

    def test_alias_sibling_error_and_case_boundaries_match_legacy_without_link_privilege(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            workspace = parent / "workspace"
            workspace.mkdir()
            target = workspace / "target.py"
            target.touch()
            sibling = parent / "workspace-sibling"
            sibling.mkdir()
            outside = sibling / "outside.py"
            outside.touch()
            aliases = {}
            for name, destination in (
                ("inside.py", str(target)), ("outside.py", str(outside)),
                ("case.py", str(target).swapcase()),
            ):
                path = workspace / name
                path.touch()
                aliases[str(path)] = destination
            failed = workspace / "failed.py"
            invalid = workspace / "invalid.py"
            failed.touch()
            invalid.touch()
            if sys.platform == "win32":
                drive = "Z:" if os.path.splitdrive(str(workspace))[0].upper() != "Z:" else "Y:"
                path = workspace / "other-drive.py"
                path.touch()
                aliases[str(path)] = drive + r"\outside\target.py"
            original = os.path.realpath

            def resolve(path, *args, **kwargs):
                value = os.fspath(path)
                if value == str(failed):
                    raise OSError("fixture resolution failure")
                if value == str(invalid):
                    raise ValueError("fixture invalid resolution")
                return aliases.get(value, original(path, *args, **kwargs))

            with mock.patch.object(base.os.path, "realpath", resolve):
                expected = _legacy_bounded_selection(workspace)
                self.assertEqual(_scan(workspace), expected)
                self.assertEqual(_scan(workspace, approval="confirm"), expected)
                self.assertIn("inside.py", {row[1] for row in expected})
                self.assertNotIn("outside.py", {row[1] for row in expected})
                self.assertNotIn("failed.py", {row[1] for row in expected})
                self.assertNotIn("invalid.py", {row[1] for row in expected})
                # Full mode bypasses canonical target checks, including all
                # broken/escaping aliases, and retains their lexical labels.
                full = _scan(workspace, approval="full")
                self.assertEqual({row[1] for row in full}, {path.name for path in workspace.iterdir()})

    def test_root_retarget_cannot_expand_authority_during_a_scan(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            workspace = parent / "workspace"
            workspace.mkdir()
            path = workspace / "candidate.py"
            path.touch()
            outside = parent / "outside"
            outside.mkdir()
            escaped = outside / "candidate.py"
            escaped.touch()
            root_calls = 0
            original = os.path.realpath

            def resolve(value, *args, **kwargs):
                nonlocal root_calls
                if os.fspath(value) == str(workspace):
                    root_calls += 1
                    return str(workspace) if root_calls <= 2 else str(outside)
                if os.fspath(value) == str(path):
                    return str(escaped)
                return original(value, *args, **kwargs)

            with mock.patch.object(base.os.path, "realpath", resolve):
                self.assertEqual(_scan(workspace), [])
            self.assertEqual(root_calls, 2)

    @unittest.skipUnless(sys.platform == "win32", "Windows junction behavior")
    def test_actual_root_and_nested_junctions_keep_bounded_targets_and_full_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            workspace = parent / "workspace"
            source = workspace / "src"
            source.mkdir(parents=True)
            (source / "inside.py").touch()
            outside = parent / "outside"
            outside.mkdir()
            (outside / "outside.py").touch()
            for link, destination in (
                (parent / "root-alias", workspace),
                (workspace / "inside-alias", source),
                (workspace / "outside-alias", outside),
            ):
                result = subprocess.run(
                    ["cmd.exe", "/c", "mklink", "/J", str(link), str(destination)],
                    capture_output=True, text=True, check=False,
                )
                if result.returncode:
                    self.skipTest("temporary directory junctions unavailable")
            for selected in (workspace, parent / "root-alias"):
                expected = _legacy_bounded_selection(selected)
                self.assertEqual(_scan(selected), expected)
                self.assertEqual(_scan(selected, approval="confirm"), expected)
                self.assertTrue(all("outside.py" not in row[1] for row in expected))
                full = _scan(selected, approval="full")
                # Python's junction traversal varies; retain exactly the full
                # mode rows which the current walker exposes, without changing
                # canonical target paths into the displayed lexical aliases.
                for _path, display, relative in full:
                    self.assertEqual(display, relative)
