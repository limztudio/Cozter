"""Behavioral regressions found during the second complete tools sweep."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest
from unittest import mock

from Cozter.agent_tools.builtin import move_file as move_file_module
from Cozter.agent_tools.builtin.move_file import MoveFileTool
from Cozter.agent_tools.plugins.git_info import GitInfoTool
from Cozter.agent_tools.plugins.git_ops import GitOpsTool
from Cozter.agent_tools.plugins.notes import (
    _NOTES_KEEP_BYTES,
    _NOTES_MAX_BYTES,
    _fit_entries,
)


class NotesTrimSweepTests(unittest.TestCase):
    def test_trim_preserves_exact_newest_entries_and_separator_newlines(self) -> None:
        entries = [
            f"## 2026-10-12 00:{index:02d}:00\nentry {index} " + "한" * 1000 + "\n\n"
            for index in range(25)
        ]
        self.assertGreater(len("".join(entries).encode("utf-8")), _NOTES_MAX_BYTES)
        expected = []
        used = 0
        for entry in reversed(entries):
            size = len(entry.encode("utf-8"))
            if used + size > _NOTES_KEEP_BYTES:
                break
            expected.append(entry)
            used += size
        actual = _fit_entries(entries)
        self.assertEqual(actual, "".join(reversed(expected)))
        self.assertLessEqual(len(actual.encode("utf-8")), _NOTES_KEEP_BYTES)

    def test_trim_retains_a_missing_final_newline(self) -> None:
        old = "## 2020-01-01 00:00:00\n" + "x" * _NOTES_MAX_BYTES + "\n\n"
        newest = "## 2026-10-12 00:00:00\nretain this exact text"
        self.assertEqual(_fit_entries([old, newest]), newest)


@unittest.skipUnless(shutil.which("git"), "git not available")
class GitSymlinkSweepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace = self.directory.name
        self.git("init", "-q")
        self.git("config", "user.name", "Cozter Tests")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "core.symlinks", "true")
        Path(self.workspace, "first.txt").write_text("first target\n", encoding="utf-8")
        Path(self.workspace, "second.txt").write_text("second target\n", encoding="utf-8")
        self.link = Path(self.workspace, "link.txt")
        self.make_link("first.txt")

    def git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", self.workspace, *args],
            check=True, capture_output=True, text=True,
        )
        return result.stdout

    def make_link(self, target: str) -> None:
        try:
            self.link.symlink_to(target)
        except (NotImplementedError, OSError) as exc:
            self.skipTest(f"symlinks unavailable: {exc}")

    def test_add_stages_the_requested_symlink_and_leaves_target_unstaged(self) -> None:
        result = asyncio.run(GitOpsTool().run(self.workspace, {
            "action": "add", "path": "link.txt",
        }))
        self.assertEqual(result, "OK")
        self.assertEqual(self.git("diff", "--cached", "--name-only").strip(), "link.txt")
        self.assertTrue(self.git("ls-files", "--stage", "link.txt").startswith("120000 "))

    def test_diff_describes_the_requested_symlink_change(self) -> None:
        self.git("add", ".")
        self.git("commit", "-qm", "initial")
        self.link.unlink()
        self.make_link("second.txt")
        result = asyncio.run(GitInfoTool().run(self.workspace, {
            "action": "diff", "path": "link.txt", "patch": True,
        }))
        self.assertIn("diff --git a/link.txt b/link.txt", result)
        self.assertIn("-first.txt", result)
        self.assertIn("+second.txt", result)

    def test_external_symlink_is_still_refused_in_auto_mode(self) -> None:
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside, "outside.txt")
            target.write_text("outside\n", encoding="utf-8")
            self.link.unlink()
            self.make_link(str(target))
            for tool, args in (
                (GitOpsTool(), {"action": "add", "path": "link.txt"}),
                (GitInfoTool(), {"action": "diff", "path": "link.txt"}),
            ):
                with self.subTest(tool=tool.name):
                    result = asyncio.run(tool.run(self.workspace, args))
                    self.assertIn("path escapes workspace", result)
            self.assertEqual(self.git("ls-files"), "")


@unittest.skipUnless(os.name == "nt", "Windows read-only file behavior")
class WindowsMoveSweepTests(unittest.TestCase):
    def test_read_only_file_moves_without_leaving_a_duplicate_entry(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            source = Path(workspace, "source.txt")
            target = Path(workspace, "target.txt")
            source.write_text("preserve content\n", encoding="utf-8")
            source.chmod(stat.S_IREAD)
            try:
                result = asyncio.run(MoveFileTool().run(workspace, {
                    "source": "source.txt", "destination": "target.txt",
                }))
                self.assertEqual(result, "Moved: source.txt -> target.txt")
                self.assertFalse(source.exists())
                self.assertEqual(target.read_text(encoding="utf-8"), "preserve content\n")
                self.assertFalse(target.stat().st_mode & stat.S_IWRITE)
            finally:
                for path in (source, target):
                    if path.exists():
                        path.chmod(stat.S_IWRITE)

    def test_read_only_file_move_preserves_a_raced_in_destination(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            source = Path(workspace, "source.txt")
            target = Path(workspace, "target.txt")
            source.write_text("original source\n", encoding="utf-8")
            source.chmod(stat.S_IREAD)
            try:
                def create_competitor(_target: str) -> None:
                    target.write_text("concurrent destination\n", encoding="utf-8")

                with mock.patch.object(
                    move_file_module, "ensure_parent_dir", side_effect=create_competitor,
                ):
                    result = asyncio.run(MoveFileTool().run(workspace, {
                        "source": "source.txt", "destination": "target.txt",
                    }))
                self.assertEqual(result, "Destination already exists: target.txt")
                self.assertEqual(source.read_text(encoding="utf-8"), "original source\n")
                self.assertEqual(
                    target.read_text(encoding="utf-8"), "concurrent destination\n",
                )
                self.assertFalse(source.stat().st_mode & stat.S_IWRITE)
            finally:
                source.chmod(stat.S_IWRITE)


if __name__ == "__main__":
    unittest.main()
