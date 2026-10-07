"""Focused bounds and matcher regressions for ``apply_patch``."""

from __future__ import annotations

import asyncio
import difflib
import os
import tempfile
import unittest
from unittest import mock

from Cozter.agent_tools.builtin import apply_patch as apply_patch_module
from Cozter.agent_tools.builtin.apply_patch import (
    ApplyPatchTool,
    _FileLimitError,
    _Hunk,
    _locate,
    _read_file_lines,
)


class _NoSliceList(list[str]):
    """A sequence that exposes accidental window-slice matching."""

    def __getitem__(self, index: int | slice) -> str | list[str]:
        if isinstance(index, slice):
            raise AssertionError("_locate must not allocate a line slice")
        return super().__getitem__(index)


class ApplyPatchLimitsTests(unittest.TestCase):
    def _run(self, workspace: str, patch: str) -> str:
        return asyncio.run(ApplyPatchTool().run(workspace, {"patch": patch}))

    def test_zero_context_insertions_use_unified_diff_line_positions(self) -> None:
        original = "one\ntwo\nthree\n"
        for expected in (
            "new\none\ntwo\nthree\n",
            "one\nnew\ntwo\nthree\n",
            "one\ntwo\nthree\nnew\n",
            "one\nfirst\ntwo\nsecond\nthree\n",
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "data.txt")
                with open(path, "w", encoding="utf-8") as file_handle:
                    file_handle.write(original)
                patch = "".join(difflib.unified_diff(
                    original.splitlines(keepends=True),
                    expected.splitlines(keepends=True),
                    fromfile="a/data.txt", tofile="b/data.txt", n=0,
                ))

                self.assertIn("applied", self._run(tmp, patch))
                with open(path, encoding="utf-8") as file_handle:
                    self.assertEqual(file_handle.read(), expected)

    def test_later_hunks_target_the_correct_duplicate_after_size_changes(self) -> None:
        for original, expected in (
            ("header\nsame\ngap\nsame\n", "header\ninserted\nsame\ngap\nchanged\n"),
            ("discard\nsame\ngap\nsame\nsame\n", "same\ngap\nchanged\nsame\n"),
        ):
            with self.subTest(original=original), tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "data.txt")
                with open(path, "w", encoding="utf-8") as file_handle:
                    file_handle.write(original)
                patch = "".join(difflib.unified_diff(
                    original.splitlines(keepends=True),
                    expected.splitlines(keepends=True),
                    fromfile="a/data.txt", tofile="b/data.txt", n=0,
                ))

                self.assertIn("applied", self._run(tmp, patch))
                with open(path, encoding="utf-8") as file_handle:
                    self.assertEqual(file_handle.read(), expected)

    def test_inserting_into_an_empty_existing_file_adds_no_phantom_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.txt")
            with open(path, "w", encoding="utf-8"):
                pass
            out = self._run(tmp, (
                "--- a/data.txt\n+++ b/data.txt\n"
                "@@ -0,0 +1 @@\n+one\n"
            ))

            self.assertIn("applied", out)
            with open(path, encoding="utf-8") as file_handle:
                # The hunk adds a terminated line even though the old file is empty.
                self.assertEqual(file_handle.read(), "one\n")

    def test_incomplete_dev_null_header_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run(tmp, "--- /dev/null\n")
        self.assertIn("missing its +++ counterpart", out)

    def test_new_header_without_old_header_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run(tmp, "+++ b/data.txt\n@@ -0,0 +1 @@\n+one\n")
            self.assertFalse(os.path.exists(os.path.join(tmp, "data.txt")))
        self.assertIn("missing its --- counterpart", out)

    def test_rename_patch_does_not_modify_an_unrelated_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for filename in ("old.txt", "new.txt"):
                with open(os.path.join(tmp, filename), "w", encoding="utf-8") as file_handle:
                    file_handle.write("same\n")
            out = self._run(tmp, (
                "--- a/old.txt\n+++ b/new.txt\n@@ -1 +1 @@\n-same\n+changed\n"
            ))
            for filename in ("old.txt", "new.txt"):
                with open(os.path.join(tmp, filename), encoding="utf-8") as file_handle:
                    self.assertEqual(file_handle.read(), "same\n")
        self.assertIn("renaming files is not supported", out)

    def test_patch_byte_limit_is_reported_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            apply_patch_module, "_MAX_PATCH_BYTES", 16,
        ):
            out = self._run(tmp, "x" * 17)

        self.assertIn("patch exceeds the 16-byte limit", out)
        self.assertIn("split it into smaller patches", out)

    def test_patch_line_limit_is_reported_before_splitlines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            apply_patch_module, "_MAX_PATCH_LINES", 2,
        ):
            out = self._run(tmp, "one\ntwo\nthree")

        self.assertIn("patch exceeds the 2-line limit", out)

    def test_target_byte_limit_leaves_file_unchanged(self) -> None:
        patch = (
            "--- a/data.txt\n+++ b/data.txt\n"
            "@@ -1 +1 @@\n-old value\n+new value\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.txt")
            with open(path, "w", encoding="utf-8") as file_handle:
                file_handle.write("old value\n")

            with mock.patch.object(
                apply_patch_module, "_MAX_FILE_BYTES", 8,
            ):
                out = self._run(tmp, patch)

            with open(path, encoding="utf-8") as file_handle:
                self.assertEqual(file_handle.read(), "old value\n")

        self.assertIn("file exceeds the 8-byte limit", out)

    def test_target_byte_limit_counts_restored_crlf_bytes(self) -> None:
        patch = (
            "--- a/data.txt\n+++ b/data.txt\n"
            "@@ -1 +1 @@\n-old\n+é\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.txt")
            with open(path, "wb") as file_handle:
                file_handle.write(b"old\r\n")

            with mock.patch.object(
                apply_patch_module, "_MAX_FILE_BYTES", 3,
            ):
                out = self._run(tmp, patch)

            with open(path, "rb") as file_handle:
                self.assertEqual(file_handle.read(), b"old\r\n")

        self.assertIn("file exceeds the 3-byte limit", out)

    def test_target_line_limit_leaves_file_unchanged(self) -> None:
        patch = (
            "--- a/data.txt\n+++ b/data.txt\n"
            "@@ -1,3 +1,3 @@\n one\n-two\n+changed\n three\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.txt")
            with open(path, "w", encoding="utf-8") as file_handle:
                file_handle.write("one\ntwo\nthree\n")

            with mock.patch.object(
                apply_patch_module, "_MAX_FILE_LINES", 2,
            ):
                out = self._run(tmp, patch)

            with open(path, encoding="utf-8") as file_handle:
                self.assertEqual(file_handle.read(), "one\ntwo\nthree\n")

        self.assertIn("file exceeds the 2-line limit", out)

    def test_read_stays_bounded_if_file_grows_after_stat(self) -> None:
        reader = mock.mock_open(read_data=b"x" * 9)
        stat_result = mock.Mock(st_size=0)
        with (
            mock.patch.object(apply_patch_module, "_MAX_FILE_BYTES", 8),
            mock.patch.object(apply_patch_module.os, "stat", return_value=stat_result),
            mock.patch("builtins.open", reader),self.assertRaises(_FileLimitError)
        ):
            _read_file_lines("grown-after-stat.txt")

        reader().read.assert_called_once_with(9)

    def test_locate_is_slice_free_and_keeps_hint_first_behavior(self) -> None:
        hunk = _Hunk(start=2_001)
        hunk.old = ["a"] * 2_000 + ["b"]
        lines = _NoSliceList(["a"] * 4_000 + ["b"])

        self.assertEqual(_locate(lines, hunk), 2_000)

        hunk = _Hunk(start=3)
        hunk.old = ["exact"]
        self.assertEqual(
            _locate(_NoSliceList(["exact", "other", "exact"]), hunk),
            2,
        )

        hunk = _Hunk(start=3)
        hunk.old = ["match"]
        self.assertEqual(
            _locate(_NoSliceList(["match ", "other", "match\t"]), hunk),
            2,
        )

    def test_locate_keeps_exact_matches_ahead_of_fuzzy_hint(self) -> None:
        hunk = _Hunk(start=1)
        hunk.old = ["match"]

        self.assertEqual(_locate(_NoSliceList(["match ", "match"]), hunk), 1)


if __name__ == "__main__":
    unittest.main()
