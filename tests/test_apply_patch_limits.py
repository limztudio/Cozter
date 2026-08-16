"""Focused bounds and matcher regressions for ``apply_patch``."""

from __future__ import annotations

import asyncio
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

    def test_patch_byte_limit_is_reported_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                apply_patch_module, "_MAX_PATCH_BYTES", 16,
            ):
                out = self._run(tmp, "x" * 17)

        self.assertIn("patch exceeds the 16-byte limit", out)
        self.assertIn("split it into smaller patches", out)

    def test_patch_line_limit_is_reported_before_splitlines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
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
            with open(path, "w", encoding="utf-8") as f:
                f.write("old value\n")

            with mock.patch.object(
                apply_patch_module, "_MAX_FILE_BYTES", 8,
            ):
                out = self._run(tmp, patch)

            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), "old value\n")

        self.assertIn("file exceeds the 8-byte limit", out)

    def test_target_line_limit_leaves_file_unchanged(self) -> None:
        patch = (
            "--- a/data.txt\n+++ b/data.txt\n"
            "@@ -1,3 +1,3 @@\n one\n-two\n+changed\n three\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("one\ntwo\nthree\n")

            with mock.patch.object(
                apply_patch_module, "_MAX_FILE_LINES", 2,
            ):
                out = self._run(tmp, patch)

            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), "one\ntwo\nthree\n")

        self.assertIn("file exceeds the 2-line limit", out)

    def test_read_stays_bounded_if_file_grows_after_stat(self) -> None:
        reader = mock.mock_open(read_data=b"x" * 9)
        stat_result = mock.Mock(st_size=0)
        with (
            mock.patch.object(apply_patch_module, "_MAX_FILE_BYTES", 8),
            mock.patch.object(apply_patch_module.os, "stat", return_value=stat_result),
            mock.patch("builtins.open", reader),
        ):
            with self.assertRaises(_FileLimitError):
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
