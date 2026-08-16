"""Bounds tests for the shared whole-file edit reader."""

from __future__ import annotations

import asyncio
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter.agent_tools import base
from Cozter.agent_tools.base import read_text_for_edit
from Cozter.agent_tools.builtin.edit_file import EditFileTool
from Cozter.agent_tools.builtin.multi_edit import MultiEditTool


class EditFileReadLimitTests(unittest.TestCase):
    def test_at_limit_utf8_crlf_content_keeps_existing_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "note.txt")
            raw = "one\r\ntwo café\r\n".encode("utf-8")
            with open(path, "wb") as f:
                f.write(raw)

            with mock.patch.object(base, "_MAX_EDIT_FILE_BYTES", len(raw)):
                self.assertEqual(
                    read_text_for_edit(path),
                    ("one\ntwo café\n", True),
                )

    def test_growth_after_size_check_still_uses_bounded_read(self) -> None:
        limit = 8

        class GrowingFile:
            read_size: int | None = None

            def __enter__(self) -> GrowingFile:
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def fileno(self) -> int:
                return 1

            def read(self, size: int = -1) -> bytes:
                self.read_size = size
                # Simulate bytes appended after fstat() reported the limit.
                return b"x" * (limit + 1)

        growing_file = GrowingFile()
        with (
            mock.patch("builtins.open", return_value=growing_file),
            mock.patch.object(
                base.os,
                "fstat",
                return_value=SimpleNamespace(st_size=limit),
            ),
            mock.patch.object(base, "_MAX_EDIT_FILE_BYTES", limit),
        ):
            result = read_text_for_edit("growing.txt")

        self.assertEqual(growing_file.read_size, limit + 1)
        self.assertEqual(
            result,
            "file is too large to edit (maximum 8 bytes)",
        )

    def test_edit_tools_reject_oversized_files_without_writing(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "note.txt")
                original = b"alpha beta"
                with open(path, "wb") as f:
                    f.write(original)

                with mock.patch.object(base, "_MAX_EDIT_FILE_BYTES", 8):
                    edit_result = await EditFileTool().run(
                        tmp,
                        {
                            "path": "note.txt",
                            "old_string": "alpha",
                            "new_string": "gamma",
                        },
                    )
                    multi_result = await MultiEditTool().run(
                        tmp,
                        {
                            "path": "note.txt",
                            "edits": [{
                                "old_string": "alpha",
                                "new_string": "gamma",
                            }],
                        },
                    )

                expected = (
                    "Error: file is too large to edit (maximum 8 bytes)"
                )
                self.assertEqual(edit_result, expected)
                self.assertEqual(multi_result, expected)
                with open(path, "rb") as f:
                    self.assertEqual(f.read(), original)

        asyncio.run(run())
