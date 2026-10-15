"""Character cursors recover all file text through the public dispatcher."""

import re
from pathlib import Path
import tempfile
import unittest

from Cozter.agent_tools import execute_tool


class ReadFileCharacterCursorTests(unittest.IsolatedAsyncioTestCase):
    async def reconstruct(self, workspace, args, *, batch=False):
        parts = []
        position = args.get("char_offset", 0)
        for _ in range(100):
            request = {**args, "char_offset": position}
            result = await execute_tool(
                "read_files" if batch else "read_file",
                {"files": [request]} if batch else request,
                workspace, "confirm", lambda _event: None,
            )
            self.assertLessEqual(len(result), 64 * 1024 if batch else 32 * 1024)
            if batch:
                result = result.removeprefix(f"=== {args['path']} ===\n")
            if "\n… [truncated file preview" not in result:
                parts.append(result)
                return "".join(parts)
            text, marker = result.rsplit("\n… [truncated file preview", 1)
            cursor = re.search(r"offset=(\d+), char_offset=(\d+)", marker)
            self.assertIsNotNone(cursor, result[-500:])
            self.assertEqual(int(cursor[1]), args.get("offset", 0))
            next_position = int(cursor[2])
            self.assertEqual(next_position, position + len(text))
            self.assertGreater(next_position, position)
            parts.append(text)
            position = next_position
        self.fail("character cursor did not reach the end of the requested range")

    async def test_dispatcher_reconstructs_long_single_line_with_multibyte_characters(self):
        with tempfile.TemporaryDirectory() as directory:
            text = "한🌐abcd" * 15_000
            Path(directory, "long.txt").write_text(text, encoding="utf-8")
            for batch in (False, True):
                with self.subTest(batch=batch):
                    self.assertEqual(await self.reconstruct(directory, {"path": "long.txt"}, batch=batch), text)

    async def test_line_offset_and_limit_keep_the_same_meaning_across_char_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            selected = "first" * 9_000 + "\n" + "second" * 9_000 + "\n"
            Path(directory, "range.txt").write_text("skip\n" + selected + "excluded\n", encoding="utf-8")
            for batch in (False, True):
                with self.subTest(batch=batch):
                    actual = await self.reconstruct(directory, {
                        "path": "range.txt", "offset": 1, "limit": 2,
                    }, batch=batch)
                    self.assertEqual(actual, selected)

    async def test_cursor_validation_and_scan_bound_are_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "data.txt").write_text("line\n", encoding="utf-8")
            for value in (True, "1", 1.5, -1, 16 * 1024 * 1024 + 1):
                with self.subTest(value=value):
                    result = await execute_tool("read_file", {
                        "path": "data.txt", "char_offset": value,
                    }, directory, "confirm", lambda _event: None)
                    self.assertTrue(result.startswith("Error:"), result)

    async def test_every_file_in_a_full_batch_keeps_a_working_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            content = "한🌐" * 25_000
            files = [{"path": f"{index}.txt"} for index in range(16)]
            for item in files:
                Path(directory, item["path"]).write_text(content, encoding="utf-8")
            result = await execute_tool("read_files", {"files": files}, directory,
                                        "confirm", lambda _event: None)
            sections = re.split(r"(?:\n\n)?=== \d+\.txt ===\n", result)[1:]
            self.assertEqual(len(sections), len(files))
            for item, section in zip(files, sections):
                prefix, marker = section.rsplit("\n… [truncated file preview", 1)
                position = int(re.search(r"char_offset=(\d+)", marker)[1])
                self.assertEqual(position, len(prefix))
                rest = await self.reconstruct(directory, {**item, "char_offset": position})
                self.assertEqual(prefix + rest, content)
