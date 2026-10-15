"""Discovery pages preserve complete matches and an exact continuation cursor."""

import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

from Cozter import agent_tools
from Cozter.agent_tools.builtin.glob import GlobTool
from Cozter.agent_tools.builtin.grep import GrepTool
from Cozter.agent_tools.discovery import render_discovery_page


def _next_offset(text: str) -> int | None:
    match = re.search(r"\boffset=(\d+)\b", text)
    return int(match.group(1)) if match else None


class DiscoveryPageRenderingTests(unittest.TestCase):
    def test_exact_cap_has_no_false_continuation(self) -> None:
        rows = ["a.txt", "b.txt", "c.txt"]
        result = render_discovery_page(rows, offset=0, max_results=3)
        self.assertEqual(result, "\n".join(rows))
        self.assertIsNone(_next_offset(result))

    def test_one_extra_match_provides_exact_lookahead(self) -> None:
        rows = ["a.txt", "b.txt", "c.txt", "d.txt"]
        result = render_discovery_page(rows, offset=7, max_results=3)
        self.assertTrue(result.startswith("\n".join(rows[:3])), result)
        self.assertNotIn(rows[3], result)
        self.assertEqual(_next_offset(result), 10)
        self.assertIn("files change", result.lower())

    def test_character_budget_advances_only_complete_emitted_matches(self) -> None:
        rows = [f"row-{index}:" + chr(65 + index) * 9_000 for index in range(8)]
        result = render_discovery_page(rows, offset=13, max_results=7)
        next_offset = _next_offset(result)
        self.assertIsNotNone(next_offset)
        assert next_offset is not None
        emitted = next_offset - 13
        self.assertGreater(emitted, 0)
        self.assertLess(emitted, 7)
        self.assertLessEqual(len(result), 32_768)
        self.assertTrue(result.startswith("\n".join(rows[:emitted])), result[:100])
        for row in rows[:emitted]:
            self.assertIn(row, result)
        self.assertNotIn(f"row-{emitted}:", result)

    def test_embedded_newlines_do_not_advance_cursor_as_extra_matches(self) -> None:
        rows = ["a\ncontinued\nagain", "b\ncontinued", "c"]
        result = render_discovery_page(rows, offset=11, max_results=2)
        self.assertTrue(result.startswith("\n".join(rows[:2])), result)
        self.assertEqual(_next_offset(result), 13)

    def test_single_oversized_row_requires_narrowing_without_false_cursor(self) -> None:
        result = render_discovery_page(["x" * 40_000, "later.txt"], offset=4, max_results=1)
        self.assertLessEqual(len(result), 32_768)
        self.assertIn("narrow", result.lower())
        self.assertIsNone(_next_offset(result))


class DiscoveryToolPagingTests(unittest.IsolatedAsyncioTestCase):
    async def test_glob_pages_cover_unchanged_tree_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            expected: set[str] = set()
            # Creation order differs from lexical traversal order.
            for directory in ("z", "a", "m"):
                for number in (3, 1, 2):
                    relative = os.path.join(directory, f"file-{number}.txt")
                    path = Path(tmp, relative)
                    path.parent.mkdir(exist_ok=True)
                    path.write_text("hit\n", encoding="utf-8")
                    expected.add(relative.replace(os.sep, "/"))
            tool = GlobTool()
            collected: list[str] = []
            offset = 0
            first = await tool.run(tmp, {"pattern": "**/*.txt", "max_results": 4})
            self.assertEqual(
                first,
                await tool.run(tmp, {"pattern": "**/*.txt", "max_results": 4}),
            )
            for _ in range(10):
                result = await tool.run(tmp, {
                    "pattern": "**/*.txt", "max_results": 4, "offset": offset,
                })
                rows = [line for line in result.splitlines() if line.endswith(".txt")]
                collected.extend(rows)
                continuation = _next_offset(result)
                if continuation is None:
                    break
                self.assertEqual(continuation, offset + len(rows))
                offset = continuation
            else:
                self.fail("Glob pagination did not reach the unchanged tree's end")
            self.assertEqual(set(collected), expected)
            self.assertEqual(len(collected), len(expected))
            self.assertEqual(collected, sorted(expected))

    async def test_grep_pages_cover_unchanged_tree_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            expected: set[str] = set()
            for filename in ("z.txt", "a.txt", "m.txt"):
                Path(tmp, filename).write_text("hit one\nhit two\nhit three\n", encoding="utf-8")
                expected.update(f"{filename}:{index}: hit {word}" for index, word in (
                    (1, "one"), (2, "two"), (3, "three"),
                ))
            tool = GrepTool()
            collected: list[str] = []
            offset = 0
            for _ in range(10):
                result = await tool.run(tmp, {
                    "pattern": "hit", "max_results": 4, "offset": offset,
                })
                rows = [line for line in result.splitlines() if re.match(r"[azm]\.txt:\d+: hit ", line)]
                collected.extend(rows)
                continuation = _next_offset(result)
                if continuation is None:
                    break
                self.assertEqual(continuation, offset + len(rows))
                offset = continuation
            else:
                self.fail("Grep pagination did not reach the unchanged tree's end")
            self.assertEqual(set(collected), expected)
            self.assertEqual(len(collected), len(expected))
            self.assertEqual(collected, sorted(expected))

    async def test_exact_cap_and_past_end_are_truthful_for_both_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for number in range(3):
                Path(tmp, f"{number}.txt").write_text("hit\n", encoding="utf-8")
            for tool, args, noun in (
                (GlobTool(), {"pattern": "*.txt"}, "files"),
                (GrepTool(), {"pattern": "hit"}, "matches"),
            ):
                with self.subTest(tool=tool.name):
                    exact = await tool.run(tmp, {**args, "max_results": 3})
                    self.assertIsNone(_next_offset(exact), exact)
                    beyond = await tool.run(tmp, {**args, "offset": 3, "max_results": 3})
                    self.assertIn(f"No more {noun}", beyond)
                    self.assertIn("offset 3", beyond)
                    self.assertIsNone(_next_offset(beyond))


class DiscoveryDispatchLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_builtin_discovery_preserves_large_page(self) -> None:
        page = "entry.txt\n" * 600
        for tool in (GlobTool(), GrepTool()):
            with self.subTest(tool=tool.name):
                with mock.patch.dict(agent_tools._BY_NAME, {tool.name: tool}), mock.patch.object(
                    tool, "run", new=mock.AsyncMock(return_value=page),
                ):
                    result = await agent_tools.execute_tool(tool.name, {}, ".", "full", lambda _event: None)
                self.assertEqual(result, page)
                self.assertGreater(len(result), 4_000)

    async def test_colliding_plugin_keeps_normal_output_cap(self) -> None:
        page = "entry.txt\n" * 600
        for tool in (GlobTool(), GrepTool()):
            tool.is_plugin = True
            with self.subTest(tool=tool.name):
                with mock.patch.dict(agent_tools._BY_NAME, {tool.name: tool}), mock.patch.object(
                    tool, "run", new=mock.AsyncMock(return_value=page),
                ):
                    result = await agent_tools.execute_tool(tool.name, {}, ".", "full", lambda _event: None)
                self.assertLessEqual(len(result), 4_000)
                self.assertIn("truncated", result)


if __name__ == "__main__":
    unittest.main()
