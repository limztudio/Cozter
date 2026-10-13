"""A corrupt memory file must not hide the other durable records."""

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from Cozter.agent_tools.plugins.memory import MemoryTool
from Cozter.tests.test_state_integrity import INVALID_JSON_VALUES


class MemoryCorruptStateTests(unittest.TestCase):
    def test_deeply_nested_session_is_skipped_and_other_memory_remains_visible(self) -> None:
        for raw in INVALID_JSON_VALUES:
            with self.subTest(raw_length=len(raw)), tempfile.TemporaryDirectory() as workspace:
                sessions = Path(workspace, ".cozter", "sessions")
                sessions.mkdir(parents=True)
                (sessions / "corrupt.json").write_text(raw, encoding="utf-8")
                (sessions / "valid.json").write_text(json.dumps({
                    "id": "valid", "name": "Surviving memory", "summary": "unique saved fact",
                }), encoding="utf-8")
                tool = MemoryTool()
                listed = asyncio.run(tool.run(workspace, {"action": "list"}))
                self.assertIn("Surviving memory", listed)
                self.assertNotIn("corrupt", listed)
                searched = asyncio.run(tool.run(workspace, {"action": "search", "query": "unique"}))
                self.assertIn("unique saved fact", searched)

    def test_deeply_nested_colony_does_not_abort_session_search_or_listing(self) -> None:
        for raw in INVALID_JSON_VALUES:
            with self.subTest(raw_length=len(raw)), tempfile.TemporaryDirectory() as workspace:
                sessions = Path(workspace, ".cozter", "sessions")
                sessions.mkdir(parents=True)
                (sessions / "valid.json").write_text(json.dumps({
                    "id": "valid", "name": "Surviving memory", "summary": "unique saved fact",
                }), encoding="utf-8")
                Path(workspace, ".cozter", "colony.json").write_text(raw, encoding="utf-8")
                tool = MemoryTool()
                listed = asyncio.run(tool.run(workspace, {"action": "list"}))
                self.assertIn("Surviving memory", listed)
                self.assertIn("Colony: 0", listed)
                searched = asyncio.run(tool.run(workspace, {"action": "search", "query": "unique"}))
                self.assertIn("unique saved fact", searched)


if __name__ == "__main__":
    unittest.main()
