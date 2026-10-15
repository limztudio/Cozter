"""Bounded batch reads preserve individual path and range semantics."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from Cozter import agent_tools
from Cozter.agent_tools.base import tool_permission_scope
from Cozter.agent_tools.builtin import read_files


class ReadFilesTests(unittest.IsolatedAsyncioTestCase):
    async def test_results_follow_request_order_and_preserve_ranges(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.txt").write_text("first\nsecond\nthird\n", encoding="utf-8")
            (root / "b.txt").write_text("beta", encoding="utf-8")
            result = await read_files.ReadFilesTool().run(temp, {"files": [
                {"path": "b.txt"}, {"path": "a.txt", "offset": 1, "limit": 1},
                {"path": "absent.txt"},
            ]})
        self.assertLess(result.index("b.txt"), result.index("a.txt"))
        self.assertLess(result.index("a.txt"), result.index("absent.txt"))
        self.assertIn("beta", result)
        self.assertIn("second\n", result)
        self.assertNotIn("third", result)
        self.assertIn("File not found", result)

    async def test_workspace_escape_stays_blocked_in_auto_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "workspace").mkdir()
            (root / "secret.txt").write_text("private contents", encoding="utf-8")
            with tool_permission_scope("auto"):
                result = await read_files.ReadFilesTool().run(str(root / "workspace"), {
                    "files": [{"path": "../secret.txt"}],
                })
            with tool_permission_scope("full"):
                full = await read_files.ReadFilesTool().run(str(root / "workspace"), {
                    "files": [{"path": "../secret.txt"}],
                })
        self.assertIn("path escapes workspace", result)
        self.assertNotIn("private contents", result)
        self.assertIn("private contents", full)

    async def test_invalid_batches_do_not_begin_file_reads(self):
        invalid = [[], [{}], [{"path": ""}], [{"path": 42}], [{"path": "x"}] * 17]
        with mock.patch.object(read_files.ReadFileTool, "run", new_callable=mock.AsyncMock) as read:
            for files in invalid:
                result = await read_files.ReadFilesTool().run("/workspace", {"files": files})
                self.assertTrue(result.startswith("Error:"))
        read.assert_not_awaited()

    async def test_parallelism_and_aggregate_output_are_bounded(self):
        active = peak = 0

        async def read(_workspace, args):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.01)
                return args["path"] + ":" + "x" * 100_000
            finally:
                active -= 1

        with mock.patch.object(read_files.ReadFileTool, "run", side_effect=read):
            result = await read_files.ReadFilesTool().run("/workspace", {
                "files": [{"path": f"file-{i}.txt"} for i in range(16)],
            })
        self.assertEqual(peak, read_files._MAX_PARALLEL_READS)
        self.assertLessEqual(len(result), read_files._MAX_RESULT_CHARS)
        self.assertIn("truncated file preview", result)
        for index in range(16):
            self.assertIn(f"=== file-{index}.txt ===", result)

    async def test_confirm_dispatcher_exposes_batch_read_and_retains_batch_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.txt").write_text("a" * 7000, encoding="utf-8")
            (root / "b.txt").write_text("b" * 7000, encoding="utf-8")
            result = await agent_tools.execute_tool(
                "read_files", {"files": [{"path": "a.txt"}, {"path": "b.txt"}]},
                temp, "confirm", lambda event: None,
            )
        self.assertIn("read_files", agent_tools.READ_ONLY_TOOL_NAMES)
        self.assertGreater(len(result), 10_000)
        self.assertIn("=== b.txt ===", result)
