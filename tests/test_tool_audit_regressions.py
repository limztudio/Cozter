"""Behavioral regressions found while auditing the complete tool surface."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from Cozter.agent_tools import base as tool_base
from Cozter.agent_tools.builtin.apply_patch import ApplyPatchTool
from Cozter.agent_tools.plugins.git_info import GitInfoTool
from Cozter.agent_tools.plugins.memory import MemoryTool, _OUTPUT_BUDGET, _fit_output
from Cozter.agent_tools.plugins.notes import NotesTool


class ToolAuditRegressionTests(unittest.TestCase):
    def test_public_http_session_uses_the_documented_one_hour_cap(self) -> None:
        async def inspect_timeout() -> None:
            async with tool_base.open_public_http_session() as session:
                self.assertEqual(session.timeout.total, 3600)

        asyncio.run(inspect_timeout())

    @unittest.skipUnless(os.name == "nt", "Windows paths are case-insensitive")
    def test_bounded_workspace_accepts_different_absolute_path_case(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            target = os.path.join(workspace, "new.txt").swapcase()
            resolved = tool_base.resolve_workspace_entry(workspace, target)
            self.assertEqual(os.path.normcase(resolved), os.path.normcase(target))

    def test_deletion_patch_removes_symlink_entry_and_preserves_target(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            target = Path(workspace, "target.txt")
            target.write_text("keep this target\n", encoding="utf-8")
            link = Path(workspace, "link.txt")
            try:
                link.symlink_to(target)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            result = asyncio.run(ApplyPatchTool().run(workspace, {"patch": (
                "--- a/link.txt\n+++ /dev/null\n"
                "@@ -1 +0,0 @@\n-keep this target\n"
            )}))
            self.assertIn("deleted", result)
            self.assertFalse(os.path.lexists(link))
            self.assertEqual(target.read_text(encoding="utf-8"), "keep this target\n")

    def test_notes_clear_removes_symlink_entry_and_preserves_target(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            target = Path(workspace, "target.txt")
            target.write_text("keep this target\n", encoding="utf-8")
            link = Path(workspace, ".cozter", "notes.md")
            link.parent.mkdir()
            try:
                link.symlink_to(target)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            result = asyncio.run(NotesTool().run(workspace, {"action": "clear"}))
            self.assertEqual(result, "Notes cleared.")
            self.assertFalse(os.path.lexists(link))
            self.assertEqual(target.read_text(encoding="utf-8"), "keep this target\n")

    def test_memory_output_reserves_the_complete_omission_footer(self) -> None:
        lines = ["a" * 500 for _ in range(8)]
        result = _fit_output(lines, "Found matching messages:")
        self.assertLessEqual(len(result), _OUTPUT_BUDGET)
        self.assertIn("PARTIAL + remainder", result)

    def test_memory_oversized_header_still_respects_output_budget(self) -> None:
        result = _fit_output(["matching message"], "query " + "x" * _OUTPUT_BUDGET)
        self.assertLessEqual(len(result), _OUTPUT_BUDGET)
        self.assertIn("PARTIAL + remainder", result)

    def test_memory_tail_read_keeps_newest_messages_when_result_budget_is_hit(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            sessions = Path(workspace, ".cozter", "sessions")
            sessions.mkdir(parents=True)
            session = {
                "id": "audit-session",
                "name": "Audit session",
                "messages": [
                    {"role": "user", "content": f"message-{index:02d} " + "x" * 290}
                    for index in range(20)
                ],
            }
            (sessions / "audit-session.json").write_text(json.dumps(session), encoding="utf-8")
            result = asyncio.run(MemoryTool().run(workspace, {
                "action": "read", "session": "last", "limit": 20,
            }))
            self.assertLessEqual(len(result), _OUTPUT_BUDGET)
            self.assertIn("message-19", result)
            self.assertNotIn("message-00", result)
            self.assertIn("PARTIAL + remainder", result)

    @unittest.skipUnless(shutil.which("git"), "git not available")
    def test_git_blame_does_not_execute_configured_textconv(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            def git(*args: str) -> None:
                subprocess.run(["git", "-C", workspace, *args], capture_output=True, check=True)

            git("init", "-q")
            git("config", "user.name", "Cozter Tests")
            git("config", "user.email", "test@example.com")
            Path(workspace, "a.txt").write_text("original line\n", encoding="utf-8")
            Path(workspace, ".gitattributes").write_text("a.txt diff=unsafe\n", encoding="utf-8")
            git("add", ".")
            git("commit", "-qm", "initial")
            marker = Path(workspace, "converter-executed")
            helper = Path(workspace, ".git", "converter.py")
            helper.write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('executed')\n"
                "print('converted line')\n",
                encoding="utf-8",
            )
            git("config", "diff.unsafe.textconv", f'"{sys.executable}" "{helper}"')
            for line in (None, 1):
                with self.subTest(line=line):
                    args: dict = {"action": "blame", "path": "a.txt"}
                    if line is not None:
                        args["line"] = line
                    result = asyncio.run(GitInfoTool().run(workspace, args))
                    self.assertIn("original line", result)
                    self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
