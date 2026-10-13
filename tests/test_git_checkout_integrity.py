"""Branch checkout must never fall back to restoring a same-named file."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from Cozter.agent_tools.plugins.git_ops import GitOpsTool


@unittest.skipUnless(shutil.which("git"), "git not available")
class GitCheckoutIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace = self.directory.name
        self.environment = {
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
        }
        self.git("init", "-q", "--initial-branch=audit-main")
        self.git("config", "user.name", "Cozter Tests")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "core.hooksPath", os.devnull)
        Path(self.workspace, "missing-branch").write_text("committed text\n", encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-qm", "initial fixture")

    def git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", self.workspace, *args], env=self.environment,
            capture_output=True, text=True, check=True, timeout=10,
        )
        return result.stdout

    def invoke(self, **args: object) -> str:
        with mock.patch.dict(os.environ, self.environment):
            return asyncio.run(GitOpsTool().run(self.workspace, dict(args)))

    def test_missing_branch_does_not_restore_the_same_named_tracked_file(self) -> None:
        target = Path(self.workspace, "missing-branch")
        target.write_text("uncommitted work must survive\n", encoding="utf-8")
        result = self.invoke(action="checkout", branch="missing-branch")
        self.assertTrue(result.startswith("Error: git checkout:"), result)
        self.assertEqual(target.read_text(encoding="utf-8"), "uncommitted work must survive\n")
        self.assertEqual(self.git("branch", "--show-current").strip(), "audit-main")

    def test_existing_branch_switches_even_when_a_tracked_file_has_the_same_name(self) -> None:
        self.git("branch", "missing-branch")
        result = self.invoke(action="checkout", branch="missing-branch")
        self.assertFalse(result.startswith("Error:"), result)
        self.assertEqual(self.git("branch", "--show-current").strip(), "missing-branch")
        self.assertEqual(
            Path(self.workspace, "missing-branch").read_text(encoding="utf-8"), "committed text\n",
        )

    def test_checkout_can_create_a_new_branch(self) -> None:
        result = self.invoke(action="checkout", branch="audit-created", create=True)
        self.assertFalse(result.startswith("Error:"), result)
        self.assertEqual(self.git("branch", "--show-current").strip(), "audit-created")


if __name__ == "__main__":
    unittest.main()
