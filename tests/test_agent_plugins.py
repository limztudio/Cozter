from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import unittest

from Cozter.agent_tools.plugins.calculator import CalculatorTool
from Cozter.agent_tools.plugins.git_info import GitInfoTool
from Cozter.agent_tools.plugins.notes import NotesTool


def _run(coro):
    return asyncio.run(coro)


class CalculatorToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = CalculatorTool()

    def eval(self, expression: str) -> str:
        return _run(self.tool.run(".", {"expression": expression}))

    def test_arithmetic_precedence_and_parentheses(self) -> None:
        self.assertEqual(self.eval("2 + 3 * 4"), "14")
        self.assertEqual(self.eval("(2 + 3) * 4"), "20")
        self.assertEqual(self.eval("2 ** 10"), "1024")
        self.assertEqual(self.eval("7 // 2"), "3")
        self.assertEqual(self.eval("7 % 3"), "1")
        self.assertEqual(self.eval("-5 + 2"), "-3")
        self.assertEqual(self.eval("10 / 4"), "2.5")

    def test_functions_and_constants(self) -> None:
        self.assertEqual(self.eval("sqrt(16)"), "4")
        self.assertEqual(self.eval("floor(2.7) + ceil(2.1)"), "5")
        self.assertEqual(self.eval("min(3, 1, 2) + max(3, 1, 2)"), "4")
        self.assertEqual(self.eval("round(0.12345, 3)"), "0.123")
        self.assertEqual(self.eval("factorial(5)"), "120")
        self.assertEqual(self.eval("round(pi, 5)"), "3.14159")
        self.assertEqual(self.eval("round(e, 3)"), "2.718")
        self.assertEqual(self.eval("gcd(12, 18)"), "6")

    def test_rejects_non_numeric_and_unknown_syntax(self) -> None:
        for expression in (
            "__import__('os').system('true')",
            "().__class__",
            "[1, 2]",
            "'a' + 'b'",
            "x + 1",
            "1 < 2",
            "1 if 2 else 3",
            "lambda: 1",
            "f'{1}'",
            "abs(-1) if True else 0",
            "sqrt",
        ):
            with self.subTest(expression=expression):
                self.assertTrue(
                    self.eval(expression).startswith("Error:"),
                    msg=expression,
                )

    def test_rejects_dangerous_magnitudes(self) -> None:
        self.assertIn("Error", self.eval("9 ** 9 ** 9"))
        self.assertIn("Error", self.eval("factorial(5000)"))
        self.assertIn("Error", self.eval("2 ** 999999999"))

    def test_error_cases(self) -> None:
        self.assertEqual(self.eval("1 / 0"), "Error: division by zero")
        self.assertEqual(self.eval("1 // 0"), "Error: division by zero")
        self.assertIn("Error", self.eval("sqrt(-1)"))
        self.assertIn("Error", self.eval("(2+"))
        self.assertIn("Error", self.eval("sqrt(1, 2, 3)"))

    def test_missing_expression_arg(self) -> None:
        result = _run(self.tool.run(".", {}))
        self.assertTrue(result.startswith("Error:"))


class NotesToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = NotesTool()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = self._tmp.name

    def invoke(self, action: str, text: str | None = None) -> str:
        args: dict[str, str] = {"action": action}
        if text is not None:
            args["text"] = text
        return _run(self.tool.run(self.workspace, args))

    def notes_path(self) -> str:
        return os.path.join(self.workspace, ".cozter", "notes.md")

    def test_append_creates_timestamped_entry(self) -> None:
        result = self.invoke("append", "Investigated grep; found timeout bug")
        self.assertIn("Noted", result)
        with open(self.notes_path(), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("## ", content)
        self.assertIn("Investigated grep", content)

    def test_append_requires_text(self) -> None:
        self.assertTrue(self.invoke("append").startswith("Error:"))
        self.assertTrue(self.invoke("append", "   ").startswith("Error:"))

    def test_append_requires_known_action(self) -> None:
        self.assertTrue(self.invoke("delete").startswith("Error:"))

    def test_read_reports_empty_then_entries(self) -> None:
        self.assertIn("empty", self.invoke("read"))
        self.invoke("append", "first finding")
        self.invoke("append", "second finding")
        notes = self.invoke("read")
        self.assertIn("first finding", notes)
        self.assertIn("second finding", notes)
        self.assertIn("End of notes", notes)

    def test_clear_removes_notes(self) -> None:
        self.invoke("append", "temporary")
        self.assertIn("cleared", self.invoke("clear"))
        self.assertFalse(os.path.exists(self.notes_path()))
        self.assertIn("already empty", self.invoke("clear"))

    def test_trim_keeps_newest_entries_within_budget(self) -> None:
        # Write enough entries that the 64 KiB ceiling forces a trim.
        for index in range(400):
            self.invoke("append", f"entry {index} " + "x" * 300)
        with open(self.notes_path(), "rb") as f:
            size = len(f.read())
        notes = self.invoke("read")
        self.assertLessEqual(size, 64 * 1024 + 16 * 1024)
        self.assertIn("entry 399", notes)  # newest entry always kept

    def test_oversized_legacy_file_keeps_newest_entries(self) -> None:
        # A pre-existing oversized notes file must not push the newest
        # entries out of the window a later append reads.
        os.makedirs(os.path.dirname(self.notes_path()), exist_ok=True)
        with open(self.notes_path(), "w", encoding="utf-8") as f:
            f.write("## 2000-01-01 00:00:00\n" + "old " * 30_000 + "\n")
            f.write("## 2099-01-01 00:00:00\nnewest entry\n\n")
        result = self.invoke("append", "fresh append")
        self.assertIn("Noted", result)
        with open(self.notes_path(), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("newest entry", content)
        self.assertIn("fresh append", content)


class GitInfoToolTests(unittest.TestCase):
    def setUp(self) -> None:
        if not GitInfoToolTests._git_available():
            self.skipTest("git not available")
        self.tool = GitInfoTool()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = self._tmp.name
        self._git("init", "-q")
        self._git("config", "user.email", "test@example.com")
        self._git("config", "user.name", "Cozter Tests")

    @staticmethod
    def _git_available() -> bool:
        try:
            subprocess.run(
                ["git", "--version"],
                capture_output=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return False
        return True

    def _git(self, *args: str) -> None:
        subprocess.run(
            ["git", "-C", self.workspace, *args],
            capture_output=True,
            check=True,
        )

    def invoke(self, **args: object) -> str:
        return _run(self.tool.run(self.workspace, args))

    def _commit(self, filename: str, content: str, message: str) -> None:
        with open(os.path.join(self.workspace, filename), "w") as f:
            f.write(content)
        self._git("add", filename)
        self._git("commit", "-q", "-m", message)

    def test_status_shows_branch_and_changes(self) -> None:
        self._commit("a.txt", "hello\n", "initial commit")
        with open(os.path.join(self.workspace, "a.txt"), "a") as f:
            f.write("more\n")
        status = self.invoke(action="status")
        self.assertIn("##", status)  # branch line
        self.assertIn("a.txt", status)

    def test_log_lists_commit_subjects(self) -> None:
        self._commit("a.txt", "one\n", "first subject")
        self._commit("b.txt", "two\n", "second subject")
        log = self.invoke(action="log", limit=5)
        self.assertIn("first subject", log)
        self.assertIn("second subject", log)

    def test_diff_summary_and_patch(self) -> None:
        self._commit("a.txt", "one\n", "initial commit")
        with open(os.path.join(self.workspace, "a.txt"), "a") as f:
            f.write("two\n")
        summary = self.invoke(action="diff")
        self.assertIn("a.txt", summary)
        patch = self.invoke(action="diff", patch=True)
        self.assertIn("+two", patch)

    def test_diff_without_head_falls_back_to_index(self) -> None:
        # Zero-commit repository: no HEAD exists yet.
        with open(os.path.join(self.workspace, "a.txt"), "w") as f:
            f.write("staged\n")
        self._git("add", "a.txt")
        result = self.invoke(action="diff")
        self.assertFalse(result.startswith("Error:"))
        self.assertIn("a.txt", result)

    def test_non_repository_reports_git_error(self) -> None:
        empty = tempfile.mkdtemp()
        self.addCleanup(os.rmdir, empty)
        result = _run(self.tool.run(empty, {"action": "status"}))
        self.assertTrue(result.startswith("Error:"))

    def test_invalid_action_rejected(self) -> None:
        self.assertTrue(
            self.invoke(action="push").startswith("Error:"),
        )

    def test_path_escaping_rejected(self) -> None:
        self.assertTrue(
            self.invoke(action="diff", path="../outside").startswith(
                "Error:",
            ),
        )


if __name__ == "__main__":
    unittest.main()
