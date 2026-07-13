"""Tests for the auto-updater's guarded git operations.

``fetch_and_pull`` must not ``git pull`` over a checkout that is being
developed on (dirty tree or local commits ahead of upstream); it should
still pull a clean, current checkout. These tests stub ``updater._git`` so
no real repository is touched.
"""

import subprocess
import unittest
from unittest import mock

from Cozter import updater


def _fake_git(outputs: dict[str, str], calls: list[tuple[str, ...]]):
    """Stand-in for ``updater._git`` keyed on each command's last arg."""

    def _run(*args: str) -> subprocess.CompletedProcess:
        calls.append(args)
        return subprocess.CompletedProcess(
            args, 0, stdout=outputs.get(args[-1], ""), stderr="",
        )

    return _run


class UpdaterAutoPullGuardTests(unittest.TestCase):
    def _run_fetch_and_pull(
        self, outputs: dict[str, str],
    ) -> list[tuple[str, ...]]:
        calls: list[tuple[str, ...]] = []
        original = updater._git
        updater._git = _fake_git(outputs, calls)
        try:
            updater.fetch_and_pull()
        finally:
            updater._git = original
        return calls

    def _pulled(self, calls: list[tuple[str, ...]]) -> bool:
        return any(c and c[0] == "pull" for c in calls)

    def test_dirty_tree_skips_pull(self) -> None:
        calls = self._run_fetch_and_pull({
            "origin": "",           # git fetch origin
            "--porcelain": " M x",  # git status --porcelain -> dirty
            "HEAD": "abc123",       # git rev-parse HEAD
        })
        self.assertFalse(
            self._pulled(calls),
            f"pull should be skipped on a dirty tree; calls={calls}",
        )

    def test_local_ahead_skips_pull(self) -> None:
        calls = self._run_fetch_and_pull({
            "origin": "",
            "--porcelain": "",         # clean
            "@{u}": "origin/main",     # upstream resolves
            "origin/main..HEAD": "2",  # 2 local commits ahead
            "HEAD": "abc123",
        })
        self.assertFalse(
            self._pulled(calls),
            f"pull should be skipped when local is ahead; calls={calls}",
        )

    def test_clean_and_current_pulls(self) -> None:
        calls = self._run_fetch_and_pull({
            "origin": "",
            "--porcelain": "",         # clean
            "@{u}": "origin/main",
            "origin/main..HEAD": "0",  # not ahead
            "--ff-only": "",           # git pull --ff-only
            "HEAD": "abc123",
        })
        self.assertTrue(
            self._pulled(calls),
            f"pull should run on a clean, current checkout; calls={calls}",
        )

    def test_failed_ahead_comparison_skips_pull(self) -> None:
        calls: list[tuple[str, ...]] = []

        def _run(*args: str) -> subprocess.CompletedProcess:
            calls.append(args)
            if args[0] == "rev-parse" and args[-1] == "@{u}":
                return subprocess.CompletedProcess(
                    args, 0, stdout="origin/main\n", stderr="",
                )
            if args[0] == "rev-list":
                return subprocess.CompletedProcess(
                    args, 128, stdout="", stderr="bad revision",
                )
            stdout = "abc123\n" if args[-1] == "HEAD" else ""
            return subprocess.CompletedProcess(
                args, 0, stdout=stdout, stderr="",
            )

        original = updater._git
        updater._git = _run
        try:
            updater.fetch_and_pull()
        finally:
            updater._git = original

        self.assertFalse(
            self._pulled(calls),
            "pull should be skipped when local/upstream comparison fails",
        )

    def test_missing_upstream_skips_pull(self) -> None:
        calls: list[tuple[str, ...]] = []

        def _run(*args: str) -> subprocess.CompletedProcess:
            calls.append(args)
            if args[0] == "rev-parse" and args[-1] == "@{u}":
                return subprocess.CompletedProcess(
                    args, 128, stdout="", stderr="no upstream",
                )
            stdout = "abc123\n" if args[-1] == "HEAD" else ""
            return subprocess.CompletedProcess(
                args, 0, stdout=stdout, stderr="",
            )

        original = updater._git
        updater._git = _run
        try:
            updater.fetch_and_pull()
        finally:
            updater._git = original

        self.assertFalse(
            self._pulled(calls),
            "pull should be skipped when the branch has no upstream",
        )

    def test_check_for_update_detects_remote_commit_without_pulling(self) -> None:
        calls: list[tuple[str, ...]] = []
        with (
            mock.patch.object(updater, "_git", _fake_git({
                "origin": "",
                "HEAD": "abc123",
                "--porcelain": "",
                "@{u}": "origin/main",
                "origin/main..HEAD": "0",
                "HEAD..origin/main": "1",
            }, calls)),
            mock.patch.object(updater, "_STARTUP_COMMIT", "abc123"),
        ):
            available = updater.check_for_update()

        self.assertTrue(available)
        self.assertFalse(
            self._pulled(calls),
            f"check must not alter the checkout; calls={calls}",
        )

    def test_check_for_update_returns_false_when_upstream_is_current(self) -> None:
        calls: list[tuple[str, ...]] = []
        with (
            mock.patch.object(updater, "_git", _fake_git({
                "origin": "",
                "HEAD": "abc123",
                "--porcelain": "",
                "@{u}": "origin/main",
                "origin/main..HEAD": "0",
                "HEAD..origin/main": "0",
            }, calls)),
            mock.patch.object(updater, "_STARTUP_COMMIT", "abc123"),
        ):
            available = updater.check_for_update()

        self.assertFalse(available)
        self.assertFalse(self._pulled(calls))


class RestartScriptTests(unittest.TestCase):
    def test_windows_update_exits_for_the_supervisor(self) -> None:
        with (
            mock.patch.object(updater.os, "name", "nt"),
            mock.patch.dict(
                updater.os.environ,
                {updater.WINDOWS_SUPERVISOR_ENV: "1"},
            ),
            mock.patch.object(updater.os, "_exit") as exit_mock,
            mock.patch.object(updater.os, "execv") as execv_mock,
        ):
            updater.restart_script()

        exit_mock.assert_called_once_with(
            updater.WINDOWS_SUPERVISOR_RESTART_EXIT_CODE,
        )
        execv_mock.assert_not_called()

    def test_unsupervised_windows_update_waits_for_replacement(self) -> None:
        with (
            mock.patch.object(updater.os, "name", "nt"),
            mock.patch.dict(
                updater.os.environ,
                {updater.WINDOWS_SUPERVISOR_ENV: ""},
            ),
            mock.patch.object(updater.os, "chdir") as chdir_mock,
            mock.patch.object(
                updater.subprocess, "call", return_value=23,
            ) as call_mock,
            mock.patch.object(
                updater.os, "_exit", side_effect=SystemExit,
            ) as exit_mock,
            mock.patch.object(updater.os, "execv") as execv_mock,
        ):
            with self.assertRaises(SystemExit):
                updater.restart_script()

        parent_dir = updater.os.path.dirname(updater.MODULE_ROOT)
        call_mock.assert_called_once_with(
            [updater.sys.executable, "-m", "Cozter", *updater.sys.argv[1:]],
            cwd=parent_dir,
        )
        exit_mock.assert_called_once_with(23)
        chdir_mock.assert_not_called()
        execv_mock.assert_not_called()

    def test_nonzero_restart_code_is_preserved(self) -> None:
        with mock.patch.object(updater.os, "_exit") as exit_mock:
            updater.restart_script(99)

        exit_mock.assert_called_once_with(99)


if __name__ == "__main__":
    unittest.main()
