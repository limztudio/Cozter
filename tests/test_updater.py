"""Tests for the auto-updater's guarded git operations.

``fetch_and_pull`` must not ``git pull`` over a checkout that is being
developed on (dirty tree or local commits ahead of upstream); it should
still pull a clean, current checkout. These tests stub ``updater._git`` so
no real repository is touched.
"""

import subprocess
import unittest

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


if __name__ == "__main__":
    unittest.main()
