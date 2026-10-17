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
    def setUp(self):
        startup = mock.patch.object(updater, "_STARTUP_COMMIT", "abc123")
        startup.start()
        self.addCleanup(startup.stop)

    def test_git_probe_closes_stdin_instead_of_sharing_cli_input(self) -> None:
        with mock.patch.object(updater.subprocess, "run") as run:
            updater._git("rev-parse", "HEAD")

        self.assertEqual(run.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertNotIn("timeout", run.call_args.kwargs)

    def test_background_git_uses_cached_credentials_without_modifying_parent_environment(self):
        with (mock.patch.dict(updater.os.environ, {
                    "GIT_TERMINAL_PROMPT": "1", "GCM_INTERACTIVE": "1",
                    "COZTER_TEST_ENV": "retained",
                }), mock.patch.object(updater.subprocess, "run") as run):
            updater._git("fetch", "origin")
            child_env = run.call_args.kwargs["env"]
            self.assertEqual(child_env["GIT_TERMINAL_PROMPT"], "0")
            self.assertEqual(child_env["GCM_INTERACTIVE"], "0")
            self.assertEqual(child_env["COZTER_TEST_ENV"], "retained")
            self.assertEqual(updater.os.environ["GCM_INTERACTIVE"], "1")
            self.assertEqual(updater.os.environ["GIT_TERMINAL_PROMPT"], "1")

    def test_already_applied_local_update_does_not_wait_for_network(self):
        with (mock.patch.object(updater, "_head_changed", return_value=True),
              mock.patch.object(updater, "_fetch_origin", side_effect=AssertionError("network unnecessary"))):
            self.assertTrue(updater.check_for_update())
            self.assertTrue(updater.fetch_and_pull())

    def test_local_update_during_fetch_is_still_detected(self):
        with (mock.patch.object(updater, "_head_changed", side_effect=[False, True]),
              mock.patch.object(updater, "_fetch_origin", return_value=False)):
            self.assertTrue(updater.check_for_update())

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
        return any(call and call[0] == "pull" for call in calls)

    def test_dirty_tree_skips_pull(self) -> None:
        calls = self._run_fetch_and_pull({
            "origin": "",
            "--porcelain": " M x",
            "HEAD": "abc123",
        })
        self.assertFalse(
            self._pulled(calls),
            f"pull should be skipped on a dirty tree; calls={calls}",
        )

    def test_local_ahead_skips_pull(self) -> None:
        calls = self._run_fetch_and_pull({
            "origin": "",
            "--porcelain": "",
            "@{u}": "origin/main",
            "origin/main..HEAD": "2",
            "HEAD": "abc123",
        })
        self.assertFalse(
            self._pulled(calls),
            f"pull should be skipped when local is ahead; calls={calls}",
        )

    def test_clean_and_current_pulls(self) -> None:
        calls = self._run_fetch_and_pull({
            "origin": "",
            "--porcelain": "",
            "@{u}": "origin/main",
            "origin/main..HEAD": "0",
            "--ff-only": "",
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

    def test_git_os_error_is_treated_as_an_unavailable_updater(self) -> None:
        # Git-unavailable errors must not crash the loop.
        with mock.patch.object(updater, "_git", side_effect=PermissionError):
            self.assertEqual(updater.get_current_version(), "(unknown)")
            self.assertFalse(updater.check_for_update())
            self.assertFalse(updater.fetch_and_pull())

    def test_failed_version_probe_does_not_publish_git_partial_stdout(self) -> None:
        # An unborn checkout prints "HEAD" to stdout before rev-parse fails.
        failed = subprocess.CompletedProcess(
            [], 128, stdout="HEAD\n", stderr="fatal: ambiguous argument 'HEAD'",
        )
        with mock.patch.object(updater, "_git", return_value=failed):
            self.assertEqual(updater.get_current_version(), "(unknown)")
            self.assertEqual(updater._get_head_commit(), "")

    def test_pull_error_still_restarts_if_head_moved(self) -> None:
        with (
            mock.patch.object(updater, "_fetch_origin", return_value=True),
            mock.patch.object(updater, "_skip_pull_reason", return_value=None),
            mock.patch.object(
                updater, "_git",
                side_effect=PermissionError("synthetic Git failure"),
            ),
            mock.patch.object(updater, "_head_changed", return_value=False),
            mock.patch.object(updater, "_get_head_commit", return_value="new456"),
        ):
            self.assertTrue(updater.fetch_and_pull())

    def test_output_decode_error_after_pull_changed_head_requires_restart(self) -> None:
        current = "abc123"

        def pull(*args: str) -> subprocess.CompletedProcess:
            nonlocal current
            self.assertEqual(args, ("pull", "--ff-only"))
            current = "new456"
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "synthetic output")

        with (
            mock.patch.object(updater, "_fetch_origin", return_value=True),
            mock.patch.object(updater, "_skip_pull_reason", return_value=None),
            mock.patch.object(updater, "_get_head_commit", side_effect=lambda: current),
            mock.patch.object(updater, "_git", side_effect=pull),
            self.assertLogs(updater.logger, level="ERROR") as logs,
        ):
            self.assertTrue(updater.fetch_and_pull())
        self.assertIn("UnicodeDecodeError", "\n".join(logs.output))

    def test_failed_pull_with_confirmed_unchanged_head_preserves_error(self) -> None:
        failure = ValueError("synthetic pull failure")
        with (
            mock.patch.object(updater, "_fetch_origin", return_value=True),
            mock.patch.object(updater, "_skip_pull_reason", return_value=None),
            mock.patch.object(updater, "_get_head_commit", return_value="abc123"),
            mock.patch.object(updater, "_git", side_effect=failure),
            self.assertRaises(ValueError) as caught,
        ):
            updater.fetch_and_pull()
        self.assertIs(caught.exception, failure)

    def test_failed_pull_with_unreadable_head_requires_restart(self) -> None:
        with (
            mock.patch.object(updater, "_fetch_origin", return_value=True),
            mock.patch.object(updater, "_skip_pull_reason", return_value=None),
            mock.patch.object(updater, "_get_head_commit", side_effect=["abc123", ""]),
            mock.patch.object(updater, "_git", side_effect=ValueError("pull failed")),
            self.assertLogs(updater.logger, level="WARNING"),
        ):
            self.assertTrue(updater.fetch_and_pull())

    def test_successful_pull_with_unreadable_head_requires_restart(self) -> None:
        with (
            mock.patch.object(updater, "_fetch_origin", return_value=True),
            mock.patch.object(updater, "_skip_pull_reason", return_value=None),
            mock.patch.object(updater, "_get_head_commit", side_effect=["abc123", ""]),
            mock.patch.object(updater, "_git", return_value=subprocess.CompletedProcess([], 0)),
            self.assertLogs(updater.logger, level="WARNING"),
        ):
            self.assertTrue(updater.fetch_and_pull())

    def test_head_probe_failure_after_pull_requires_restart(self) -> None:
        with (
            mock.patch.object(updater, "_fetch_origin", return_value=True),
            mock.patch.object(updater, "_skip_pull_reason", return_value=None),
            mock.patch.object(
                updater, "_get_head_commit",
                side_effect=["abc123", ValueError("bad HEAD output"), ValueError("still unreadable")],
            ),
            mock.patch.object(updater, "_git", return_value=subprocess.CompletedProcess([], 0)),
            self.assertLogs(updater.logger, level="ERROR"),
        ):
            self.assertTrue(updater.fetch_and_pull())

    def test_error_before_pull_does_not_classify_checkout_as_mutated(self) -> None:
        with (
            mock.patch.object(updater, "_head_changed", return_value=False),
            mock.patch.object(updater, "_fetch_origin", return_value=True),
            mock.patch.object(updater, "_skip_pull_reason", side_effect=ValueError("bad status")),
            mock.patch.object(updater, "_git") as git,
            self.assertRaisesRegex(ValueError, "bad status"),
        ):
            updater.fetch_and_pull()
        git.assert_not_called()


class UpdaterRequirementsTests(unittest.TestCase):
    def test_unchanged_requirements_skip_install(self) -> None:
        unchanged = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with (
            mock.patch.object(updater.os.path, "exists", return_value=True),
            mock.patch.object(updater, "_STARTUP_COMMIT", "old"),
            mock.patch.object(updater, "_git", return_value=unchanged),
            mock.patch.object(updater.subprocess, "run") as run,
        ):
            updater.install_requirements()

        run.assert_not_called()

    def test_changed_requirements_install(self) -> None:
        changed = subprocess.CompletedProcess([], 1, stdout="", stderr="")
        installed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with (
            mock.patch.object(updater.os.path, "exists", return_value=True),
            mock.patch.object(updater, "_STARTUP_COMMIT", "old"),
            mock.patch.object(updater, "_git", return_value=changed),
            mock.patch.object(
                updater.subprocess, "run", return_value=installed,
            ) as run,
        ):
            updater.install_requirements()

        self.assertNotIn("timeout", run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs["stdin"], subprocess.DEVNULL)

    def test_updated_requirements_failure_is_reported(self) -> None:
        with (
            mock.patch.object(updater.os.path, "exists", return_value=True),
            mock.patch.object(
                updater.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="synthetic pip failure"),
            ), self.assertRaisesRegex(RuntimeError, "pip install failed")
        ):
            updater.install_requirements()


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

    def test_unsupervised_windows_update_exits_without_nesting_python(self) -> None:
        with (
            mock.patch.object(updater.os, "name", "nt"),
            mock.patch.dict(
                updater.os.environ,
                {updater.WINDOWS_SUPERVISOR_ENV: ""},
            ),
            mock.patch.object(updater.os, "chdir") as chdir_mock,
            mock.patch.object(updater.subprocess, "call") as call_mock,
            mock.patch.object(
                updater.os, "_exit", side_effect=SystemExit,
            ) as exit_mock,
            mock.patch.object(updater.os, "execv") as execv_mock,
            self.assertRaises(SystemExit),
        ):
            updater.restart_script()

        call_mock.assert_not_called()
        exit_mock.assert_called_once_with(
            updater.WINDOWS_SUPERVISOR_RESTART_EXIT_CODE,
        )
        chdir_mock.assert_not_called()
        execv_mock.assert_not_called()

    def test_nonzero_restart_code_is_preserved(self) -> None:
        with mock.patch.object(updater.os, "_exit") as exit_mock:
            updater.restart_script(99)

        exit_mock.assert_called_once_with(99)


if __name__ == "__main__":
    unittest.main()
