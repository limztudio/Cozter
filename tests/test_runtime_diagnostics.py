"""Behavioral tests for the runtime-diagnostics plumbing in ``__main__``.

``__main__`` checks whether a fresh virtual environment is missing or has an
incompatible version of a required runtime dependency at import time. We
neutralize its dependency-repair call before importing so these tests stay
hermetic and don't hit the network, while still exercising the real
``dump_runtime_diagnostics`` / ``_enable_faulthandler`` code paths.
"""

import asyncio
import contextlib
import faulthandler
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


def _load_main_module():
    """Import ``Cozter.__main__`` with the import-time pip install disabled."""
    # Checkout must win over the live install on sys.path.
    workspace_pkg_parent = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sys.path = [
        entry for entry in sys.path
        if os.path.abspath(entry) not in {
            os.path.abspath(workspace_pkg_parent),
            "/home/utilities/AutoStart",
        }
    ]
    sys.path.insert(0, workspace_pkg_parent)

    real_check_call = subprocess.check_call
    old_reexec = os.environ.get("COZTER_VENV_REEXEC")
    subprocess.check_call = lambda *_args, **_kwargs: 0
    os.environ["COZTER_VENV_REEXEC"] = "1"
    try:
        import importlib
        with contextlib.redirect_stderr(io.StringIO()):
            import Cozter.__main__ as main_mod
            importlib.reload(main_mod)  # in case an earlier import cached deps
        return main_mod
    finally:
        subprocess.check_call = real_check_call
        if old_reexec is None:
            os.environ.pop("COZTER_VENV_REEXEC", None)
        else:
            os.environ["COZTER_VENV_REEXEC"] = old_reexec


class _StubBot:
    """Minimal stand-in for the per-platform turn-tracking surface."""

    def __init__(self, platform_id: str, *, active: bool, diag: str = ""):
        self.platform_id = platform_id
        self._active = active
        self._diag = diag

    def has_active_turns(self) -> bool:
        return self._active

    def stuck_turn_diagnostics(self) -> str:
        return self._diag


class VenvBootstrapTests(unittest.TestCase):
    def test_dependency_bootstrap_skips_pip_when_runtime_is_complete(self) -> None:
        main = _load_main_module()
        with (
            mock.patch.object(main, "_runtime_dependency_issues", return_value=[]),
            mock.patch.object(main.subprocess, "check_call") as install,
        ):
            main._install_deps()

        install.assert_not_called()

    def test_dependency_bootstrap_installs_when_runtime_is_incompatible(self) -> None:
        main = _load_main_module()
        with (
            mock.patch.object(
                main,
                "_runtime_dependency_issues",
                return_value=["aiohttp 3.14.1 does not satisfy >=3.14.3"],
            ),
            mock.patch.object(main.subprocess, "check_call") as install,
        ):
            main._install_deps()

        install.assert_called_once()
        args, kwargs = install.call_args
        self.assertEqual(args[0][:4], [main.sys.executable, "-m", "pip", "install"])
        self.assertEqual(kwargs["timeout"], main._DEPENDENCY_INSTALL_TIMEOUT_SEC)

    def test_dependency_check_detects_an_outdated_declared_version(self) -> None:
        main = _load_main_module()
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as req_file:
            req_file.write("aiohttp>=3.14.3,<4\n")
            req_file.flush()
            with (
                mock.patch.object(main, "find_spec", return_value=object()),
                mock.patch.object(
                    main, "distribution_version", return_value="3.14.1",
                ),
            ):
                issues = main._runtime_dependency_issues(req_file.name)

        self.assertEqual(
            issues, ["aiohttp 3.14.1 does not satisfy <4,>=3.14.3"],
        )

    def test_dependency_bootstrap_reports_a_bounded_install_timeout(self) -> None:
        main = _load_main_module()
        with (
            mock.patch.object(
                main, "_runtime_dependency_issues", return_value=["missing aiohttp"],
            ),
            mock.patch.object(
                main.subprocess,
                "check_call",
                side_effect=subprocess.TimeoutExpired("pip", 1),
            ),self.assertRaisesRegex(RuntimeError, "Timed out installing")
        ):
            main._install_deps()

    def test_unsupported_python_is_rejected_before_bootstrap(self) -> None:
        main = _load_main_module()
        stderr = io.StringIO()
        with (
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as exited,
        ):
            main._require_supported_python((3, 10))

        self.assertEqual(exited.exception.code, 1)
        self.assertIn("requires Python 3.11", stderr.getvalue())

    def test_windows_bootstrap_supervises_venv_restarts(self) -> None:
        main = _load_main_module()
        with (
            mock.patch.object(main, "_running_in_venv", return_value=False),
            mock.patch.dict(
                main.os.environ, {main._VENV_REEXEC_ENV: ""}, clear=False,
            ),
            mock.patch.object(main.os.path, "exists", return_value=True),
            mock.patch.object(main.os, "name", "nt"),
            mock.patch.object(
                main.subprocess, "call", side_effect=[
                    main.updater.WINDOWS_SUPERVISOR_RESTART_EXIT_CODE, 23,
                ],
            ) as call_mock,
            mock.patch.object(main.time, "sleep") as sleep_mock,
            mock.patch.object(
                main.os, "_exit", side_effect=SystemExit,
            ) as exit_mock,
            mock.patch.object(main.os, "execve") as execve_mock,
        ):
            python = main._venv_python()
            with self.assertRaises(SystemExit):
                main._ensure_venv_and_reexec()

        self.assertEqual(call_mock.call_count, 2)
        args, kwargs = call_mock.call_args
        self.assertEqual(args[0], [python, "-m", "Cozter", *main.sys.argv[1:]])
        self.assertEqual(kwargs["cwd"], main._pkg_parent)
        self.assertEqual(kwargs["env"][main._VENV_REEXEC_ENV], "1")
        self.assertEqual(
            kwargs["env"][main.updater.WINDOWS_SUPERVISOR_ENV], "1",
        )
        sleep_mock.assert_called_once_with(
            main._WINDOWS_CHILD_RESTART_DELAY_SEC,
        )
        exit_mock.assert_called_once_with(23)
        execve_mock.assert_not_called()


class LaunchArgumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._main = _load_main_module()

    def test_help_exits_without_starting_the_runtime(self) -> None:
        stdout = io.StringIO()
        with (
            mock.patch.object(self._main.sys, "stdout", stdout),
            self.assertRaises(SystemExit) as exited,
        ):
            self._main._validate_launch_args(["--help"])

        self.assertEqual(exited.exception.code, 0)
        self.assertIn("Usage: python -m Cozter", stdout.getvalue())
        self.assertIn("--cli", stdout.getvalue())

    def test_unknown_launch_argument_fails_instead_of_starting_daemon(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.object(self._main.sys, "stderr", stderr),
            self.assertRaises(SystemExit) as exited,
        ):
            self._main._validate_launch_args(["--typo"])

        self.assertEqual(exited.exception.code, 2)
        self.assertIn("--typo", stderr.getvalue())
        self.assertIn("--help", stderr.getvalue())


class DumpRuntimeDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._main = _load_main_module()
        # Point LOG_DIR at temp for a fresh diagnostics.log.
        self._orig_log_dir = self._main.LOG_DIR
        self._orig_dump_file = self._main._dump_file
        self._faulthandler_was_enabled = faulthandler.is_enabled()
        self._main.LOG_DIR = self._tmp
        self._main._dump_file = None

    def tearDown(self):
        if not self._faulthandler_was_enabled:
            with contextlib.suppress(Exception):
                faulthandler.disable()
        if self._main._dump_file is not None:
            with contextlib.suppress(Exception):
                self._main._dump_file.close()
        self._main.LOG_DIR = self._orig_log_dir
        self._main._dump_file = self._orig_dump_file
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _read_dump(self) -> str:
        with open(
            os.path.join(self._tmp, "diagnostics.log"), encoding="utf-8",
        ) as file_handle:
            return file_handle.read()

    def test_dump_writes_header_reason_tasks_and_threads(self):
        # A real loop keeps the tasks section non-empty.
        async def _driver():
            self._main.dump_runtime_diagnostics(None, reason="unit-test")
        asyncio.run(_driver())

        body = self._read_dump()
        self.assertIn("diagnostics dump (unit-test)", body)
        self.assertIn("-- asyncio tasks", body)
        self.assertIn("-- active threads", body)
        self.assertIn("--- thread MainThread", body)

    def test_dump_records_per_bot_turn_state(self):
        bot_active = _StubBot("test:active", active=True, diag="LEAKED")
        bot_idle = _StubBot("test:idle", active=False)

        self._main.dump_runtime_diagnostics([bot_active, bot_idle])

        body = self._read_dump()
        self.assertIn("-- bot turn state --", body)
        self.assertIn("test:active: has_active_turns=True LEAKED", body)
        self.assertIn("test:idle: has_active_turns=False <idle>", body)

    def test_dump_tolerates_a_bot_that_raises(self):
        class _BrokenBot:
            platform_id = "test:broken"

            def has_active_turns(self):
                raise RuntimeError("boom")

        # Should not raise; error captured inline.
        self._main.dump_runtime_diagnostics([_BrokenBot()])
        body = self._read_dump()
        self.assertIn("test:broken", body)
        self.assertIn("boom", body)

    def test_bot_label_falls_back_to_class_name(self):
        class _NoId:
            pass

        # platform_id raises; fall back to the class name.
        self.assertEqual(
            self._main._bot_label(_NoId()), _NoId().__class__.__name__,
        )

    def test_update_idle_diagnostic_keeps_waiting_for_active_turn(self):
        bot = _StubBot("test:active", active=True, diag="still-running")
        reasons: list[str] = []
        sleeps = 0

        async def fake_sleep(_seconds):
            nonlocal sleeps
            sleeps += 1
            bot._active = False

        old_timeout = self._main.cfg.get_update_idle_timeout
        old_dump = self._main.dump_runtime_diagnostics
        old_sleep = self._main.asyncio.sleep
        old_critical = self._main.logger.critical
        self._main.cfg.get_update_idle_timeout = lambda: 0
        self._main.dump_runtime_diagnostics = (
            lambda _bots, *, reason: reasons.append(reason)
        )
        self._main.asyncio.sleep = fake_sleep
        self._main.logger.critical = lambda *args, **kwargs: None
        try:
            asyncio.run(
                self._main._wait_for_update_idle(
                    [bot], log_message="waiting in test",
                )
            )
        finally:
            self._main.cfg.get_update_idle_timeout = old_timeout
            self._main.dump_runtime_diagnostics = old_dump
            self._main.asyncio.sleep = old_sleep
            self._main.logger.critical = old_critical

        self.assertEqual(reasons, ["update-idle-still-waiting"])
        self.assertGreaterEqual(sleeps, 1)


class UpdateLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_update_restart_waits_for_active_reply_delivery(self):
        """An update may be detected mid-reply, but never cuts it short."""
        main = _load_main_module()
        reply_delivered = asyncio.Event()
        restart_started = asyncio.Event()
        events: list[str] = []

        class _Bot:
            platform_id = "test:active-reply"

            async def begin_update_restart(self):
                events.append("pause")
                restart_started.set()

            def has_active_turns(self):
                # True until final reply I/O and cleanup run.
                return not reply_delivered.is_set()

            def stuck_turn_diagnostics(self):
                return "final reply is still being delivered"

            async def cancel_update_restart(self):
                events.append("resume")

            async def notify_users(self, _message):
                events.append("notify")

            async def stop(self):
                events.append("stop")

        bot = _Bot()

        def fetch_and_pull() -> bool:
            self.assertTrue(
                reply_delivered.is_set(),
                "the checkout must not change before the reply is delivered",
            )
            events.append("pull")
            return True

        old_poll = main._UPDATE_IDLE_POLL_SEC
        main._UPDATE_IDLE_POLL_SEC = 0.001
        try:
            with (
                mock.patch.object(
                    main.updater, "fetch_and_pull", side_effect=fetch_and_pull,
                ) as pull,
                mock.patch.object(
                    main.updater, "install_requirements",
                    side_effect=lambda: events.append("install"),
                ) as install,
                mock.patch.object(
                    main.updater, "restart_script",
                    side_effect=lambda _code: events.append("restart"),
                ) as restart,
            ):
                update = asyncio.create_task(
                    main._restart_after_update([bot], restart_code=0),
                )
                await asyncio.wait_for(restart_started.wait(), timeout=1)
                await asyncio.sleep(0)

                self.assertEqual(events, ["pause"])
                pull.assert_not_called()
                install.assert_not_called()
                restart.assert_not_called()

                reply_delivered.set()
                await asyncio.wait_for(update, timeout=1)

                pull.assert_called_once_with()
                install.assert_called_once_with()
                restart.assert_called_once_with(0)
                self.assertEqual(
                    events,
                    ["pause", "pull", "install", "notify", "stop", "restart"],
                )
        finally:
            main._UPDATE_IDLE_POLL_SEC = old_poll

    async def test_no_update_does_not_pause_message_intake(self):
        main = _load_main_module()

        class _Bot:
            restart_calls = 0

            async def begin_update_restart(self):
                self.restart_calls += 1

        sleeps = 0

        async def fake_sleep(_seconds):
            nonlocal sleeps
            sleeps += 1
            if sleeps > 1:
                raise asyncio.CancelledError

        bot = _Bot()
        with (
            mock.patch.object(main.asyncio, "sleep", side_effect=fake_sleep),
            mock.patch.object(
                main.updater, "check_for_update", return_value=False,
            ) as check_mock,self.assertRaises(asyncio.CancelledError)
        ):
            await main.update_loop([bot], interval=1)

        check_mock.assert_called_once_with()
        # No update: intake must not pause.
        self.assertEqual(bot.restart_calls, 0)


class PlatformLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def _bot(self):
        return SimpleNamespace(
            start=mock.AsyncMock(), stop=mock.AsyncMock(),
            send_startup_messages=mock.AsyncMock(), notify_users=mock.AsyncMock(),
        )

    async def _run_daemon(self, main, bots) -> None:
        stop_event = asyncio.Event()
        stop_event.set()
        with (
            mock.patch.object(main, "_cli_mode_requested", return_value=False),
            mock.patch.object(main.cfg, "load_config", return_value={"update_check_interval": 300}),
            mock.patch.object(main, "create_platforms", return_value=bots),
            mock.patch.object(main.updater, "init_startup_commit"),
            mock.patch.object(main.updater, "get_current_version", return_value="test"),
            mock.patch.object(main.updater, "get_last_commit_date", return_value="test"),
            mock.patch.object(main.asyncio, "Event", return_value=stop_event),
            mock.patch.object(asyncio.get_running_loop(), "add_signal_handler"),
        ):
            await main.main()

    async def test_failed_start_cleans_started_and_partially_started_platforms(self) -> None:
        main = _load_main_module()
        first, failed, untouched = self._bot(), self._bot(), self._bot()
        failed.start.side_effect = RuntimeError("startup failed")
        with self.assertRaisesRegex(RuntimeError, "startup failed"):
            await self._run_daemon(main, [first, failed, untouched])
        first.stop.assert_awaited_once()
        failed.stop.assert_awaited_once()
        untouched.start.assert_not_awaited()

    async def test_shutdown_failures_do_not_skip_other_platform_cleanup(self) -> None:
        main = _load_main_module()
        first, second = self._bot(), self._bot()
        first.notify_users.side_effect = RuntimeError("send failed")
        second.stop.side_effect = RuntimeError("stop failed")
        with self.assertLogs(main.logger, level="ERROR"):
            await self._run_daemon(main, [first, second])
        first.stop.assert_awaited_once()
        second.stop.assert_awaited_once()

    async def test_cli_failed_start_still_stops_partial_platform(self) -> None:
        from Cozter.backends_bot import cli

        main = _load_main_module()
        bot = self._bot()
        bot.start.side_effect = RuntimeError("CLI startup failed")
        with (
            mock.patch.object(cli, "CliBot", return_value=bot),
            mock.patch.object(main.updater, "init_startup_commit"),
            mock.patch.object(main.updater, "get_current_version", return_value="test"),
            mock.patch.object(main.updater, "get_last_commit_date", return_value="test"),
            self.assertRaisesRegex(RuntimeError, "CLI startup failed"),
        ):
            await main.main_cli()
        bot.stop.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
