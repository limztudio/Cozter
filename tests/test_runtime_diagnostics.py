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
import threading
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
        if os.path.abspath(entry) != os.path.abspath(workspace_pkg_parent)
    ]
    sys.path.insert(0, workspace_pkg_parent)

    with (
        mock.patch.object(subprocess, "check_call", return_value=0),
        # Import as a virtual-environment runtime even when the test runner
        # uses system Python. A reexec marker alone is not proof of a venv.
        mock.patch.object(sys, "base_prefix", sys.prefix + "-test-base"),
    ):
        import importlib
        with contextlib.redirect_stderr(io.StringIO()):
            import Cozter.__main__ as main_mod
            importlib.reload(main_mod)  # in case an earlier import cached deps
        return main_mod


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
    def test_system_python_reexecs_despite_inherited_venv_marker(self) -> None:
        main = _load_main_module()
        python = main._venv_python()
        with (
            mock.patch.object(main, "_running_in_venv", return_value=False),
            mock.patch.object(main.sys, "executable", "/usr/bin/python3"),
            mock.patch.dict(main.os.environ, {main._VENV_REEXEC_ENV: "1"}),
            mock.patch.object(main.os, "name", "posix"),
            mock.patch.object(main, "_venv_python", return_value=python),
            mock.patch.object(main.os.path, "exists", return_value=True),
            mock.patch.object(main.os, "execve") as reexec,
            mock.patch.object(main.subprocess, "check_call") as install,
        ):
            main._ensure_venv_and_reexec()

        reexec.assert_called_once()
        args = reexec.call_args.args
        self.assertEqual(args[0], python)
        self.assertEqual(args[1], [python, "-m", "Cozter", *main.sys.argv[1:]])
        self.assertEqual(args[2][main._VENV_REEXEC_ENV], "1")
        install.assert_not_called()

    def test_broken_venv_interpreter_fails_before_reexec_or_system_install(self) -> None:
        main = _load_main_module()
        python = main._venv_python()
        with (
            mock.patch.object(main, "_running_in_venv", return_value=False),
            mock.patch.object(main.sys, "executable", python),
            mock.patch.dict(main.os.environ, {main._VENV_REEXEC_ENV: "1"}),
            mock.patch.object(main.os, "execve") as reexec,
            mock.patch.object(main.subprocess, "check_call") as install,
            self.assertRaisesRegex(RuntimeError, "recreate the project .venv"),
        ):
            main._ensure_venv_and_reexec()

        reexec.assert_not_called()
        install.assert_not_called()

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
        self.assertNotIn("timeout", kwargs)

    def test_dependency_check_detects_an_outdated_declared_version(self) -> None:
        main = _load_main_module()
        with tempfile.TemporaryDirectory() as tmp:
            req_path = os.path.join(tmp, "requirements.txt")
            with open(req_path, "w", encoding="utf-8") as req_file:
                req_file.write("aiohttp>=3.14.3,<4\n")
            with (
                mock.patch.object(main, "find_spec", return_value=object()),
                mock.patch.object(
                    main, "distribution_version", return_value="3.14.1",
                ),
            ):
                issues = main._runtime_dependency_issues(req_path)

        self.assertEqual(
            issues, ["aiohttp 3.14.1 does not satisfy <4,>=3.14.3"],
        )

    def test_dependency_bootstrap_reports_install_failure(self) -> None:
        main = _load_main_module()
        with (
            mock.patch.object(
                main, "_runtime_dependency_issues", return_value=["missing aiohttp"],
            ),
            mock.patch.object(
                main.subprocess,
                "check_call",
                side_effect=subprocess.CalledProcessError(1, "pip"),
            ), self.assertRaises(subprocess.CalledProcessError)
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
        class _MissingPlatformIdBot:
            pass

        # platform_id raises; fall back to the class name.
        self.assertEqual(
            self._main._bot_label(_MissingPlatformIdBot()), _MissingPlatformIdBot().__class__.__name__,
        )

    def test_log_crash_retention_caps_crash_files(self):
        cap = self._main._MAX_CRASH_FILES
        self.assertGreaterEqual(cap, 1)
        # Seed cap+5 stale crash files; newest write must prune back to cap.
        for index in range(cap + 5):
            with open(
                os.path.join(self._tmp, f"crash_20261009_0{index:05d}.log"),
                "w", encoding="utf-8",
            ) as seed:
                seed.write("stale\n")
        crash_path = self._main.log_crash(RuntimeError("unit-test-boom"))
        self.assertTrue(os.path.isfile(crash_path))
        remaining = sorted(
            name for name in os.listdir(self._tmp)
            if name.startswith("crash_") and name.endswith(".log")
        )
        self.assertEqual(len(remaining), cap)
        self.assertIn(os.path.basename(crash_path), remaining)
        self.assertEqual(remaining[-1], os.path.basename(crash_path))
        with open(crash_path, encoding="utf-8") as written:
            body = written.read()
        self.assertIn("unit-test-boom", body)

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

    async def _run_daemon(self, main, bots, *, stop_event=None) -> None:
        automatic_stop = stop_event is None
        if stop_event is None:
            stop_event = asyncio.Event()

        async def updates(_bots, _interval, *, startup_tasks):
            if automatic_stop:
                await asyncio.wait(startup_tasks)
                stop_event.set()
            else:
                await stop_event.wait()

        with (
            mock.patch.object(main, "_cli_mode_requested", return_value=False),
            mock.patch.object(main.cfg, "load_config", return_value={"update_check_interval": 300}),
            mock.patch.object(main, "create_platforms", return_value=bots),
            mock.patch.object(main.updater, "init_startup_commit"),
            mock.patch.object(main.updater, "get_current_version", return_value="test"),
            mock.patch.object(main.updater, "get_last_commit_date", return_value="test"),
            mock.patch.object(main.asyncio, "Event", return_value=stop_event),
            mock.patch.object(
                main, "update_loop", new=mock.AsyncMock(side_effect=updates),
            ),
            mock.patch.object(asyncio.get_running_loop(), "add_signal_handler"),
        ):
            await main.main()

    async def test_failed_start_cleans_started_and_partially_started_platforms(self) -> None:
        main = _load_main_module()
        first, failed, concurrent = self._bot(), self._bot(), self._bot()
        failed.start.side_effect = RuntimeError("startup failed")
        with self.assertRaisesRegex(RuntimeError, "startup failed"):
            await self._run_daemon(main, [first, failed, concurrent])
        first.stop.assert_awaited_once()
        failed.stop.assert_awaited_once()
        concurrent.start.assert_awaited_once()
        concurrent.stop.assert_awaited_once()

    async def test_shutdown_failures_do_not_skip_other_platform_cleanup(self) -> None:
        main = _load_main_module()
        first, second = self._bot(), self._bot()
        first.notify_users.side_effect = RuntimeError("send failed")
        second.stop.side_effect = RuntimeError("stop failed")
        with self.assertLogs(main.logger, level="ERROR"):
            await self._run_daemon(main, [first, second])
        first.stop.assert_awaited_once()
        second.stop.assert_awaited_once()

    async def test_cancel_replayed_after_stop_waiter_join_still_stops_platforms(self):
        main = _load_main_module()
        first, second = self._bot(), self._bot()
        original = main._cancel_platform_starts

        async def join(tasks):
            await original(tasks)
            if len(tasks) == 1 and tasks[0].get_coro().__qualname__ == "Event.wait":
                raise asyncio.CancelledError("fixture repeated shutdown cancellation")

        with (
            mock.patch.object(main, "_cancel_platform_starts", side_effect=join),
            self.assertRaises(asyncio.CancelledError),
        ):
            await self._run_daemon(main, [first, second])
        first.stop.assert_awaited_once()
        second.stop.assert_awaited_once()

    async def test_one_platform_replaying_cancel_does_not_skip_other_stops(self):
        main = _load_main_module()
        first, second = self._bot(), self._bot()
        second.stop.side_effect = asyncio.CancelledError("fixture cleanup cancellation")
        with self.assertRaises(asyncio.CancelledError):
            await self._run_daemon(main, [first, second])
        first.stop.assert_awaited_once()
        second.stop.assert_awaited_once()

    async def test_repeated_parent_cancel_keeps_all_platform_shutdowns_owned(self):
        main = _load_main_module()
        first, second = self._bot(), self._bot()
        entered = [asyncio.Event(), asyncio.Event()]
        release = asyncio.Event()
        cleaned = []
        previous_tasks = asyncio.all_tasks()

        async def stop(index):
            entered[index].set()
            await release.wait()
            cleaned.append(index)

        async def stop_first():
            await stop(0)

        async def stop_second():
            await stop(1)

        first.stop.side_effect = stop_first
        second.stop.side_effect = stop_second
        run = asyncio.create_task(main._stop_platforms([first, second]))
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered)), 1)
        run.cancel()
        await asyncio.sleep(0)
        run.cancel()
        await asyncio.sleep(0)
        self.assertFalse(run.done())
        release.set()
        with self.assertRaises(BaseExceptionGroup) as raised:
            await asyncio.wait_for(run, 1)
        self.assertTrue(all(isinstance(error, asyncio.CancelledError)
                            for error in raised.exception.exceptions))
        self.assertEqual(set(cleaned), {0, 1})
        self.assertEqual(asyncio.all_tasks() - previous_tasks, set())

    async def test_shutdown_during_start_joins_attempts_and_cleans_every_platform(self):
        main = _load_main_module()
        stop_event = asyncio.Event()
        entered = asyncio.Event()
        first, concurrent = self._bot(), self._bot()

        async def start():
            entered.set()
            await asyncio.Future()

        first.start.side_effect = start
        run = asyncio.create_task(self._run_daemon(
            main, [first, concurrent], stop_event=stop_event,
        ))
        await asyncio.wait_for(entered.wait(), 1)
        stop_event.set()
        await asyncio.wait_for(run, 1)
        first.stop.assert_awaited_once()
        first.send_startup_messages.assert_not_awaited()
        concurrent.start.assert_awaited_once()
        concurrent.stop.assert_awaited_once()

    async def test_offline_platform_does_not_block_healthy_startup_or_updates(self):
        from Cozter.backends_bot.base import PlatformNotReadyError

        main = _load_main_module()
        stop_event = asyncio.Event()
        healthy_announced = asyncio.Event()
        offline, healthy = self._bot(), self._bot()
        offline.start.side_effect = PlatformNotReadyError("dependency unavailable")
        healthy.send_startup_messages.side_effect = lambda *_: healthy_announced.set()
        with self.assertLogs(main.logger, level="WARNING"):
            run = asyncio.create_task(self._run_daemon(
                main, [offline, healthy], stop_event=stop_event,
            ))
            await asyncio.wait_for(healthy_announced.wait(), 1)
            main.update_loop.assert_awaited_once()
            self.assertFalse(run.done())
            offline.send_startup_messages.assert_not_awaited()
            stop_event.set()
            await asyncio.wait_for(run, 1)
        offline.stop.assert_awaited_once()
        healthy.stop.assert_awaited_once()

    async def test_shutdown_joins_simultaneous_start_and_backoff(self):
        from Cozter.backends_bot.base import PlatformNotReadyError

        main = _load_main_module()
        stop_event = asyncio.Event()
        starting, offline = self._bot(), self._bot()
        entered, backoff, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
        previous_tasks = asyncio.all_tasks()

        async def start():
            entered.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                cleaned.set()

        starting.start.side_effect = start
        offline.start.side_effect = PlatformNotReadyError("dependency unavailable")
        with mock.patch.object(main.logger, "warning", side_effect=lambda *_: backoff.set()):
            run = asyncio.create_task(self._run_daemon(
                main, [starting, offline], stop_event=stop_event,
            ))
            await asyncio.wait_for(asyncio.gather(entered.wait(), backoff.wait()), 1)
            stop_event.set()
            await asyncio.wait_for(run, 1)
        self.assertTrue(cleaned.is_set())
        self.assertEqual(asyncio.all_tasks() - previous_tasks, set())
        starting.stop.assert_awaited_once()
        offline.stop.assert_awaited_once()

    async def test_confirmed_update_joins_startup_before_stopping_platform(self):
        main = _load_main_module()
        entered, cleaned = asyncio.Event(), asyncio.Event()

        async def start():
            entered.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                cleaned.set()

        async def stop():
            self.assertTrue(cleaned.is_set())

        bot = self._bot()
        bot.stop.side_effect = stop
        bot.begin_update_restart = mock.AsyncMock()
        bot.cancel_update_restart = mock.AsyncMock()
        bot.has_active_turns = lambda: False
        startup = asyncio.create_task(start())
        await entered.wait()
        with (
            mock.patch.object(main.updater, "fetch_and_pull", return_value=True),
            mock.patch.object(main.updater, "install_requirements"),
            mock.patch.object(main.updater, "restart_script") as restart,
        ):
            await main._restart_after_update([bot], 0, startup_tasks=[startup])
        self.assertTrue(startup.cancelled())
        bot.stop.assert_awaited_once()
        restart.assert_called_once_with(0)

    async def test_aborted_update_keeps_dependency_startup_retry_owned(self):
        main = _load_main_module()
        bot = self._bot()
        bot.begin_update_restart = mock.AsyncMock()
        bot.cancel_update_restart = mock.AsyncMock()
        bot.has_active_turns = lambda: False
        startup = asyncio.create_task(asyncio.sleep(60))
        try:
            with mock.patch.object(main.updater, "fetch_and_pull", return_value=False):
                await main._restart_after_update([bot], 0, startup_tasks=[startup])
            self.assertFalse(startup.done())
            bot.cancel_update_restart.assert_awaited_once()
            bot.stop.assert_not_awaited()
        finally:
            startup.cancel()
            await asyncio.gather(startup, return_exceptions=True)

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


class SharedDaemonLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def _bot(self):
        from Cozter.tests.helpers import TestBot

        return TestBot(["fixture"])

    async def test_slack_closes_transport_after_shared_consumer_cleanup_fails(self):
        from Cozter.backends_bot.slack import SlackBot

        bot = SlackBot("fixture-bot", "fixture-app", ["fixture-channel"])
        failure = RuntimeError("fixture consumer cleanup failed")
        bot._stop_daemon_services = mock.AsyncMock(side_effect=failure)
        handler = SimpleNamespace(close_async=mock.AsyncMock())
        bot._handler = handler
        bot.app = object()
        with self.assertRaises(RuntimeError) as raised:
            await bot.stop()
        self.assertIs(raised.exception, failure)
        handler.close_async.assert_awaited_once()
        self.assertIsNone(bot._handler)
        self.assertIsNone(bot.app)

    async def test_telegram_closes_transport_after_shared_consumer_cleanup_fails(self):
        from Cozter.backends_bot.telegram import TelegramBot

        bot = TelegramBot("fixture-token", ["fixture"])
        failure = RuntimeError("fixture consumer cleanup failed")
        bot._stop_daemon_services = mock.AsyncMock(side_effect=failure)
        app = SimpleNamespace(
            updater=SimpleNamespace(running=True, stop=mock.AsyncMock()),
            running=True, stop=mock.AsyncMock(), shutdown=mock.AsyncMock(),
        )
        bot.app = app
        with self.assertRaises(RuntimeError) as raised:
            await bot.stop()
        self.assertIs(raised.exception, failure)
        app.updater.stop.assert_awaited_once()
        app.stop.assert_awaited_once()
        app.shutdown.assert_awaited_once()
        self.assertIsNone(bot.app)

    async def test_service_cleanup_failure_does_not_skip_other_owned_services(self):
        bot = self._bot()
        failure = RuntimeError("fixture watcher stop failed")
        bot._daemon_services_started = True
        bot._detached_launch_tasks = {"launch": set()}
        bot._attachment_tasks = {"attachment": set()}
        bot._reply_delivery_tasks = {"reply": set()}
        bot._cancel_owned_deliveries = mock.AsyncMock()
        bot.stop_detached_task_watcher = mock.AsyncMock(side_effect=failure)
        bot.stop_scheduler = mock.AsyncMock()
        bot._cancel_attachment_uploads = mock.AsyncMock()
        with self.assertRaises(RuntimeError) as raised:
            await bot._stop_daemon_services()
        self.assertIs(raised.exception, failure)
        self.assertFalse(bot._daemon_services_started)
        bot.stop_scheduler.assert_awaited_once()
        bot._cancel_attachment_uploads.assert_awaited_once_with("attachment")
        self.assertEqual(bot._cancel_owned_deliveries.await_args_list, [
            mock.call("launch", bot._detached_launch_tasks),
            mock.call("reply", bot._reply_delivery_tasks),
        ])

    async def test_aborted_update_does_not_start_services_on_offline_platform(self):
        bot = self._bot()
        bot.start_scheduler = mock.AsyncMock()
        bot._start_queue_drain = mock.Mock()
        await bot.begin_update_restart()
        await bot.cancel_update_restart()
        self.assertFalse(bot._update_restart_pending)
        bot.start_scheduler.assert_not_awaited()
        bot._start_queue_drain.assert_not_called()

    async def test_start_during_update_restores_state_but_defers_scheduler_until_abort(self):
        bot = self._bot()
        bot._update_restart_pending = True
        bot.restore_queues = mock.AsyncMock()
        bot.start_detached_task_watcher = mock.Mock()
        bot.start_scheduler = mock.AsyncMock()
        await bot._start_daemon_services()
        self.assertTrue(bot._daemon_services_started)
        bot.restore_queues.assert_awaited_once()
        bot.start_detached_task_watcher.assert_called_once()
        bot.start_scheduler.assert_not_awaited()
        await bot.cancel_update_restart()
        bot.start_scheduler.assert_awaited_once()

    async def test_cancelled_admission_settling_after_shutdown_cannot_start_consumer(self):
        from Cozter.queue_store import AdmissionResult

        bot = self._bot()
        entered, release = asyncio.Event(), threading.Event()
        accepted = []
        loop = asyncio.get_running_loop()

        def admit(uid, entry, occurrence):
            loop.call_soon_threadsafe(entered.set)
            self.assertTrue(release.wait(2))
            accepted.append(entry)
            return AdmissionResult(entry, True)

        bot._queue_store = mock.Mock(return_value=SimpleNamespace(admit=admit))
        bot._drain_message_queue = mock.AsyncMock()
        admission = asyncio.create_task(bot._admit_queue_entry(
            "fixture", "accepted work", "chat", 0,
        ))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            admission.cancel()
            await bot._stop_daemon_services()
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await admission
        await asyncio.sleep(0)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(bot._message_queues["fixture"].qsize(), 1)
        self.assertFalse(bot._queue_drain_tasks)
        bot._drain_message_queue.assert_not_awaited()

    async def test_shutdown_owns_deduplicated_drain_and_joins_its_cleanup(self):
        bot = self._bot()
        entered, cleaning, release, cleaned = (
            asyncio.Event(), asyncio.Event(), asyncio.Event(), asyncio.Event()
        )

        async def drain(_uid):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cleaning.set()
                await release.wait()
                cleaned.set()

        bot._drain_message_queue = mock.AsyncMock(side_effect=drain)
        first = bot._start_queue_drain("fixture")
        await asyncio.wait_for(entered.wait(), 1)
        self.assertIs(bot._start_queue_drain("fixture"), first)
        shutdown = asyncio.create_task(bot._stop_daemon_services())
        await asyncio.wait_for(cleaning.wait(), 1)
        self.assertTrue(bot._stopping)
        self.assertFalse(shutdown.done())
        self.assertIsNone(bot._start_queue_drain("fixture"))
        release.set()
        await asyncio.wait_for(shutdown, 1)
        self.assertTrue(cleaned.is_set())
        self.assertFalse(bot._queue_drain_tasks)
        self.assertFalse(bot._queue_drain_requests)
        bot._drain_message_queue.assert_awaited_once_with("fixture")

    async def test_wake_while_consumer_is_closing_starts_successor(self):
        bot = self._bot()
        entered, cleaning, release, continued = (
            asyncio.Event(), asyncio.Event(), asyncio.Event(), asyncio.Event()
        )
        calls = 0

        async def drain(_uid):
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                try:
                    await asyncio.Future()
                finally:
                    cleaning.set()
                    await release.wait()
            else:
                continued.set()

        bot._drain_message_queue = mock.AsyncMock(side_effect=drain)
        first = bot._start_queue_drain("fixture")
        await asyncio.wait_for(entered.wait(), 1)
        first.cancel()
        await asyncio.wait_for(cleaning.wait(), 1)
        self.assertIs(bot._start_queue_drain("fixture"), first)
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await first
        await asyncio.wait_for(continued.wait(), 1)
        await asyncio.sleep(0)
        self.assertEqual(calls, 2)
        self.assertFalse(bot._queue_drain_tasks)

    async def test_shutdown_also_joins_active_inline_turn_without_cancelling_itself(self):
        bot = self._bot()
        entered, cleaned = asyncio.Event(), asyncio.Event()

        async def inline():
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cleaned.set()

        active = asyncio.create_task(inline())
        await asyncio.wait_for(entered.wait(), 1)
        bot._running_tasks["fixture"] = active
        bot._running_tasks["caller"] = asyncio.current_task()
        await bot._stop_daemon_services()
        self.assertTrue(active.cancelled())
        self.assertTrue(cleaned.is_set())

    async def test_stopped_drain_does_not_pop_work_and_restart_opens_gate_before_restore(self):
        bot = self._bot()
        queue = bot._ensure_message_queue("fixture")
        queue.put_nowait(("kept for restart", "chat", "entry", False))
        bot._stopping = True
        await bot._drain_message_queue("fixture")
        self.assertEqual(queue.qsize(), 1)

        async def restore():
            self.assertFalse(bot._stopping)

        bot.restore_queues = mock.AsyncMock(side_effect=restore)
        bot.start_detached_task_watcher = mock.Mock()
        bot.start_scheduler = mock.AsyncMock()
        await bot._start_daemon_services()
        self.assertFalse(bot._stopping)


class PlatformStartupReadinessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.main = _load_main_module()

    def _bot(self, start):
        return SimpleNamespace(
            platform_id="fixture:test", start=start, stop=mock.AsyncMock(),
        )

    async def test_dependency_readiness_retries_with_capped_backoff_until_ready(self):
        from Cozter.backends_bot.base import PlatformNotReadyError

        bot = self._bot(mock.AsyncMock(side_effect=[
            *[PlatformNotReadyError("account still loading") for _ in range(8)],
            None,
        ]))
        with (
            mock.patch.object(
                self.main.asyncio, "wait_for", side_effect=TimeoutError,
            ) as backoff,
            self.assertLogs(self.main.logger, level="WARNING"),
        ):
            ready = await self.main._start_platform(bot, asyncio.Event())

        self.assertTrue(ready)
        self.assertEqual(bot.start.await_count, 9)
        self.assertEqual(
            [call.args[1] for call in backoff.call_args_list],
            [1, 2, 4, 8, 16, 30, 30, 30],
        )
        # Each platform attempt owns its cleanup; don't separately stop a bot
        # between attempts or tear down a successfully initialized retry.
        bot.stop.assert_not_awaited()

    async def test_failed_cleanup_is_fatal_even_when_original_failure_is_retryable(self):
        from Cozter.backends_bot.base import PlatformNotReadyError

        failure = ExceptionGroup("startup and cleanup failed", [
            PlatformNotReadyError("dependency unavailable"),
            PermissionError("cleanup state write failed"),
        ])
        bot = self._bot(mock.AsyncMock(side_effect=failure))
        with self.assertRaises(ExceptionGroup) as raised:
            await self.main._start_platform(bot, asyncio.Event())
        self.assertIs(raised.exception, failure)
        bot.start.assert_awaited_once()

    async def test_configuration_failure_does_not_retry(self):
        bot = self._bot(mock.AsyncMock(side_effect=ValueError("bad account")))
        with (
            mock.patch.object(self.main.asyncio, "wait_for") as backoff,
            self.assertRaisesRegex(ValueError, "bad account"),
        ):
            await self.main._start_platform(bot, asyncio.Event())
        bot.start.assert_awaited_once()
        backoff.assert_not_called()

    async def test_other_signal_errors_do_not_retry(self):
        from Cozter.backends_bot.signal import SignalCliError

        bot = self._bot(mock.AsyncMock(side_effect=SignalCliError("bad group")))
        with self.assertRaisesRegex(SignalCliError, "bad group"):
            await self.main._start_platform(bot, asyncio.Event())
        bot.start.assert_awaited_once()

    async def test_already_requested_stop_does_not_start_a_platform(self):
        stop_event = asyncio.Event()
        stop_event.set()
        bot = self._bot(mock.AsyncMock())
        self.assertFalse(await self.main._start_platform(bot, stop_event))
        bot.start.assert_not_awaited()

    async def test_stop_during_start_joins_attempt_cleanup(self):
        entered = asyncio.Event()
        cleaned = asyncio.Event()
        stop_event = asyncio.Event()
        owned_attempts = []

        async def start():
            owned_attempts.append(asyncio.current_task())
            entered.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                cleaned.set()

        bot = self._bot(start)
        run = asyncio.create_task(self.main._start_platform(bot, stop_event))
        await asyncio.wait_for(entered.wait(), 1)
        stop_event.set()
        self.assertFalse(await asyncio.wait_for(run, 1))
        self.assertTrue(cleaned.is_set())
        self.assertTrue(all(task.done() for task in owned_attempts))

    async def test_external_cancellation_joins_start_and_stop_waiter(self):
        entered = asyncio.Event()
        cleaned = asyncio.Event()
        previous_tasks = asyncio.all_tasks()

        async def start():
            entered.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                cleaned.set()

        run = asyncio.create_task(
            self.main._start_platform(self._bot(start), asyncio.Event()),
        )
        await asyncio.wait_for(entered.wait(), 1)
        run.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await run
        self.assertTrue(cleaned.is_set())
        self.assertEqual(asyncio.all_tasks() - previous_tasks, set())

    async def test_repeated_cancellation_does_not_interrupt_adapter_cleanup(self):
        entered, cleaning, release, cleaned = (
            asyncio.Event(), asyncio.Event(), asyncio.Event(), asyncio.Event()
        )
        previous_tasks = asyncio.all_tasks()

        async def start():
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cleaning.set()
                await release.wait()
                cleaned.set()

        run = asyncio.create_task(
            self.main._start_platform(self._bot(start), asyncio.Event()),
        )
        await asyncio.wait_for(entered.wait(), 1)
        run.cancel()
        await asyncio.wait_for(cleaning.wait(), 1)
        run.cancel()
        await asyncio.sleep(0)
        self.assertFalse(run.done())
        self.assertFalse(cleaned.is_set())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(run, 1)
        self.assertTrue(cleaned.is_set())
        self.assertEqual(asyncio.all_tasks() - previous_tasks, set())

    async def test_stop_during_backoff_does_not_wait_or_retry(self):
        from Cozter.backends_bot.base import PlatformNotReadyError

        in_backoff = asyncio.Event()
        stop_event = asyncio.Event()
        bot = self._bot(mock.AsyncMock(side_effect=PlatformNotReadyError("loading")))
        previous_tasks = asyncio.all_tasks()
        with mock.patch.object(
            self.main.logger, "warning", side_effect=lambda *_: in_backoff.set(),
        ):
            run = asyncio.create_task(self.main._start_platform(bot, stop_event))
            await asyncio.wait_for(in_backoff.wait(), 1)
            stop_event.set()
            self.assertFalse(await asyncio.wait_for(run, 1))
        bot.start.assert_awaited_once()
        self.assertEqual(asyncio.all_tasks() - previous_tasks, set())

    async def test_external_cancellation_during_backoff_leaves_no_waiter(self):
        from Cozter.backends_bot.base import PlatformNotReadyError

        in_backoff = asyncio.Event()
        bot = self._bot(mock.AsyncMock(side_effect=PlatformNotReadyError("loading")))
        previous_tasks = asyncio.all_tasks()
        with mock.patch.object(
            self.main.logger, "warning", side_effect=lambda *_: in_backoff.set(),
        ):
            run = asyncio.create_task(
                self.main._start_platform(bot, asyncio.Event()),
            )
            await asyncio.wait_for(in_backoff.wait(), 1)
            run.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await run
        bot.start.assert_awaited_once()
        self.assertEqual(asyncio.all_tasks() - previous_tasks, set())


if __name__ == "__main__":
    unittest.main()
