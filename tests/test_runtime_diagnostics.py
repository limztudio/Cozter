"""Behavioral tests for the runtime-diagnostics plumbing in ``__main__``.

``__main__`` checks whether a fresh virtual environment is missing required
runtime modules at import time. We neutralize its dependency-repair call
before importing so these tests stay hermetic and don't hit the network, while
still exercising the real ``dump_runtime_diagnostics`` / ``_enable_faulthandler``
code paths.
"""

import asyncio
import contextlib
import faulthandler
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


def _load_main_module():
    """Import ``Cozter.__main__`` with the import-time pip install disabled."""
    # Workspace checkout must win over the live AutoStart install if both are
    # on sys.path (the dev host has the latter earlier on the path).
    workspace_pkg_parent = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sys.path = [
        p for p in sys.path
        if os.path.abspath(p) not in {
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
            mock.patch.object(main, "find_spec", return_value=object()) as find,
            mock.patch.object(main.subprocess, "check_call") as install,
        ):
            main._install_deps()

        self.assertEqual(find.call_count, len(main._REQUIRED_RUNTIME_MODULES))
        install.assert_not_called()

    def test_dependency_bootstrap_installs_only_when_a_module_is_missing(self) -> None:
        main = _load_main_module()
        missing_module = main._REQUIRED_RUNTIME_MODULES[-1]

        def find(module: str):
            return None if module == missing_module else object()

        with (
            mock.patch.object(main, "find_spec", side_effect=find),
            mock.patch.object(main.subprocess, "check_call") as install,
        ):
            main._install_deps()

        install.assert_called_once()
        args, kwargs = install.call_args
        self.assertEqual(args[0][:4], [main.sys.executable, "-m", "pip", "install"])
        self.assertEqual(kwargs["timeout"], main._DEPENDENCY_INSTALL_TIMEOUT_SEC)

    def test_dependency_bootstrap_reports_a_bounded_install_timeout(self) -> None:
        main = _load_main_module()
        with (
            mock.patch.object(main, "find_spec", return_value=None),
            mock.patch.object(
                main.subprocess,
                "check_call",
                side_effect=subprocess.TimeoutExpired("pip", 1),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "Timed out installing"):
                main._install_deps()

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


class DumpRuntimeDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._main = _load_main_module()
        # Point LOG_DIR at our temp dir and reset the cached file handle so
        # _get_dump_file opens a fresh diagnostics.log inside it.
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
        ) as f:
            return f.read()

    def test_dump_writes_header_reason_tasks_and_threads(self):
        # Running inside a real event loop lets asyncio.all_tasks() see the
        # current task, so the "asyncio tasks" section is non-empty.
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

        # Should not raise; the error is captured inline.
        self._main.dump_runtime_diagnostics([_BrokenBot()])
        body = self._read_dump()
        self.assertIn("test:broken", body)
        self.assertIn("boom", body)

    def test_bot_label_falls_back_to_class_name(self):
        class _NoId:
            pass

        # platform_id access raises, so we fall back to the class name.
        self.assertEqual(
            self._main._bot_label(_NoId()), _NoId().__class__.__name__,
        )

    def test_enable_faulthandler_is_best_effort_and_silent(self):
        # With dump_traceback_interval=0 (default) it must still enable the
        # crash handler without raising. Runs against the real faulthandler;
        # a failure here would surface as an exception, not a silent skip.
        self._main._enable_faulthandler()

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
            ) as check_mock,
        ):
            with self.assertRaises(asyncio.CancelledError):
                await main.update_loop([bot], interval=1)

        check_mock.assert_called_once_with()
        # No update found, so intake must not be paused for a restart.
        self.assertEqual(bot.restart_calls, 0)


if __name__ == "__main__":
    unittest.main()
