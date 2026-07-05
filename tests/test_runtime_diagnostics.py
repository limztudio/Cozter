"""Behavioral tests for the runtime-diagnostics plumbing in ``__main__``.

``__main__`` runs ``pip install -r requirements.txt`` at import time via
``_install_deps()``, which the rest of the suite deliberately avoids by not
importing it. We neutralize that call before importing so these tests stay
hermetic and don't hit the network, while still exercising the real
``dump_runtime_diagnostics`` / ``_enable_faulthandler`` code paths.
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
    subprocess.check_call = lambda *a, **kw: 0
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


if __name__ == "__main__":
    unittest.main()
