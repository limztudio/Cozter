"""Only canonical native launch APIs and module entrypoints remain."""

import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from Cozter import process_launch
from Cozter.backends_agent import base, claude_code, codex, copilot, grok
from Cozter.tests.test_runtime_diagnostics import load_main_module


class NativeLaunchContractTests(unittest.TestCase):
    def test_native_backends_use_canonical_launchers_without_base_aliases(self):
        for module in (claude_code, codex, copilot):
            self.assertIs(module.create_prompt_subprocess, process_launch.create_prompt_subprocess)
        for module in (claude_code, grok):
            self.assertIs(module.create_captured_subprocess, process_launch.create_captured_subprocess)
        for name in (
            "create_prompt_subprocess", "create_captured_subprocess",
            "_create_piped_subprocess", "_await_process_launch",
            "_reap_failed_prompt_subprocess", "_reap_cancelled_process_launch",
        ):
            self.assertFalse(hasattr(base, name), name)

    def test_shared_helpers_have_no_forwarding_callback_arguments(self):
        self.assertEqual(list(inspect.signature(process_launch.await_process_launch).parameters),
                         ["launch", "cancel_cleanup"])
        self.assertEqual(list(inspect.signature(process_launch.reap_failed_subprocess).parameters),
                         ["proc"])
        self.assertNotIn("mark_group", inspect.signature(process_launch.create_piped_subprocess).parameters)

    def test_single_dash_cli_alias_is_rejected_and_current_flag_is_accepted(self):
        main = load_main_module()
        main._validate_launch_args(["--cli"])
        with self.assertRaises(SystemExit) as raised:
            main._validate_launch_args(["-cli"])
        self.assertEqual(raised.exception.code, 2)

    def run_entrypoint(self, args):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        with tempfile.TemporaryDirectory() as directory:
            return subprocess.run(
                [sys.executable, "-B", *args], cwd=directory, env=env,
                capture_output=True, text=True, timeout=10,
            )

    def test_module_help_still_works_before_bootstrap(self):
        child = self.run_entrypoint(["-m", "Cozter", "--help"])
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIn("--cli", child.stdout)
        self.assertNotIn("-cli,", child.stdout)

    def test_directory_and_script_entrypoints_reject_before_relaunch(self):
        package = Path(__file__).resolve().parents[1]
        for entrypoint in (package, package / "__main__.py"):
            with self.subTest(entrypoint=entrypoint.name):
                child = self.run_entrypoint([str(entrypoint), "--help"])
                self.assertEqual(child.returncode, 2)
                self.assertEqual(child.stdout, "")
                self.assertIn("python -m Cozter", child.stderr)


if __name__ == "__main__":
    unittest.main()
