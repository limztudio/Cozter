"""Behavioral regressions found by the checkpoint-three complete tools audit."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from Cozter import agent_tools, utils
from Cozter.agent_tools.builtin import bash


class ToolCallParsingTests(unittest.TestCase):
    def test_excessively_nested_model_arguments_are_rejected_cleanly(self) -> None:
        raw = '{"nested":' + '[' * 2_000 + '0' + ']' * 2_000 + '}'
        # Decoder recursion limits vary among supported Python versions.
        with mock.patch.object(agent_tools.json, "loads", side_effect=RecursionError):
            self.assertEqual(agent_tools.parse_openai_call({
                "function": {"name": "read_file", "arguments": raw},
            }), ("read_file", {}))

    def test_integer_digit_limit_in_model_arguments_is_rejected_cleanly(self) -> None:
        raw = '{"number":' + '9' * 5_000 + '}'
        with mock.patch.object(agent_tools.json, "loads", side_effect=ValueError("integer digit limit")):
            self.assertEqual(agent_tools.parse_openai_call({
                "function": {"name": "calculator", "arguments": raw},
            }), ("calculator", {}))

    def test_plugin_cli_reports_decoder_limits_without_a_traceback(self) -> None:
        package_parent = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        # Python 3.14's decoder accepts deeper nesting than older versions;
        # the integer limit is explicitly fixed for this native CLI fixture.
        for raw in ('{"number":' + '9' * 5_000 + '}',):
            with self.subTest(raw_length=len(raw)):
                result = subprocess.run(
                    [sys.executable, "-X", "int_max_str_digits=4300", "-m",
                     "Cozter.agent_tools.plugins.current_time", raw],
                    cwd=package_parent, capture_output=True, text=True, timeout=20,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertTrue(result.stderr.startswith("Error: invalid JSON args:"), result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(result.stdout, "")


class BashLaunchOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_cancel_during_launch_reaps_the_started_process(self) -> None:
        created, handover = asyncio.Event(), asyncio.Event()
        spawned: list[asyncio.subprocess.Process] = []
        original_spawn = asyncio.create_subprocess_exec

        async def delayed_spawn(*args, **kwargs):
            proc = await original_spawn(*args, **kwargs)
            spawned.append(proc)
            created.set()
            await handover.wait()
            return proc

        executable = getattr(sys, "_base_executable", None) or sys.executable
        with (
            tempfile.TemporaryDirectory() as workspace,
            mock.patch.object(bash, "_find_shell", return_value=[executable, "-u", "-c"]),
            mock.patch.object(asyncio, "create_subprocess_exec", delayed_spawn),
        ):
            before = asyncio.all_tasks()
            task = asyncio.create_task(bash.BashTool().run(workspace, {
                "command": "import time; time.sleep(30)",
            }))
            try:
                await asyncio.wait_for(created.wait(), timeout=5)
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done(), "cancel returned before process ownership settled")
                handover.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=5)
                self.assertIsNotNone(spawned[0].returncode)
                await asyncio.sleep(0)
                self.assertFalse(asyncio.all_tasks() - before)
            finally:
                handover.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                for proc in spawned:
                    if proc.returncode is None:
                        await utils.kill_and_wait(proc)
                    utils.close_subprocess_pipe(proc, 1)


if __name__ == "__main__":
    unittest.main()
