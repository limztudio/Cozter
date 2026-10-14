"""A quiet command emits liveness metadata without losing output or cancellation."""

import asyncio
import tempfile
import unittest
from unittest import mock

from Cozter.agent_tools.builtin import bash
from Cozter.tests.helpers import create_python_script_process
from Cozter.utils import close_subprocess_pipe, kill_and_wait


class BashProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_quiet_command_survives_multiple_heartbeats_and_keeps_its_output(self):
        proc = await create_python_script_process(
            'import time;time.sleep(.18);print("completed output",flush=True)'
        )
        try:
            with (mock.patch.object(bash, "_BASH_PROGRESS_INTERVAL_SEC", .02),
                  self.assertLogs("Cozter.performance", level="INFO") as capture):
                output, truncated, incomplete = await asyncio.wait_for(
                    bash._read_command_output(proc, 4096), 3,
                )
            self.assertIn(b"completed output", output)
            self.assertFalse(truncated)
            self.assertFalse(incomplete)
            self.assertGreaterEqual(len(capture.output), 2)
            self.assertTrue(all("outcome=running" in line for line in capture.output))
            self.assertTrue(all("stage=tool_wait tool=bash pid=" in line for line in capture.output))
        finally:
            await kill_and_wait(proc)
            close_subprocess_pipe(proc, 1)
            close_subprocess_pipe(proc, 2)

    async def test_heartbeat_wait_does_not_delay_tool_cancellation(self):
        tool = bash.BashTool()
        proc = await create_python_script_process('import time;time.sleep(30)')
        try:
            with tempfile.TemporaryDirectory() as ws:
                with (mock.patch.object(bash, "_BASH_PROGRESS_INTERVAL_SEC", .02),
                      mock.patch.object(bash, "_spawn_command", return_value=proc) as spawn):
                    command = asyncio.create_task(tool.run(ws, {"command": "owned quiet child"}))
                    await asyncio.sleep(.05)
                    spawn.assert_awaited_once()
                    command.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(command, 3)
                    self.assertIsNotNone(proc.returncode)
        finally:
            await kill_and_wait(proc)
            close_subprocess_pipe(proc, 1)
            close_subprocess_pipe(proc, 2)
