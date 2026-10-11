"""CLI picker input must finish before subsequent lines change its handler."""

import asyncio
import os
from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock

from Cozter.backends_bot.cli import CliBot
from Cozter.utils import close_subprocess_pipe, kill_and_wait


class CliCommandOrderingTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(shutil.which("git"), "requires a local Git executable")
    async def test_version_returns_while_native_input_reader_waits_for_next_line(self):
        """Inherited stdin must not wedge Git while the CLI reader awaits input."""
        package_parent = str(Path(__file__).resolve().parents[2])
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=package_parent)
        child = (
            "import asyncio\n"
            "from Cozter.backends_bot.cli import CliBot\n"
            "asyncio.run(CliBot()._input_loop())\n"
        )
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "-c", child,
            cwd=package_parent, env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            proc.stdin.write(b"/version\n")
            await proc.stdin.drain()
            first = await asyncio.wait_for(proc.stdout.readline(), timeout=8)
            self.assertIn(b"Version:", first)
            second = await asyncio.wait_for(proc.stdout.readline(), timeout=8)
            self.assertIn(b"Updated:", second)
            # The response was checked with stdin still open. Now let the
            # reader observe EOF so its daemon thread ends before shutdown.
            proc.stdin.close()
            await asyncio.wait_for(proc.communicate(), timeout=8)
            self.assertEqual(proc.returncode, 0)
        finally:
            if proc.stdin is not None:
                proc.stdin.close()
            if proc.returncode is None:
                await kill_and_wait(proc)
            close_subprocess_pipe(proc, 1)
            close_subprocess_pipe(proc, 2)

    async def test_picker_answer_finishes_before_next_command(self):
        bot = CliBot()
        answers = []

        async def receive(ctx):
            await asyncio.sleep(0)
            answers.append(ctx.text)

        bot._expect_input("local", receive)
        with mock.patch("Cozter.backends_bot.cli.create_background_task") as schedule:
            await bot._handle_line("llama")
        self.assertEqual(answers, ["llama"])
        self.assertNotIn("local", bot._pending_input)
        schedule.assert_not_called()

    async def test_backslash_command_runs_before_next_picker_answer(self):
        bot = CliBot()
        answers = []

        async def receive(ctx):
            answers.append(ctx.text)

        async def show(_bot, _ctx):
            bot._expect_input("local", receive)

        with mock.patch.object(bot, "_COMMANDS", {"model": show}):
            await bot._handle_line(r"\model")
            await bot._handle_line("smoke-model")
        self.assertEqual(answers, ["smoke-model"])

    async def test_agent_text_stays_concurrent_with_stop_command(self):
        bot = CliBot()
        entered = asyncio.Event()
        release = asyncio.Event()
        tasks = []

        async def dispatch(_ctx):
            entered.set()
            await release.wait()

        def start(coro, **_kwargs):
            task = asyncio.create_task(coro)
            tasks.append(task)
            return task

        with (
            mock.patch.object(bot, "dispatch_text", side_effect=dispatch),
            mock.patch("Cozter.backends_bot.cli.create_background_task", side_effect=start),
        ):
            await bot._handle_line("long task")
            await asyncio.wait_for(entered.wait(), timeout=1)
            self.assertFalse(tasks[0].done())
            release.set()
            await asyncio.gather(*tasks)


if __name__ == "__main__":
    unittest.main()
