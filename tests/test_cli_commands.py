"""CLI picker input must finish before subsequent lines change its handler."""

import asyncio
import unittest
from unittest import mock

from Cozter.backends_bot.cli import CliBot


class CliCommandOrderingTests(unittest.IsolatedAsyncioTestCase):
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
