"""Pure calculator CPU work does not occupy the asynchronous tool dispatcher."""

import asyncio
import threading
import unittest
from unittest import mock

from Cozter.agent_tools.plugins import calculator


class CalculatorResponsivenessTests(unittest.IsolatedAsyncioTestCase):
    async def test_loop_progresses_while_evaluation_is_held_on_worker(self) -> None:
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        loop_thread = threading.get_ident()
        worker_threads: list[int] = []

        def held_evaluate(_tree):
            worker_threads.append(threading.get_ident())
            entered.set()
            try:
                if not release.wait(3):
                    raise RuntimeError("test did not release calculator")
                return 7
            finally:
                finished.set()

        with mock.patch.object(calculator, "_evaluate", side_effect=held_evaluate):
            task = asyncio.create_task(calculator.CalculatorTool().run(".", {"expression": "3 + 4"}))
            try:
                self.assertTrue(await asyncio.wait_for(asyncio.to_thread(entered.wait, 1), 2))
                # This callback must run before the held computation completes.
                progressed = asyncio.Event()
                asyncio.get_running_loop().call_soon(progressed.set)
                await asyncio.wait_for(progressed.wait(), 1)
                self.assertFalse(finished.is_set())
                self.assertFalse(task.done())
                self.assertNotEqual(worker_threads, [loop_thread])
                release.set()
                self.assertEqual(await asyncio.wait_for(task, 2), "7")
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)

    async def test_cancellation_can_return_while_pure_evaluation_finishes(self) -> None:
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()

        def held_evaluate(_tree):
            entered.set()
            try:
                release.wait(3)
                return 11
            finally:
                finished.set()

        with mock.patch.object(calculator, "_evaluate", side_effect=held_evaluate):
            task = asyncio.create_task(calculator.CalculatorTool().run(".", {"expression": "5 + 6"}))
            try:
                self.assertTrue(await asyncio.wait_for(asyncio.to_thread(entered.wait, 1), 2))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 1)
                self.assertFalse(finished.is_set())
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
                self.assertTrue(await asyncio.wait_for(asyncio.to_thread(finished.wait, 1), 2))

    async def test_parse_format_and_error_behavior_are_preserved(self) -> None:
        tool = calculator.CalculatorTool()
        cases = {
            "2 + 3 * 4": "14",
            "abs((-1)**0.5)": "1",
            "1 / 0": "Error: division by zero",
            "(10**500)**2 % 17": "16",
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(await tool.run(".", {"expression": expression}), expected)
        self.assertEqual(await tool.run(".", {"expression": "x" * 1001}), "Error: expression exceeds 1000 characters")


if __name__ == "__main__":
    unittest.main()
