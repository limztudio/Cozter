"""Reader concurrency, writer barriers, and whole-wave cancellation, offline."""

import asyncio
import tempfile
import unittest
from unittest import mock

from Cozter import agent, flexible, workspace
from Cozter.backends_agent.base import AgentResult


class FlexibleParallelRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = self.tmp.name
        for tier in flexible.TIERS:
            workspace.set_flexible_backend_name(self.ws, tier, "meta")
        self.merge_prompt = ""
        self.plan = ("[PLAN]\n1. [low] [parallel-read] inspect A\n"
                     "2. [mid] [parallel-read] inspect B\n"
                     "3. [high] repair and verify\n[/PLAN]")

    async def internal(self, *args, **kwargs):
        if kwargs["label"] == "Flexible planner":
            return self.plan
        self.merge_prompt = args[2]
        return "Complete"

    async def run_flexible(self, drive, *, queue=None, injected=None, limit=3):
        with (mock.patch.object(agent, "run_internal_backend", self.internal),
              mock.patch.object(agent, "_drive_backend", drive),
              mock.patch.object(agent.config, "get_flexible_max_parallel_workers", return_value=limit)):
            return await agent._run_flexible(
                "context", "request", self.ws, approval="full", effort=100,
                collaborative=False, summary_backend_name="meta", summary_model="summary",
                on_event=None, inject_queue=queue, injected=injected if injected is not None else [],
            )

    async def test_readers_overlap_writer_waits_and_reports_keep_plan_order(self):
        started = set()
        both = asyncio.Event()
        finished = set()
        calls = []

        async def drive(_backend, _ws, prompt, _model, approval, **kwargs):
            index = int(prompt.split("[Sub-task ")[1].split("/")[0])
            calls.append((index, approval, kwargs["effort"]))
            if index < 3:
                self.assertIsNone(kwargs["inject_queue"])
                started.add(index)
                if len(started) == 2:
                    both.set()
                await asyncio.wait_for(both.wait(), 1)
                if index == 1:
                    await asyncio.sleep(.02)  # report 2 completes first
            else:
                self.assertEqual(finished, {1, 2})
                self.assertLess(prompt.index("report-1"), prompt.index("report-2"))
            finished.add(index)
            return AgentResult(text=f"report-{index}", usage={"input_tokens": index}), False

        result, restart = await self.run_flexible(drive)
        self.assertFalse(restart)
        self.assertEqual(calls, [(1, "confirm", 100), (2, "confirm", 100), (3, "full", 100)])
        self.assertEqual(result.usage, {"input_tokens": 6})
        self.assertLess(self.merge_prompt.index("report-1"), self.merge_prompt.index("report-2"))

    async def test_unmarked_writers_and_disabled_parallelism_stay_serial(self):
        async def check(plan, limit):
            self.plan = plan
            active = 0
            peak = 0

            async def drive(*args, **kwargs):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(.001)
                active -= 1
                return AgentResult(text="done"), False

            await self.run_flexible(drive, limit=limit)
            self.assertEqual(peak, 1)

        await check(self.plan, 1)
        await check("[PLAN]\n1. [low] edit A\n2. [high] edit B\n[/PLAN]", 3)

    async def test_unknown_backend_capability_falls_back_to_serial_original_approval(self):
        backend = agent.backends_agent.get_backend("meta")
        seen = []

        async def drive(*args, **kwargs):
            seen.append(args[4])
            self.assertIsNotNone(kwargs.get("injected"))
            return AgentResult(text="done"), False

        with mock.patch.object(backend, "supports_read_only_workers", False):
            await self.run_flexible(drive)
        self.assertEqual(seen, ["full", "full", "full"])

    async def test_injection_cancels_all_readers_and_does_not_run_writer(self):
        queue = asyncio.Queue()
        injected = []
        started = set()
        stopped = set()

        async def drive(*args, **kwargs):
            index = int(args[2].split("[Sub-task ")[1].split("/")[0])
            started.add(index)
            if len(started) == 2:
                queue.put_nowait("added requirement")
            try:
                await asyncio.Event().wait()
            finally:
                stopped.add(index)

        _, restart = await asyncio.wait_for(self.run_flexible(drive, queue=queue, injected=injected), 2)
        self.assertTrue(restart)
        self.assertEqual(started, {1, 2})
        self.assertEqual(stopped, started)
        self.assertEqual(injected, ["added requirement"])
        self.assertEqual(queue.qsize(), 0)

    async def test_deadline_retains_finished_reader_and_pending_sibling(self):
        first_done = asyncio.Event()
        stopped = asyncio.Event()

        async def drive(*args, **kwargs):
            if "Your task: inspect A" in args[2]:
                first_done.set()
                return AgentResult(text="A findings", usage={"input_tokens": 7}), False
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        with agent.turn_budget() as budget:
            task = asyncio.create_task(self.run_flexible(drive))
            await asyncio.wait_for(first_done.wait(), 1)
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(stopped.is_set())
            self.assertIn(("inspect A", "A findings"), budget.reports)
            self.assertNotIn("inspect A", budget.pending)
            self.assertIn("inspect B", budget.pending)
            self.assertIn("repair and verify", budget.pending)
            self.assertEqual(budget.output.usage, {"input_tokens": 7})

    async def test_worker_budget_limits_wave_without_losing_finished_reports(self):
        calls = []

        async def drive(*args, **kwargs):
            calls.append(args[4])
            return AgentResult(text="findings"), False

        with agent.turn_budget() as budget:
            budget.max_workers = 1
            with self.assertRaises(agent._TurnBudgetExhaustedError):
                await self.run_flexible(drive)
            self.assertEqual(budget.workers_started, 1)
            self.assertEqual(len(budget.reports), 1)
            self.assertEqual(calls, ["confirm"])

    async def test_launch_failure_stops_sibling_before_propagating(self):
        ready = asyncio.Event()
        stopped = asyncio.Event()

        async def fail():
            await ready.wait()
            raise RuntimeError("launch failed")

        async def sibling():
            ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        with self.assertRaisesRegex(RuntimeError, "launch failed"):
            await agent._run_parallel_flexible_workers([fail(), sibling()], None, [])
        self.assertTrue(stopped.is_set())

    async def test_followup_cannot_repeat_existing_reader_under_serial_metadata(self):
        calls = []

        async def drive(*args, **kwargs):
            calls.append(args[2])
            return AgentResult(text="findings\n[followup:low] inspect A"), False

        await self.run_flexible(drive)
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
