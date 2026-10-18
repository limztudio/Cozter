"""Blocking file mutations yield the loop without yielding write ownership."""

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from Cozter import agent_tools
from Cozter.agent_tools.base import get_tool_permission, tool_permission_scope
from Cozter.agent_tools.owned_worker import run_owned_worker


class OwnedWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_result_failure_and_permission_context_are_preserved(self):
        with tool_permission_scope("full"):
            self.assertEqual(await run_owned_worker(get_tool_permission), "full")
            self.assertEqual(await run_owned_worker(lambda value, *, suffix: value + suffix, "a", suffix="b"), "ab")
        self.assertEqual(get_tool_permission(), "auto")

        def fail():
            raise OSError("fixture write failed")

        with self.assertRaisesRegex(OSError, "fixture write failed"):
            await run_owned_worker(fail)

    async def test_preexisting_cancel_never_submits_the_operation(self):
        calls = []

        async def caller():
            asyncio.current_task().cancel()
            with self.assertRaises(asyncio.CancelledError):
                await run_owned_worker(lambda: calls.append("started"))

        await asyncio.gather(asyncio.create_task(caller()), return_exceptions=True)
        self.assertEqual(calls, [])

    async def test_repeated_cancel_settles_and_consumes_late_failure(self):
        loop = asyncio.get_running_loop()
        diagnostics = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))
        self.addCleanup(loop.set_exception_handler, previous_handler)
        for fail in (False, True):
            with self.subTest(fail=fail):
                started = threading.Event()
                release = threading.Event()
                settled = threading.Event()

                def operation():
                    started.set()
                    try:
                        if not release.wait(5):
                            raise AssertionError("caller never released mutation")
                        if fail:
                            raise OSError("late write failed")
                        return "finished"
                    finally:
                        settled.set()

                task = asyncio.create_task(run_owned_worker(operation))
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 5))
                    task.cancel("first cancellation")
                    await asyncio.sleep(0)
                    self.assertFalse(task.done())
                    task.cancel("second cancellation")
                    await asyncio.sleep(0)
                    self.assertFalse(task.done())
                    release.set()
                    with self.assertRaises(asyncio.CancelledError) as caught:
                        await task
                    self.assertEqual(caught.exception.args, ("first cancellation",))
                    self.assertTrue(settled.is_set())
                finally:
                    release.set()
                    await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
        self.assertEqual(diagnostics, [])

    async def test_worker_has_no_child_task_for_shutdown_to_cancel(self):
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        before = asyncio.all_tasks()

        def operation():
            started.set()
            try:
                if not release.wait(5):
                    raise AssertionError("caller never released mutation")
            finally:
                finished.set()

        task = asyncio.create_task(run_owned_worker(operation))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 5))
            # asyncio.run cancels every Task at shutdown. There must be no
            # child to_thread Task whose cancellation abandons the real write.
            self.assertEqual(asyncio.all_tasks() - before, {task})
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(finished.is_set())
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)


class FileMutationDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_outer_cancel_acknowledges_only_after_write_settles(self):
        loop = asyncio.get_running_loop()
        diagnostics = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))
        self.addCleanup(loop.set_exception_handler, previous_handler)
        tool = agent_tools.TOOLS_BY_NAME["write_file"]
        original_run = tool.run

        for late_failure in (False, True):
            with self.subTest(late_failure=late_failure), tempfile.TemporaryDirectory() as workspace:
                started = threading.Event()
                release = threading.Event()
                finished = threading.Event()
                inner_tasks = []
                attempted = []
                emitted = []

                async def observe_run(path, args):
                    inner_tasks.append(asyncio.current_task())
                    return await original_run(path, args)

                def write_body(_workspace, args):
                    attempted.append(args["path"])
                    started.set()
                    try:
                        if not release.wait(5):
                            raise AssertionError("caller never released the first write")
                        Path(workspace, args["path"]).write_text("settled", encoding="utf-8")
                        if late_failure:
                            raise OSError("failure after fixture publication")
                        return "normal success must not escape cancellation"
                    finally:
                        finished.set()

                async def provider():
                    for path in ("first.txt", "second.txt"):
                        await agent_tools.execute_tool(
                            "write_file", {"path": path}, workspace, "full", emitted.append,
                        )

                with (
                    mock.patch.object(tool, "run", observe_run),
                    mock.patch.object(tool, "_run_sync", write_body),
                ):
                    task = asyncio.create_task(provider())
                    try:
                        self.assertTrue(await asyncio.to_thread(started.wait, 5))
                        task.cancel("first outer cancellation")
                        async with asyncio.timeout(5):
                            while not inner_tasks[0].cancelling():
                                await asyncio.sleep(0)
                        task.cancel("second outer cancellation")
                        await asyncio.sleep(0)
                        await asyncio.sleep(0)
                        self.assertFalse(task.done(), "outer caller abandoned its still-writing tool")
                        self.assertFalse(finished.is_set())
                        release.set()
                        with self.assertRaises(asyncio.CancelledError) as caught:
                            await task
                        self.assertEqual(caught.exception.args, ("first outer cancellation",))
                        self.assertTrue(finished.is_set())
                        self.assertEqual(Path(workspace, "first.txt").read_text(), "settled")
                        self.assertEqual(attempted, ["first.txt"])
                        self.assertFalse(Path(workspace, "second.txt").exists())
                        self.assertFalse(any(event["type"] == "tool_result" for event in emitted))
                    finally:
                        release.set()
                        await asyncio.gather(task, *inner_tasks, return_exceptions=True)
        await asyncio.sleep(0)
        self.assertEqual(diagnostics, [])

    async def test_all_mutation_builtins_run_their_sync_body_off_the_loop(self):
        loop_thread = threading.get_ident()
        for name in (
            "write_file", "edit_file", "multi_edit", "apply_patch", "copy_file",
            "move_file", "delete_file", "make_dir",
        ):
            with self.subTest(tool=name):
                tool = agent_tools.TOOLS_BY_NAME[name]
                observed = []

                def body(_workspace, _args):
                    observed.append(threading.get_ident())
                    return "fixture result"

                with mock.patch.object(tool, "_run_sync", body):
                    self.assertEqual(await tool.run("fixture", {}), "fixture result")
                self.assertEqual(len(observed), 1)
                self.assertNotEqual(observed[0], loop_thread)

    async def test_dispatcher_waits_for_the_actual_write_without_a_deadline(self):
        started = threading.Event()
        release = threading.Event()
        inner_tasks = []
        order = []
        results = []
        tool = agent_tools.TOOLS_BY_NAME["write_file"]
        original_run = tool.run

        async def observe_run(workspace, args):
            inner_tasks.append(asyncio.current_task())
            return await original_run(workspace, args)

        with tempfile.TemporaryDirectory() as workspace:
            def write_body(_workspace, args):
                if args["path"] == "first.txt":
                    started.set()
                    if not release.wait(5):
                        raise AssertionError("caller never released first write")
                Path(workspace, args["path"]).write_text(args["content"], encoding="utf-8")
                order.append(args["path"])
                return "Wrote fixture content"

            async def provider():
                for path in ("first.txt", "second.txt"):
                    results.append(await agent_tools.execute_tool(
                        "write_file", {"path": path, "content": path},
                        workspace, "full", lambda event: None,
                    ))

            with (
                mock.patch.object(tool, "run", observe_run),
                mock.patch.object(tool, "_run_sync", write_body),
            ):
                task = asyncio.create_task(provider())
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 5))
                    for _ in range(10):
                        await asyncio.sleep(0)
                    self.assertEqual(inner_tasks[0].cancelling(), 0)
                    self.assertFalse(task.done())
                    self.assertEqual(order, [])
                    self.assertEqual(results, [])
                    self.assertFalse(Path(workspace, "second.txt").exists())
                    release.set()
                    await task
                    self.assertEqual(order, ["first.txt", "second.txt"])
                    self.assertEqual(results, ["Wrote fixture content"] * 2)
                finally:
                    release.set()
                    await asyncio.gather(task, return_exceptions=True)
