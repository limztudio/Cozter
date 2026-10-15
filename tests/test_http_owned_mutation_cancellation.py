"""Repeated HTTP-worker cancellation cannot abandon a dispatched disk write."""

import asyncio
import copy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent_tools
from Cozter.backends_agent import _openai_agent as oa


class _Record:
    def __init__(self):
        self.values = {}

    def get(self, key, default=None):
        return copy.deepcopy(self.values.get(key, default))

    async def save(self, **values):
        self.values.update(copy.deepcopy(values))


class _Backend(oa.OpenAIChatBackend):
    name = "offline-owned-write-cancellation"

    def _chat_endpoint(self):
        return "http://offline.invalid/chat/completions"


def _call(path):
    return {
        "id": path, "type": "function",
        "function": {"name": "write_file", "arguments": json.dumps({
            "path": path, "content": path,
        })},
    }


class HttpOwnedMutationCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_cancel_settles_write_and_recovers_unknown_intent_without_replay(self):
        entered = threading.Event()
        release = threading.Event()
        settled = threading.Event()
        tool_tasks = []
        record = _Record()
        tool = agent_tools._BY_NAME["write_file"]
        original_run = tool.run
        events = []

        async def observe_run(workspace, args):
            tool_tasks.append(asyncio.current_task())
            return await original_run(workspace, args)

        with tempfile.TemporaryDirectory() as workspace:
            def write(path, args):
                if args["path"] == "first.txt":
                    entered.set()
                    if not release.wait(5):
                        raise AssertionError("fixture did not release the owned write")
                try:
                    Path(path, args["path"]).write_text(args["content"], encoding="utf-8")
                    return "completed write"
                finally:
                    settled.set()

            async def run():
                await _Backend()._run_agent(
                    SimpleNamespace(emit=events.append), workspace, "finish both files",
                    "offline-model", "full", False, 50,
                )

            with (
                mock.patch.object(oa, "current_run", return_value=record),
                mock.patch.object(oa, "_stream_completion", new_callable=mock.AsyncMock,
                                  return_value=("", "", [_call("first.txt"), _call("second.txt")])),
                mock.patch.object(tool, "run", new=observe_run),
                mock.patch.object(tool, "_run_sync", new=write),
            ):
                owner = asyncio.create_task(run())
                try:
                    self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                    owner.cancel("first cancellation")
                    # Wait until the dispatcher has cancelled its child and
                    # is waiting for the blocked real write to settle.
                    async with asyncio.timeout(3):
                        while not tool_tasks[0].cancelling():
                            await asyncio.sleep(0)
                    owner.cancel("second cancellation")
                    for _ in range(10):
                        await asyncio.sleep(0)
                    self.assertFalse(owner.done(), "HTTP owner abandoned its actual write")
                    self.assertFalse(settled.is_set())
                    self.assertFalse(Path(workspace, "second.txt").exists())
                    release.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(owner, 3)
                    self.assertTrue(settled.is_set())
                    self.assertTrue(tool_tasks[0].done())
                    self.assertEqual(Path(workspace, "first.txt").read_text(encoding="utf-8"), "first.txt")
                    self.assertFalse(Path(workspace, "second.txt").exists())
                    self.assertFalse(any(event.get("output") == "completed write" for event in events))
                finally:
                    release.set()
                    await asyncio.gather(owner, *tool_tasks, return_exceptions=True)

            # Cancellation can race a successful physical write before its
            # result becomes durable. Resume must retain the saved intent as
            # unknown, inspect the workspace, and never automatically replay.
            self.assertEqual(record.values["http_pending_results"], {})
            restored = oa._restore_http_messages(record.values["http_messages"], {})
            tool_messages = [message for message in restored if message["role"] == "tool"]
            self.assertEqual([message["tool_call_id"] for message in tool_messages],
                             ["first.txt", "second.txt"])
            self.assertTrue(all("outcome is unknown" in message["content"] for message in tool_messages))
            requests = []

            async def resume(*args, **kwargs):
                requests.append(copy.deepcopy(args[1]))
                return "existing workspace state inspected", "", []

            with (
                mock.patch.object(oa, "current_run", return_value=record),
                mock.patch.object(oa, "_stream_completion", new=resume),
                mock.patch.object(agent_tools, "execute_tool", new_callable=mock.AsyncMock) as execute,
            ):
                await run()
            execute.assert_not_awaited()
            recovered = [message for message in requests[0]["messages"] if message["role"] == "tool"]
            self.assertEqual(recovered, tool_messages)
            self.assertFalse(Path(workspace, "second.txt").exists())


class ToolDispatchCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_outer_cancel_does_not_interrupt_owned_plugin_cleanup(self):
        started = asyncio.Event()
        cancelling = asyncio.Event()
        release = asyncio.Event()
        settled = asyncio.Event()
        children = []
        events = []

        async def run(*args):
            children.append(asyncio.current_task())
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelling.set()
                await release.wait()
                settled.set()
                return "late success after swallowing cancellation"

        plugin = SimpleNamespace(run=run, file_action=None, is_plugin=True)
        with mock.patch.dict(agent_tools._BY_NAME, {"late_plugin": plugin}):
            owner = asyncio.create_task(agent_tools.execute_tool(
                "late_plugin", {}, "/unused", "full", events.append,
            ))
            try:
                await asyncio.wait_for(started.wait(), 3)
                self.assertEqual(children[0].cancelling(), 0)
                owner.cancel("outer cancellation")
                await asyncio.wait_for(cancelling.wait(), 3)
                self.assertFalse(owner.done())
                owner.cancel("repeated outer cancellation")
                for _ in range(5):
                    await asyncio.sleep(0)
                self.assertFalse(owner.done())
                self.assertEqual(children[0].cancelling(), 1)
                release.set()
                with self.assertRaises(asyncio.CancelledError) as caught:
                    await owner
                self.assertEqual(caught.exception.args, ("outer cancellation",))
                self.assertFalse(any(event["type"] == "tool_result" for event in events))
                self.assertTrue(settled.is_set())
            finally:
                release.set()
                await asyncio.gather(owner, *children, return_exceptions=True)
