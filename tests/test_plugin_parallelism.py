"""Offline coverage of explicit read-only plugin concurrency contracts."""

import asyncio
import copy
import json
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent_tools
from Cozter.agent_tools.base import AgentTool
from Cozter.backends_agent import _openai_agent as oa


class _Record:
    def __init__(self):
        self.values = {}
        self.snapshots = []

    def get(self, key, default=None):
        return copy.deepcopy(self.values.get(key, default))

    async def save(self, **values):
        self.values.update(copy.deepcopy(values))
        self.snapshots.append(copy.deepcopy(self.values))


class _Backend(oa.OpenAIChatBackend):
    name = "offline-plugin-parallelism"

    def _chat_endpoint(self):
        return "http://offline.invalid/chat/completions"


def _plugin(name="read_plugin", *, parallel_safe=False):
    return SimpleNamespace(
        name=name,
        is_plugin=True,
        parallel_safe=parallel_safe,
        requires_full_permission=False,
        max_result_chars=4000,
        file_action=None,
        run=mock.AsyncMock(return_value="plugin evidence"),
        schema={"name": name, "description": "offline fixture", "parameters": {}},
    )


def _call(identity, name="read_plugin"):
    return {
        "id": identity, "type": "function",
        "function": {"name": name, "arguments": json.dumps({"path": identity})},
    }


class PluginParallelismTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, *, approval="full"):
        events = []
        process = SimpleNamespace(emit=events.append)
        await _Backend()._run_agent(
            process, "/workspace", "inspect the requested scope", "offline-model",
            approval, False, 50,
        )
        return events

    def test_tool_metadata_defaults_preserve_serial_and_result_budget(self):
        self.assertIs(AgentTool.parallel_safe, False)
        self.assertEqual(AgentTool.max_result_chars, 4000)

    def test_only_literal_true_opt_in_groups_a_plugin(self):
        for declaration in (False, None, 0, 1, "true", [], {}):
            with self.subTest(declaration=declaration):
                plugin = _plugin(parallel_safe=declaration)
                with mock.patch.dict(agent_tools._BY_NAME, {plugin.name: plugin}):
                    self.assertFalse(oa._parallel_read_tool(plugin.name))
        plugin = _plugin(parallel_safe=True)
        with mock.patch.dict(agent_tools._BY_NAME, {plugin.name: plugin}):
            self.assertTrue(oa._parallel_read_tool(plugin.name))

    def test_shadowing_a_read_builtin_requires_its_own_contract(self):
        self.assertTrue(oa._parallel_read_tool("read_file"))
        replacement = _plugin("read_file")
        with mock.patch.dict(agent_tools._BY_NAME, {"read_file": replacement}):
            self.assertFalse(oa._parallel_read_tool("read_file"))
        self.assertFalse(oa._parallel_read_tool("unregistered_plugin"))

    async def test_plugin_waves_are_bounded_ordered_and_durable_before_barriers(self):
        record = _Record()
        plugin = _plugin(parallel_safe=True)
        barrier = _plugin("mixed_action_plugin")
        calls = [_call(f"before-{index}") for index in range(7)]
        calls += [_call("barrier", barrier.name), _call("write", "write_file")]
        calls += [_call(f"after-{index}") for index in range(3)]
        requests = []
        active = peak = 0
        finished = []
        mutations = []

        async def stream(*args, **kwargs):
            requests.append(copy.deepcopy(args[1]))
            return ("", "", calls) if len(requests) == 1 else ("done", "", [])

        async def execute(name, args, *unused):
            nonlocal active, peak
            identity = args["path"]
            if name in {barrier.name, "write_file"}:
                self.assertEqual(active, 0)
                self.assertEqual(len(finished), 7)
                self.assertEqual(
                    [record.values["http_pending_results"][str(index)] for index in range(7)],
                    [f"result before-{index}" for index in range(7)],
                )
                if name == "write_file":
                    self.assertEqual(record.values["http_pending_results"]["7"],
                                     "mixed action finished")
                mutations.append(name)
                return "mixed action finished" if name == barrier.name else "write finished"
            self.assertEqual(bool(mutations), identity.startswith("after-"))
            if mutations:
                self.assertEqual(record.values["http_pending_results"]["8"], "write finished")
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0)
                finished.append(identity)
                return f"result {identity}"
            finally:
                active -= 1

        with (
            mock.patch.dict(agent_tools._BY_NAME, {plugin.name: plugin, barrier.name: barrier}),
            mock.patch.object(oa, "current_run", return_value=record),
            mock.patch.object(oa, "_stream_completion", side_effect=stream),
            mock.patch.object(agent_tools, "execute_tool", side_effect=execute),
        ):
            await self._run()
        self.assertEqual(peak, 4)
        self.assertEqual(mutations, [barrier.name, "write_file"])
        self.assertEqual(
            [message["tool_call_id"] for message in requests[1]["messages"]
             if message["role"] == "tool"],
            [call["id"] for call in calls],
        )

    async def test_undeclared_plugins_remain_serial_even_with_read_arguments(self):
        plugin = _plugin("mixed_action_plugin")
        active = peak = 0
        requests = 0

        async def stream(*args, **kwargs):
            nonlocal requests
            requests += 1
            return ("", "", [_call(str(index), plugin.name) for index in range(4)]) \
                if requests == 1 else ("done", "", [])

        async def execute(*args):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0)
                return "read result from a mixed-action tool"
            finally:
                active -= 1

        with (
            mock.patch.dict(agent_tools._BY_NAME, {plugin.name: plugin}),
            mock.patch.object(oa, "current_run", return_value=None),
            mock.patch.object(oa, "_stream_completion", side_effect=stream),
            mock.patch.object(agent_tools, "execute_tool", side_effect=execute),
        ):
            await self._run()
        self.assertEqual(peak, 1)

    async def test_cancelled_plugin_wave_settles_sibling_and_retains_finished_result(self):
        record = _Record()
        plugin = _plugin(parallel_safe=True)
        waiting = asyncio.Event()
        stopped = asyncio.Event()

        async def execute(name, args, *unused):
            if args["path"] == "finished":
                return "completed plugin evidence"
            waiting.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        with (
            mock.patch.dict(agent_tools._BY_NAME, {plugin.name: plugin}),
            mock.patch.object(oa, "current_run", return_value=record),
            mock.patch.object(oa, "_stream_completion", new_callable=mock.AsyncMock,
                              return_value=("", "", [_call("finished"), _call("waiting")])),
            mock.patch.object(agent_tools, "execute_tool", side_effect=execute),
        ):
            task = asyncio.create_task(self._run())
            await asyncio.wait_for(waiting.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(stopped.is_set())
        self.assertEqual(record.values["http_pending_results"], {"0": "completed plugin evidence"})
        restored = oa._restore_http_messages(record.values["http_messages"],
                                            record.values["http_pending_results"])
        self.assertEqual(restored[-2]["content"], "completed plugin evidence")
        self.assertIn("outcome is unknown", restored[-1]["content"])

    async def test_parallel_opt_in_does_not_change_confirm_permissions(self):
        plugin = _plugin(parallel_safe=True)
        requests = []

        async def stream(*args, **kwargs):
            requests.append(copy.deepcopy(args[1]))
            return ("", "", [_call("blocked")]) if len(requests) == 1 else ("done", "", [])

        with (
            mock.patch.dict(agent_tools._BY_NAME, {plugin.name: plugin}),
            mock.patch.object(oa, "current_run", return_value=None),
            mock.patch.object(oa, "_stream_completion", side_effect=stream),
        ):
            events = await self._run(approval="confirm")
        plugin.run.assert_not_awaited()
        self.assertFalse(agent_tools._is_confirm_read_only(plugin))
        self.assertNotIn(plugin.name, [entry["function"]["name"]
                                       for entry in requests[0]["tools"]])
        self.assertTrue(any(event.get("output", "").startswith("Blocked:") for event in events))
