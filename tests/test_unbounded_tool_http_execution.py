"""Tool and generation execution have no global deadline or retired overrides."""

import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent_tools, config
from Cozter.agent_tools import _http_types, base
from Cozter.backends_agent import _openai_agent as oa
from Cozter.backends_agent.llama import LlamaBackend
from Cozter.backends_agent.meta import MetaModelApiBackend
from Cozter.backends_agent.zai import ZaiBackend


class UnboundedToolHttpExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_dispatcher_waits_without_a_deadline_even_with_retired_config_values(self):
        started, release = asyncio.Event(), asyncio.Event()
        waits = []
        real_wait = asyncio.wait

        async def operation(*_args):
            started.set()
            await release.wait()
            return "completed after the caller released the operation"

        async def observe_wait(*args, **kwargs):
            waits.append(kwargs.get("timeout"))
            self.assertIsNone(kwargs.get("timeout"))
            return await real_wait(*args, **kwargs)

        tool = SimpleNamespace(run=operation, file_action=None, is_plugin=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "config.json")
            path.write_text(json.dumps({"tool_timeout": 0.001}), encoding="utf-8")
            with (
                mock.patch.object(config, "CONFIG_PATH", str(path)),
                mock.patch.dict(agent_tools.TOOLS_BY_NAME, {"waiting_plugin": tool}),
                mock.patch.object(asyncio, "wait", side_effect=observe_wait),
            ):
                owner = asyncio.create_task(agent_tools.execute_tool(
                    "waiting_plugin", {}, directory, "full", lambda event: None,
                ))
                try:
                    await asyncio.wait_for(started.wait(), 2)
                    await asyncio.sleep(0.02)
                    self.assertFalse(owner.done())
                    release.set()
                    self.assertEqual(await owner, "completed after the caller released the operation")
                finally:
                    release.set()
                    await asyncio.gather(owner, return_exceptions=True)
        self.assertTrue(waits)
        self.assertFalse(hasattr(agent_tools, "_tool_timeout_seconds"))

    async def test_generation_continues_past_simulated_two_hours_and_cancels_normally(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                started, release, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()
                elapsed = 0.0

                async def stream(*_args, **kwargs):
                    self.assertNotIn("deadline", kwargs)
                    started.set()
                    try:
                        await release.wait()
                        return "completed generation", "", []
                    finally:
                        settled.set()

                with (
                    mock.patch.object(oa, "_stream_once", side_effect=stream),
                    mock.patch.object(oa, "time", SimpleNamespace(monotonic=lambda: elapsed)),
                    mock.patch.object(asyncio, "wait_for", side_effect=AssertionError("no generation deadline")),
                ):
                    owner = asyncio.create_task(oa._stream_completion(
                        "http://offline.invalid", {}, {}, 0, "offline",
                    ))
                    await started.wait()
                    elapsed = 7201.0
                    await asyncio.sleep(0)
                    self.assertFalse(owner.done())
                    if cancel:
                        owner.cancel("operator stop")
                        with self.assertRaises(asyncio.CancelledError):
                            await owner
                    else:
                        release.set()
                        self.assertEqual(await owner, ("completed generation", "", []))
                self.assertTrue(settled.is_set())

    async def test_all_http_workers_ignore_retired_socket_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "config.json")
            path.write_text(json.dumps({
                "llama_socket_timeout": 0.001,
                "meta_socket_timeout": 0.001,
                "zai_socket_timeout": 0.001,
            }), encoding="utf-8")
            with (
                mock.patch.object(config, "CONFIG_PATH", str(path)),
                mock.patch.object(oa, "current_run", return_value=None),
                mock.patch.object(oa, "_stream_completion", new_callable=mock.AsyncMock,
                                  return_value=("verified", "", [])),
            ):
                for backend in (LlamaBackend(), MetaModelApiBackend(), ZaiBackend()):
                    with self.subTest(backend=backend.name):
                        await backend._run_agent(SimpleNamespace(emit=lambda event: None), directory,
                                                 "synthetic request", "model-a", "full", False, 50)
                        self.assertFalse(hasattr(backend, "_socket_timeout"))
                        self.assertFalse(hasattr(backend, "_socket_timeout_setting"))
        for key in ("tool_timeout", "llama_socket_timeout", "meta_socket_timeout", "zai_socket_timeout"):
            self.assertNotIn(key, config._DEFAULT_CONFIG)
            self.assertFalse(hasattr(config, "get_" + key))

    async def test_web_request_defaults_unbounded_and_preserves_explicit_operation_timeout(self):
        observed = []

        @asynccontextmanager
        async def response():
            yield SimpleNamespace(status=200)

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                pass

            def get(self, _url, **kwargs):
                observed.append(kwargs)
                return response()

        with mock.patch.object(_http_types.aiohttp, "ClientSession", return_value=Session()):
            async with base.open_http_response("https://offline.invalid"):
                pass
            async with base.open_http_response("https://offline.invalid", timeout=7201, allow_redirects=False):
                pass
        self.assertIsNone(observed[0]["timeout"].total)
        self.assertIsNone(observed[0]["timeout"].sock_read)
        self.assertEqual(observed[1]["timeout"].total, 7201)
        self.assertFalse(observed[1]["allow_redirects"])
