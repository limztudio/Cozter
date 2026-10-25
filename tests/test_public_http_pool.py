"""Offline local-server checks for worker-owned public HTTP connections.

Fixture resolvers deliberately route fixture.test to loopback. Production
PublicResolver is separately exercised without that bypass for refusals.
"""

import asyncio
import copy
import socket
from types import SimpleNamespace
import unittest
from unittest import mock

import aiohttp
from aiohttp import web

from Cozter.agent_tools import base, _http_types, _public_resolver
from Cozter.agent_tools.plugins.http_request import HttpRequestTool
from Cozter.backends_agent import _openai_agent as oa
from Cozter.tests.helpers import OfflineHttpBackend


class PublicHttpPoolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.connections = set()
        self.resolvers = []
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.four_active = asyncio.Event()
        self.active = self.peak = 0

        async def handler(request):
            self.connections.add(request.transport)
            self.requests.append({
                "path": request.path,
                "authorization": request.headers.get("Authorization"),
                "cookie": request.headers.get("Cookie"),
            })
            if request.path == "/redirect":
                response = web.HTTPFound("/cookie-check")
                response.set_cookie("within_call", "secret")
                raise response
            if request.path in {"/wait", "/hold"}:
                self.active += 1
                self.peak = max(self.peak, self.active)
                self.started.set()
                if self.active == 4:
                    self.four_active.set()
                try:
                    await self.release.wait()
                finally:
                    self.active -= 1
            response = web.Response(text="offline fixture", content_type="text/plain")
            if request.path == "/close":
                response.force_close()
            return response

        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        self.url = f"http://fixture.test:{self.port}"

    async def asyncTearDown(self):
        self.release.set()
        await self.runner.cleanup()

    def _resolver(self):
        owner = self

        class FixtureResolver:
            calls = 0
            closes = 0

            async def resolve(self, host, port=0, family=socket.AF_UNSPEC):
                self.calls += 1
                return [{
                    "hostname": host, "host": "127.0.0.1", "port": port,
                    "family": socket.AF_INET, "proto": 0, "flags": 0,
                }]

            async def close(self):
                self.closes += 1

        result = FixtureResolver()
        owner.resolvers.append(result)
        return result

    async def _get(self, path="/", *, headers=None):
        async with base.open_public_http_session() as session:
            async with session.get(self.url + path, headers=headers) as response:
                return await response.text()

    async def test_unused_scope_does_not_allocate_http_resources(self):
        with (
            mock.patch.object(_http_types, "PublicResolver") as resolver,
            mock.patch.object(_http_types.aiohttp, "TCPConnector") as connector,
        ):
            async with base.public_http_pool_scope():
                await asyncio.sleep(0)
            resolver.assert_not_called()
            connector.assert_not_called()

    async def test_connections_reuse_without_sharing_headers_or_cookies(self):
        with mock.patch.object(_http_types, "PublicResolver", side_effect=self._resolver):
            async with base.public_http_pool_scope():
                pool = base._PUBLIC_HTTP_POOL.get()
                result = await HttpRequestTool().run("/unused", {
                    "url": self.url + "/redirect",
                    "headers": {"Authorization": "Bearer call-one"},
                })
                self.assertIn("HTTP 200", result)
                result = await HttpRequestTool().run("/unused", {"url": self.url + "/"})
                self.assertIn("HTTP 200", result)
                connector = pool.connector()
                self.assertEqual(connector.limit, 8)
                self.assertEqual(connector.limit_per_host, 4)
                self.assertFalse(connector.use_dns_cache)
                self.assertFalse(connector.closed)
        self.assertEqual(len(self.connections), 1)
        self.assertEqual(len(self.resolvers), 1)
        self.assertEqual(self.resolvers[0].calls, 1)
        self.assertEqual(self.resolvers[0].closes, 1)
        self.assertTrue(connector.closed)
        self.assertEqual([request["authorization"] for request in self.requests],
                         ["Bearer call-one", "Bearer call-one", None])
        self.assertIsNone(self.requests[0]["cookie"])
        self.assertEqual(self.requests[1]["cookie"], "within_call=secret")
        self.assertIsNone(self.requests[2]["cookie"])

    async def test_borrowed_sessions_ignore_proxy_environment(self):
        with (
            mock.patch.object(_http_types, "PublicResolver", side_effect=self._resolver),
            mock.patch.dict("os.environ", {"HTTP_PROXY": "http://127.0.0.1:1"}),
        ):
            async with base.public_http_pool_scope():
                async with base.open_public_http_session() as session:
                    self.assertFalse(session.trust_env)
                    self.assertIsNone(session.timeout.total)
                    async with session.get(self.url) as response:
                        self.assertEqual(await response.text(), "offline fixture")

    async def test_each_new_connection_resolves_again(self):
        with mock.patch.object(_http_types, "PublicResolver", side_effect=self._resolver):
            async with base.public_http_pool_scope():
                await self._get("/close")
                await self._get("/close")
        self.assertEqual(len(self.connections), 2)
        self.assertEqual(self.resolvers[0].calls, 2)

    async def test_real_public_resolver_refuses_private_answer_before_connecting(self):
        resolver = SimpleNamespace(
            resolve=mock.AsyncMock(return_value=[{
                "hostname": "fixture.test", "host": "127.0.0.1", "port": self.port,
                "family": socket.AF_INET, "proto": 0, "flags": 0,
            }]),
            close=mock.AsyncMock(),
        )
        with mock.patch.object(_public_resolver, "DefaultResolver", return_value=resolver):
            async with base.public_http_pool_scope():
                with self.assertRaises(aiohttp.ClientConnectorError):
                    await self._get()
        self.assertEqual(self.requests, [])
        resolver.resolve.assert_awaited_once()
        resolver.close.assert_awaited_once()

    async def test_same_host_connection_concurrency_is_bounded(self):
        with mock.patch.object(_http_types, "PublicResolver", side_effect=self._resolver):
            async with base.public_http_pool_scope():
                tasks = [asyncio.create_task(self._get("/hold")) for _ in range(8)]
                try:
                    await asyncio.wait_for(self.four_active.wait(), 3)
                    self.assertEqual(self.active, 4)
                    self.release.set()
                    await asyncio.gather(*tasks)
                finally:
                    self.release.set()
                    await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(self.peak, 4)
        self.assertEqual(len(self.connections), 4)

    async def test_nested_scopes_restore_outer_owner(self):
        with mock.patch.object(_http_types, "PublicResolver", side_effect=self._resolver):
            async with base.public_http_pool_scope():
                outer = base._PUBLIC_HTTP_POOL.get()
                await self._get()
                async with base.public_http_pool_scope():
                    inner = base._PUBLIC_HTTP_POOL.get()
                    await self._get()
                self.assertIsNot(outer, inner)
                self.assertTrue(inner._connector.closed)
                self.assertIs(base._PUBLIC_HTTP_POOL.get(), outer)
                await self._get()
                self.assertFalse(outer.connector().closed)
        self.assertEqual(len(self.resolvers), 2)
        self.assertEqual(len(self.connections), 2)

    async def test_unscoped_calls_keep_independent_original_lifetimes(self):
        with mock.patch.object(_http_types, "PublicResolver", side_effect=self._resolver):
            await self._get()
            await self._get()
        self.assertEqual(len(self.resolvers), 2)
        self.assertEqual(len(self.connections), 2)
        self.assertEqual([resolver.closes for resolver in self.resolvers], [1, 1])

    async def test_cancelling_a_borrow_does_not_close_its_workers_pool(self):
        with mock.patch.object(_http_types, "PublicResolver", side_effect=self._resolver):
            async with base.public_http_pool_scope():
                pool = base._PUBLIC_HTTP_POOL.get()
                task = asyncio.create_task(self._get("/wait"))
                try:
                    await asyncio.wait_for(self.started.wait(), 3)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    self.assertFalse(pool.connector().closed)
                    self.assertEqual(await self._get(), "offline fixture")
                finally:
                    self.release.set()
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(pool._connector.closed)

    async def test_repeated_owner_cancellation_settles_exact_owned_cleanup(self):
        closing = asyncio.Event()
        release_close = asyncio.Event()
        body_entered = asyncio.Event()
        fixture = self._resolver()

        async def close():
            fixture.closes += 1
            closing.set()
            await release_close.wait()

        fixture.close = close
        pools = []

        async def owner():
            async with base.public_http_pool_scope():
                pools.append(base._PUBLIC_HTTP_POOL.get())
                await self._get()
                body_entered.set()
                await asyncio.Event().wait()

        with mock.patch.object(_http_types, "PublicResolver", return_value=fixture):
            task = asyncio.create_task(owner())
            try:
                await asyncio.wait_for(body_entered.wait(), 3)
                task.cancel()
                await asyncio.wait_for(closing.wait(), 3)
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                release_close.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 3)
            finally:
                release_close.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(fixture.closes, 1)
        self.assertTrue(pools[0]._connector.closed)

    async def test_success_and_error_both_close_resources(self):
        with mock.patch.object(_http_types, "PublicResolver", side_effect=self._resolver):
            for failure in (False, True):
                with self.subTest(failure=failure):
                    try:
                        async with base.public_http_pool_scope():
                            pool = base._PUBLIC_HTTP_POOL.get()
                            await self._get()
                            if failure:
                                raise RuntimeError("offline failure")
                    except RuntimeError:
                        self.assertTrue(failure)
                    self.assertTrue(pool._connector.closed)
                    with self.assertRaisesRegex(RuntimeError, "worker has finished"):
                        pool.connector()
        self.assertEqual([resolver.closes for resolver in self.resolvers], [1, 1])

    async def test_concurrent_http_workers_have_separate_owned_pools(self):
        requests = {}
        pools = {}

        async def stream(*args, **kwargs):
            payload = copy.deepcopy(args[1])
            model = payload["model"]
            requests[model] = requests.get(model, 0) + 1
            if requests[model] > 2:
                return "done", "", []
            return "", "", [{
                "id": f"{model}-{requests[model]}", "type": "function",
                "function": {"name": "web_fetch", "arguments": "{}"},
            }]

        async def execute(*args):
            pool = base._PUBLIC_HTTP_POOL.get()
            pools[id(pool)] = pool
            await self._get()
            return "offline evidence"

        async def worker(model):
            await OfflineHttpBackend()._run_agent(
                SimpleNamespace(emit=lambda event: None), "/workspace", "request",
                model, "full", False, 0,
            )

        with (
            mock.patch.object(_http_types, "PublicResolver", side_effect=self._resolver),
            mock.patch.object(oa, "current_run", return_value=None),
            mock.patch.object(oa, "_stream_completion", side_effect=stream),
            mock.patch.object(oa.tools, "execute_tool", side_effect=execute),
        ):
            await asyncio.gather(worker("worker-a"), worker("worker-b"))
        self.assertEqual(len(pools), 2)
        self.assertEqual(len(self.resolvers), 2)
        self.assertEqual(len(self.connections), 2)
        self.assertEqual([resolver.calls for resolver in self.resolvers], [1, 1])
        self.assertTrue(all(pool._connector.closed for pool in pools.values()))
