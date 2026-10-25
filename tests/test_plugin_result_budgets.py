"""Declared tool result budgets survive dispatch without trusting call args."""

from contextlib import asynccontextmanager, redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent_tools
from Cozter.agent_tools.builtin import web_fetch
from Cozter.agent_tools.plugins import http_request
from Cozter.agent_tools.plugins.git_info import GitInfoTool
from Cozter.agent_tools.plugins.git_ops import GitOpsTool
from Cozter.agent_tools.plugins.git_sync import GitSyncTool
from Cozter.tests.test_agent_plugins import FakeResponse, FakeSession


class PluginCliHelpTests(unittest.TestCase):
    def test_help_describes_arguments_without_running_the_plugin(self):
        for option in ("--help", "-h"):
            output = io.StringIO()
            with (
                self.subTest(option=option),
                mock.patch("sys.argv", ["git_info", option]),
                mock.patch.object(GitInfoTool, "run", new_callable=mock.AsyncMock) as run,
                redirect_stdout(output),
            ):
                GitInfoTool.run_as_script()
            help_data = json.loads(output.getvalue())
            self.assertEqual(help_data["name"], "git_info")
            self.assertIn("blame", help_data["parameters"]["properties"]["action"]["enum"])
            self.assertIn("offset", help_data["parameters"]["properties"])
            self.assertTrue(help_data["parallel_safe"])
            run.assert_not_awaited()

    def test_cli_guidance_marks_only_explicit_independent_reads(self):
        safe = SimpleNamespace(
            name="safe", is_plugin=True, parallel_safe=True, description="safe read", parameters={},
        )
        ordered = SimpleNamespace(
            name="ordered", is_plugin=True, description="mixed actions", parameters={},
        )
        with mock.patch.object(agent_tools, "_TOOLS", [safe, ordered]):
            prelude = agent_tools.cli_plugin_prelude()
        self.assertIn("--help", prelude)
        self.assertIn("safe [independent read-only]", prelude)
        self.assertIn("- ordered:", prelude)
        self.assertNotIn("ordered [independent read-only]", prelude)


class PluginResultBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def dispatch(self, name, args=None):
        return await agent_tools.execute_tool(name, args or {}, ".", "full", lambda _event: None)

    async def test_http_request_preserves_the_requested_body_and_metadata(self):
        body = "request-body-" * 900
        session = FakeSession([FakeResponse(
            status=200, url="https://api.example.com/data",
            headers={"Content-Type": "text/plain"}, body=body.encode(),
        )])

        @asynccontextmanager
        async def open_session():
            yield session

        with mock.patch.object(http_request, "open_public_http_session", open_session):
            result = await self.dispatch("http_request", {
                "url": "https://api.example.com/data", "max_chars": 12_000,
            })
        self.assertIn("HTTP 200", result)
        self.assertIn(body, result)
        self.assertNotIn("truncated", result)
        self.assertGreater(len(result), 4_000)
        self.assertLessEqual(len(result), 16 * 1024)

    async def test_web_fetch_preserves_its_advertised_preview(self):
        body = "fetch-body-" * 2_500

        @asynccontextmanager
        async def open_session():
            yield object()

        with (
            mock.patch.object(web_fetch, "_open_public_http_session", open_session),
            mock.patch.object(web_fetch, "_fetch_following_redirects", new_callable=mock.AsyncMock,
                              return_value=("https://example.com/data", "text/plain", body, False)),
        ):
            result = await self.dispatch("web_fetch", {
                "url": "https://example.com/data", "max_chars": 30_000,
            })
        self.assertIn(body, result)
        self.assertNotIn("truncated", result)
        self.assertLessEqual(len(result), 32 * 1024)

    async def test_git_plugins_keep_their_own_bounded_output(self):
        evidence = "git evidence\n" * 800
        for tool in (GitInfoTool(), GitOpsTool(), GitSyncTool()):
            with (
                self.subTest(tool=tool.name),
                mock.patch.dict(agent_tools.TOOLS_BY_NAME, {tool.name: tool}),
                mock.patch.object(tool, "run", new_callable=mock.AsyncMock, return_value=evidence),
            ):
                self.assertEqual(await self.dispatch(tool.name), evidence)

    async def test_undeclared_replacement_plugin_retains_default_budget(self):
        replacement = SimpleNamespace(
            name="web_fetch", is_plugin=True, file_action=None,
            run=mock.AsyncMock(return_value="x" * 10_000),
        )
        with mock.patch.dict(agent_tools.TOOLS_BY_NAME, {replacement.name: replacement}):
            result = await self.dispatch(replacement.name, {"max_result_chars": 60_000})
        self.assertEqual(len(result), 4_000)
        self.assertIn("truncated", result)

    async def test_invalid_metadata_does_not_expand_the_default_budget(self):
        tool = SimpleNamespace(
            name="budget_test", is_plugin=True, file_action=None,
            run=mock.AsyncMock(return_value="x" * 10_000),
        )
        for invalid in (None, True, False, 0, -1, "12000", 12_000.0):
            tool.max_result_chars = invalid
            with self.subTest(metadata=invalid), mock.patch.dict(agent_tools.TOOLS_BY_NAME, {tool.name: tool}):
                self.assertEqual(len(await self.dispatch(tool.name)), 4_000)

    async def test_declared_budget_cannot_exceed_the_global_bound(self):
        tool = SimpleNamespace(
            name="budget_test", is_plugin=True, file_action=None, max_result_chars=10**12,
            run=mock.AsyncMock(return_value="x" * 100_000),
        )
        with mock.patch.dict(agent_tools.TOOLS_BY_NAME, {tool.name: tool}):
            result = await self.dispatch(tool.name)
        self.assertEqual(len(result), 64 * 1024)
        self.assertIn("truncated", result)


if __name__ == "__main__":
    unittest.main()
