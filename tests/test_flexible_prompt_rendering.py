"""Ordinary workers render once while HTTP lease deltas keep a full bootstrap."""

import unittest
from unittest import mock

from Cozter import agent
from Cozter.backends_agent import _openai_agent as oa
from Cozter.backends_agent.base import AgentResult
from Cozter.tests import test_flexible_tier_leases as lease_tests


class FlexiblePromptRenderingTests(unittest.IsolatedAsyncioTestCase):
    internal = lease_tests.FlexibleTierLeaseTests.internal
    invoke = lease_tests.FlexibleTierLeaseTests.invoke

    async def asyncSetUp(self):
        await lease_tests.FlexibleTierLeaseTests.asyncSetUp(self)
        self.renders = []
        build = agent._build_backend_prompt

        def render(backend, context, **kwargs):
            prompt = build(backend, context, **kwargs)
            self.renders.append((backend.name, context, prompt))
            return prompt

        self.enterContext(mock.patch.object(agent, "_build_backend_prompt", side_effect=render))

    async def test_ordinary_worker_reuses_full_render_without_bootstrap_scope(self):
        self.plan = "[PLAN]\n1. [mid] Repair the bounded cache dependency and verify its regression\n[/PLAN]"

        async def drive(backend, _ws, prompt, *_args, **_kwargs):
            self.assertEqual(backend.name, "zai")
            self.assertIsNone(oa._HTTP_CONTINUATION_BOOTSTRAP.get())
            self.assertEqual(len(self.renders), 1)
            self.assertEqual(prompt, self.renders[0][2])
            self.assertIn("Original request", prompt)
            self.assertIn("bounded cache dependency", prompt)
            return AgentResult(text="Bounded cache repair and regression verified."), False

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual(len(self.renders), 1)

    async def test_http_helper_resume_renders_distinct_delta_and_complete_bootstrap(self):
        calls = []

        async def drive(backend, _ws, prompt, *_args, **_kwargs):
            bootstrap = oa._HTTP_CONTINUATION_BOOTSTRAP.get()
            calls.append((backend.name, prompt, bootstrap))
            if len(calls) == 1:
                self.assertIsNone(bootstrap)
                return AgentResult(text=lease_tests.lease("low")), False
            if backend.name == "llama":
                self.assertIsNone(bootstrap)
                return AgentResult(text="HELPER-FINDING-MARKER: cache.cpp:84 owns the stale key."), False
            self.assertEqual(prompt, self.renders[-1][2])
            self.assertEqual(bootstrap, self.renders[-2][2])
            self.assertNotEqual(prompt, bootstrap)
            self.assertIn("HELPER-FINDING-MARKER", prompt)
            self.assertIn("HELPER-FINDING-MARKER", bootstrap)
            self.assertNotIn("[Sub-task 1/1:", prompt)
            self.assertIn("[Sub-task 1/1:", bootstrap)
            self.assertIn("Original request", bootstrap)
            self.assertIn("original cache dependency", prompt)
            return AgentResult(text="Original cache dependency repaired and verified."), False

        result, restarting = await self.invoke(drive)

        self.assertFalse(restarting)
        self.assertFalse(result.error)
        self.assertEqual([name for name, _, _ in calls], ["meta", "llama", "meta"])
        self.assertEqual([name for name, _, _ in self.renders], ["meta", "llama", "meta", "meta"])
        self.assertEqual(len({context for name, context, _ in self.renders if name == "meta"}), 3)


if __name__ == "__main__":
    unittest.main()
