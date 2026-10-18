"""Skill tools obey dispatcher permissions and preserve progressive read results."""

import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent_tools, skills
from Cozter.backends_agent import _openai_agent as oa


class SkillToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.object(skills, "_native_roots", return_value=[]))
        self.skill_dir = Path(self.ws, ".cozter", "skills")
        self.skill_dir.mkdir(parents=True)
        self.skill_file = self.skill_dir / "build.md"
        self.skill_file.write_text(
            "---\nname: build\ndescription: Run the fixture checks.\n---\n"
            "# Build checks\nUse fixture-check-marker for the requested work.\n",
            encoding="utf-8",
        )

    async def _execute(self, name: str, args: dict, approval: str = "auto") -> str:
        events = []
        result = await agent_tools.execute_tool(name, args, self.ws, approval, events.append)
        self.assertEqual(events[-1], {"type": "tool_result", "name": name, "output": result})
        return result

    async def test_skill_tools_are_read_only_and_available_in_normal_read_modes(self) -> None:
        for mode in ("auto", "confirm", "full"):
            with self.subTest(mode=mode):
                schema = oa._tools_for_approval(mode, False)
                names = {entry["function"]["name"] for entry in schema}
                self.assertTrue(agent_tools.SKILL_TOOL_NAMES <= names)
                listing = await self._execute("list_skills", {}, mode)
                self.assertIn("build", listing)
                result = json.loads(await self._execute("read_skill", {"name": "build"}, mode))
                self.assertIn("fixture-check-marker", result["content"])
        for name in agent_tools.SKILL_TOOL_NAMES:
            tool = agent_tools.TOOLS_BY_NAME[name]
            self.assertFalse(tool.is_plugin)
            self.assertTrue(tool.parallel_safe)
            self.assertFalse(tool.requires_full_permission)
            self.assertEqual(tool.max_result_chars, 64 * 1024)
            self.assertTrue((await self._execute(name, {"name": "build"}, "deny")).startswith("Blocked:"))

    async def test_strict_paging_and_revision_arguments_fail_before_service_calls(self) -> None:
        cases = [
            ("list_skills", {"offset": value}) for value in (True, 1.0, "1", -1, None)
        ] + [
            ("list_skills", {"max_results": value}) for value in (False, 0, 201, 1.5)
        ] + [
            ("read_skill", {"name": "build", "offset": value}) for value in (True, 1.0, "1", -1)
        ] + [
            ("read_skill", {"name": "build", "max_chars": value}) for value in (False, 0, 48_001)
        ] + [
            ("read_skill", {"name": value}) for value in (None, True, "", "  ")
        ] + [
            ("read_skill", {"name": "build", "revision": value}) for value in (False, 4, "")
        ] + [
            ("read_skill", {"name": "build", "resource": value}) for value in (False, 4, " ")
        ]
        with (
            mock.patch.object(skills, "list_skills") as listing,
            mock.patch.object(skills, "read_skill") as reading,
        ):
            for name, arguments in cases:
                with self.subTest(name=name, arguments=arguments):
                    self.assertTrue((await self._execute(name, arguments)).startswith("Error:"))
            listing.assert_not_called()
            reading.assert_not_called()

    async def test_large_service_page_retains_cursor_through_actual_dispatcher(self) -> None:
        page = json.dumps({"content": "instruction " * 4_000, "next_offset": 48_000, "revision": "revision"})
        with mock.patch.object(skills, "read_skill", return_value=page) as reading:
            result = await self._execute("read_skill", {
                "name": "build", "offset": 12, "max_chars": 48_000, "revision": "revision",
            }, "confirm")
        self.assertEqual(result, page)
        self.assertEqual(json.loads(result)["next_offset"], 48_000)
        reading.assert_called_once_with(
            self.ws, "build", offset=12, max_chars=48_000, revision="revision", resource=None,
        )

    async def test_supporting_reference_is_readable_but_traversal_is_not(self) -> None:
        reference = self.skill_dir / "references" / "usage.txt"
        reference.parent.mkdir()
        reference.write_text("supporting-reference-marker", encoding="utf-8")
        private = Path(self.ws, "private.txt")
        private.write_text("private-outside-skill-marker", encoding="utf-8")
        for mode in ("auto", "confirm", "full"):
            with self.subTest(mode=mode):
                result = json.loads(await self._execute("read_skill", {
                    "name": "build", "resource": "references/usage.txt",
                }, mode))
                self.assertEqual(result["content"], "supporting-reference-marker")
                escaped = await self._execute("read_skill", {
                    "name": "build", "resource": "../../private.txt",
                }, mode)
                self.assertNotIn("private-outside-skill-marker", escaped)
                self.assertNotIn("private-outside-skill-marker", await self._execute(
                    "read_skill", {"name": str(private)}, mode,
                ))

    async def test_skill_only_scope_blocks_every_other_tool_even_with_full_permission(self) -> None:
        with agent_tools.skill_planning_scope():
            names = {entry["function"]["name"] for entry in oa._tools_for_approval("deny", True)}
            self.assertEqual(names, set(agent_tools.SKILL_TOOL_NAMES))
            result = json.loads(await self._execute("read_skill", {"name": "build"}, "deny"))
            self.assertIn("fixture-check-marker", result["content"])
            for name, args in (
                ("read_file", {"path": str(self.skill_file)}),
                ("write_file", {"path": "forbidden.txt", "content": "forbidden"}),
                ("bash", {"command": "must not execute"}),
                ("web_fetch", {"url": "https://unused.invalid"}),
            ):
                with self.subTest(name=name):
                    self.assertIn("skill-only planning", await self._execute(name, args, "full"))
        self.assertFalse(agent_tools.skill_planning_active())
        self.assertIsNone(oa._tools_for_approval("deny", True))
        self.assertFalse(Path(self.ws, "forbidden.txt").exists())

    async def test_scope_is_task_local_and_plugin_shadow_cannot_gain_read_privileges(self) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def planner() -> str:
            with agent_tools.skill_planning_scope():
                entered.set()
                await release.wait()
                return await self._execute("read_file", {"path": str(self.skill_file)}, "full")

        async def ordinary() -> str:
            await entered.wait()
            try:
                self.assertFalse(agent_tools.skill_planning_active())
                return await self._execute("read_file", {"path": str(self.skill_file)}, "full")
            finally:
                release.set()

        planned, normal = await asyncio.wait_for(asyncio.gather(planner(), ordinary()), timeout=2)
        self.assertIn("skill-only planning", planned)
        self.assertIn("fixture-check-marker", normal)
        fake = SimpleNamespace(name="read_skill", is_plugin=True, file_action=None, run=mock.AsyncMock())
        with mock.patch.dict(agent_tools.TOOLS_BY_NAME, {"read_skill": fake}):
            with agent_tools.skill_planning_scope():
                self.assertTrue((await self._execute("read_skill", {"name": "build"}, "full")).startswith("Blocked:"))
            self.assertTrue((await self._execute("read_skill", {"name": "build"}, "confirm")).startswith("Blocked:"))
        fake.run.assert_not_awaited()
