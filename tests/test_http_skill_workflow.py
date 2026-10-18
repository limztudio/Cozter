"""Real skill tools through mocked OpenAI-compatible HTTP completions."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter import agent_tools, skills
from Cozter.backends_agent import _openai_agent as oa
from Cozter.tests.test_http_execution_continuation import _Record


class _SkillBackend(oa.OpenAIChatBackend):
    name = "offline-skill-workflow"
    supports_vision = False

    def _chat_endpoint(self) -> str:
        return "http://unused.invalid/chat/completions"

    def _max_agent_turns(self) -> int:
        return 12


class _ChatOnlySkillBackend(_SkillBackend):
    def _supports_tools_for_model(self, _model) -> bool:
        return False


def _call(identifier: str, tool_name: str, **args) -> dict:
    return {"id": identifier, "type": "function", "function": {
        "name": tool_name, "arguments": json.dumps(args),
    }}


class HttpSkillWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.ws = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.object(skills, "_native_roots", return_value=[]))
        self.skill_file = Path(self.ws, ".cozter", "skills", "build.md")
        self.skill_file.parent.mkdir(parents=True)
        self.skill_file.write_text(
            "---\nname: build\ndescription: Fixture build workflow.\n---\n"
            "# Build\nbody-instructions-marker " + "Inspect retained evidence. " * 25,
            encoding="utf-8",
        )

    async def _run(self, *, approval="auto", compaction=False, prompt="Complete the fixture checks", backend=None):
        events = []
        await (backend or _SkillBackend())._run_agent(
            SimpleNamespace(emit=events.append), self.ws, prompt, "fixture-model", approval, compaction, 50,
        )
        return events

    async def test_mock_completion_lists_and_pages_skill_using_actual_read_only_tools(self) -> None:
        for approval in ("auto", "confirm", "full"):
            with self.subTest(approval=approval):
                requests = []
                pages = []

                async def stream(*args, **_kwargs):
                    payload = copy.deepcopy(args[1])
                    requests.append(payload)
                    if len(requests) == 1:
                        self.assertIn("build", payload["messages"][0]["content"])
                        self.assertNotIn("body-instructions-marker", payload["messages"][0]["content"])
                        return "", "", [_call("list", "list_skills")]
                    last = payload["messages"][-1]
                    if last.get("name") == "list_skills":
                        self.assertIn("build", last["content"])
                        return "", "", [_call("read-0", "read_skill", name="build", max_chars=200)]
                    page = json.loads(last["content"])
                    pages.append(page["content"])
                    if page["next_offset"] is not None:
                        return "", "", [_call(
                            f"read-{page['next_offset']}", "read_skill", name="build",
                            max_chars=200, offset=page["next_offset"], revision=page["revision"],
                        )]
                    return "Skill loaded and used.", "", []

                with mock.patch.object(oa, "_stream_completion", side_effect=stream):
                    events = await self._run(approval=approval)
                self.assertEqual("".join(pages), self.skill_file.read_bytes().decode("utf-8"))
                self.assertTrue(any(event.get("text") == "Skill loaded and used." for event in events))
                for payload in requests:
                    names = {item["function"]["name"] for item in payload["tools"]}
                    self.assertTrue(agent_tools.SKILL_TOOL_NAMES <= names)

    async def test_planner_scope_exposes_only_skills_and_blocks_hallucinated_host_tools(self) -> None:
        requests = []
        secret = Path(self.ws, "secret.txt")
        secret.write_text("private-workspace-marker", encoding="utf-8")

        async def stream(*args, **_kwargs):
            payload = copy.deepcopy(args[1])
            requests.append(payload)
            if len(requests) == 1:
                return "", "", [
                    _call("host-read", "read_file", path="secret.txt"),
                    _call("host-write", "write_file", path="forbidden.txt", content="forbidden"),
                    _call("host-shell", "bash", command="must not execute"),
                    _call("skills", "list_skills"),
                    _call("skill", "read_skill", name="build"),
                ]
            return "[PLAN]\n1. [low] Apply the loaded skill.\n[/PLAN]", "", []

        with (
            agent_tools.skill_planning_scope(),
            mock.patch.object(oa, "_stream_completion", side_effect=stream),
        ):
            events = await self._run(approval="deny", compaction=True)
        self.assertEqual(len(requests), 2)
        for payload in requests:
            names = {item["function"]["name"] for item in payload["tools"]}
            self.assertEqual(names, set(agent_tools.SKILL_TOOL_NAMES))
            self.assertIn("Effective permission: confirm", payload["messages"][0]["content"])
        outputs = [event["output"] for event in events if event.get("type") == "tool_result"]
        self.assertEqual(sum("skill-only planning" in result for result in outputs), 3)
        self.assertTrue(any("body-instructions-marker" in result for result in outputs))
        self.assertFalse(any("private-workspace-marker" in result for result in outputs))
        self.assertFalse(Path(self.ws, "forbidden.txt").exists())
        self.assertFalse(agent_tools.skill_planning_active())

    async def test_chat_only_planner_receives_metadata_and_delegates_reading_without_host_tools(self) -> None:
        requests = []
        secret = Path(self.ws, "secret.txt")
        secret.write_text("private-workspace-marker", encoding="utf-8")

        async def stream(*args, **_kwargs):
            requests.append(copy.deepcopy(args[1]))
            if len(requests) == 1:
                return "", "", [
                    _call("host-read", "read_file", path="secret.txt"),
                    _call("host-write", "write_file", path="forbidden.txt", content="forbidden"),
                    _call("host-shell", "bash", command="must not execute"),
                ]
            return "[PLAN]\n1. [low] Load the full build skill, then follow its workflow.\n[/PLAN]", "", []

        with (
            agent_tools.skill_planning_scope(),
            mock.patch.object(oa, "_stream_completion", side_effect=stream),
        ):
            events = await self._run(approval="deny", compaction=True, backend=_ChatOnlySkillBackend())
        self.assertEqual(len(requests), 2)
        for payload in requests:
            self.assertNotIn("tools", payload)
            system = payload["messages"][0]["content"]
            self.assertIn("Fixture build workflow.", system)
            self.assertIn("This planner has no skill-reading interface", system)
            self.assertIn("from catalog metadata only", system)
            self.assertIn("worker to load the full skill", system)
            self.assertNotIn("body-instructions-marker", system)
        outputs = [event["output"] for event in events if event.get("type") == "tool_result"]
        self.assertEqual(sum("skill-only planning" in result for result in outputs), 3)
        self.assertFalse(any("private-workspace-marker" in result for result in outputs))
        self.assertFalse(Path(self.ws, "forbidden.txt").exists())
        self.assertFalse(agent_tools.skill_planning_active())

    async def test_ordinary_compaction_and_deny_stay_tool_free_without_catalog(self) -> None:
        for approval, compaction in (("full", True), ("deny", False)):
            with self.subTest(approval=approval, compaction=compaction):
                requests = []

                async def stream(*args, **_kwargs):
                    requests.append(copy.deepcopy(args[1]))
                    if len(requests) == 1:
                        return "", "", [_call("unexpected-skill", "read_skill", name="build")]
                    return "No tool access.", "", []

                with (
                    mock.patch.object(oa, "_stream_completion", side_effect=stream),
                    mock.patch.object(skills, "current_catalog") as catalog,
                ):
                    events = await self._run(approval=approval, compaction=compaction)
                catalog.assert_not_called()
                for payload in requests:
                    self.assertNotIn("tools", payload)
                    self.assertNotIn("Fixture build workflow", payload["messages"][0]["content"])
                self.assertTrue(any(event.get("output", "").startswith("Blocked:") for event in events))

    async def test_next_scope_refreshes_edited_catalog_without_discarding_saved_tool_evidence(self) -> None:
        record = _Record()
        first_requests = []

        async def first_stream(*args, **_kwargs):
            first_requests.append(copy.deepcopy(args[1]))
            if len(first_requests) == 1:
                return "", "", [_call("skill", "read_skill", name="build")]
            return "Original skill inspected.", "", []

        async with skills.scope(self.ws):
            with (
                mock.patch.object(oa, "current_run", return_value=record),
                mock.patch.object(oa, "_stream_completion", side_effect=first_stream),
            ):
                await self._run()
        previous_identity = record.get("http_identity")
        previous_system = first_requests[0]["messages"][0]["content"]
        self.skill_file.write_text(self.skill_file.read_text(encoding="utf-8") + "\nnew revision marker\n", encoding="utf-8")
        resumed_requests = []

        async def resumed_stream(*args, **_kwargs):
            resumed_requests.append(copy.deepcopy(args[1]))
            self.assertIn(skills.refresh_instructions(), args[1]["messages"][0]["content"])
            if len(resumed_requests) == 1:
                return "", "", [_call("skill-fresh", "read_skill", name="build")]
            self.assertIn("new revision marker", json.loads(args[1]["messages"][-1]["content"])["content"])
            return "Retained evidence and refreshed catalog.", "", []

        async with skills.scope(self.ws):
            with (
                mock.patch.object(oa, "current_run", return_value=record),
                mock.patch.object(oa, "_stream_completion", side_effect=resumed_stream),
            ):
                await self._run(prompt="Follow the current build skill for this next request")
        self.assertEqual(record.get("http_identity"), previous_identity)
        self.assertNotEqual(resumed_requests[0]["messages"][0]["content"], previous_system)
        self.assertTrue(any(
            message.get("role") == "tool" and "body-instructions-marker" in message.get("content", "")
            for message in resumed_requests[0]["messages"]
        ))
        self.assertEqual(len(resumed_requests), 2)

    async def test_removed_last_skill_cannot_be_reused_from_restored_http_history(self) -> None:
        record = _Record()
        first_calls = 0

        async def first_stream(*_args, **_kwargs):
            nonlocal first_calls
            first_calls += 1
            if first_calls == 1:
                return "", "", [_call("old-skill", "read_skill", name="build")]
            return "Earlier workflow used.", "", []

        async with skills.scope(self.ws):
            with (mock.patch.object(oa, "current_run", return_value=record),
                  mock.patch.object(oa, "_stream_completion", side_effect=first_stream)):
                await self._run()
        self.skill_file.unlink()
        resumed_requests = []

        async def next_stream(*args, **_kwargs):
            payload = copy.deepcopy(args[1])
            resumed_requests.append(payload)
            system = payload["messages"][0]["content"]
            self.assertIn(skills.refresh_instructions(), system)
            self.assertIn("No skills are currently discoverable", system)
            self.assertIn("Previously loaded skill instructions are unavailable", system)
            if len(resumed_requests) == 1:
                self.assertTrue(any(message.get("role") == "tool" and "body-instructions-marker" in message.get("content", "")
                                    for message in payload["messages"]))
                return "", "", [_call("current-skill", "read_skill", name="build")]
            self.assertEqual(json.loads(payload["messages"][-1]["content"])["error"], "Skill not found")
            return "The removed workflow is unavailable.", "", []

        async with skills.scope(self.ws):
            with (mock.patch.object(oa, "current_run", return_value=record),
                  mock.patch.object(oa, "_stream_completion", side_effect=next_stream)):
                await self._run(prompt="Use current instructions for the next request")
        self.assertEqual(len(resumed_requests), 2)
