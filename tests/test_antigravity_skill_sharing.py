"""Antigravity CLI skills share fresh files through its documented native roots."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from Cozter import skills


class AntigravitySkillSharingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.ws = self.root / "workspace"
        self.ws.mkdir()
        self.home = self.root / "home"
        self.home.mkdir()
        self.enterContext(mock.patch.object(skills.Path, "home", return_value=self.home))
        self.enterContext(mock.patch.dict(os.environ, {
            "CODEX_HOME": "", "CLAUDE_CONFIG_DIR": "", "COPILOT_HOME": "", "GROK_HOME": "",
            "COPILOT_SKILLS_DIRS": "",
        }))

    def write(self, path, name, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\nname: {name}\ndescription: Stable native skill metadata\n---\n{body}", encoding="utf-8")
        return path

    async def test_cli_global_root_stays_distinct_from_app_ide_and_unverified_plugin_directories(self):
        personal = self.write(self.home / ".gemini/antigravity-cli/skills/review/SKILL.md",
                              "CLI review", "Actual Antigravity CLI workflow")
        project = self.write(self.ws / ".agents/skills/project/SKILL.md", "Project review", "Project workflow")
        self.write(self.home / ".gemini/config/skills/app-only/SKILL.md", "App procedure", "App and IDE location")
        self.write(self.home / ".gemini/antigravity-cli/plugins/unverified/skills/hidden/SKILL.md",
                   "Inactive plugin", "Directory existence does not establish plugin activation")
        self.write(self.home / ".gemini/skills/unrelated/SKILL.md", "Other native user", "Unrelated user workflow")
        self.write(self.ws / ".gemini/skills/unrelated/SKILL.md", "Other native project", "Unrelated project workflow")

        catalog = skills.discover(str(self.ws))

        self.assertEqual({entry.path for entry in catalog.entries}, {personal.resolve(), project.resolve()})
        self.assertEqual({entry.source for entry in catalog.entries},
                         {"native-user:antigravity-cli", "native-project:agents"})
        self.assertEqual(catalog.warnings, ())

    async def test_resumed_scope_refreshes_native_body_references_and_owned_portable_bridge(self):
        native = self.write(self.home / ".gemini/antigravity-cli/skills/review/SKILL.md",
                            "Native review", "Old native instructions")
        reference = native.parent / "checks.md"
        reference.write_text("Old native checks", encoding="utf-8")
        source = self.write(self.ws / ".cozter/skills/procedure.md", "Cozter procedure", "Old workspace procedure")
        foreign = self.write(self.ws / ".agents/skills/foreign/SKILL.md", "Foreign native", "Preserve native edits")
        async with skills.scope(str(self.ws)) as initial:
            first = next(entry for entry in initial.entries if entry.path == native.resolve())
            native.write_text(native.read_text(encoding="utf-8").replace("Old native", "New native"), encoding="utf-8")
            reference.write_text("New native checks", encoding="utf-8")
            source.write_text(source.read_text(encoding="utf-8").replace("Old workspace", "New workspace"), encoding="utf-8")

            refreshed = await skills.refresh(str(self.ws))

            current = next(entry for entry in refreshed.entries if entry.id == first.id)
            self.assertNotEqual(current.revision, first.revision)
            self.assertIn("New native instructions", json.loads(skills.read_skill(str(self.ws), first.id))["content"])
            self.assertEqual(json.loads(skills.read_skill(str(self.ws), first.id, resource="checks.md"))["content"],
                             "New native checks")
            self.assertIn("worker starting or resuming", skills.catalog_prompt(refreshed, cli=True))
            manifest = json.loads((self.ws / ".cozter/skill_bridges.json").read_text(encoding="utf-8"))
            native_exports = [relative for relative in manifest["files"] if relative.startswith(".agents/")]
            self.assertEqual(len(native_exports), 1)
            bridge = (self.ws / native_exports[0]).read_text(encoding="utf-8")
            flat = next(entry for entry in refreshed.entries if entry.path == source.resolve())
            self.assertIn(flat.revision, bridge)
            self.assertIn(json.dumps(str(skills._canonical(source)), ensure_ascii=False), bridge)
            self.assertEqual(foreign.read_text(encoding="utf-8").split("---\n")[-1], "Preserve native edits")
        self.assertEqual(len(skills.discover(str(self.ws)).entries), 3)
