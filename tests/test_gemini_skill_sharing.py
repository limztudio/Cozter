"""Gemini native skills share fresh sources while owned Cozter bridges stay portable."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from Cozter import skills


class GeminiSkillSharingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.ws = self.root / "workspace"
        self.ws.mkdir()
        self.home = self.root / "actual-home"
        self.home.mkdir()
        self.gemini_home = self.root / "gemini-home"
        self.gemini_home.mkdir()
        self.enterContext(mock.patch.object(skills.Path, "home", return_value=self.home))
        self.enterContext(mock.patch.dict(os.environ, {
            "CODEX_HOME": "", "CLAUDE_CONFIG_DIR": "", "COPILOT_HOME": "", "GROK_HOME": "",
            "COPILOT_SKILLS_DIRS": "", "GEMINI_CLI_HOME": str(self.gemini_home),
        }))

    def write(self, path, name, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\nname: {name}\ndescription: Stable skill metadata\n---\n{body}", encoding="utf-8")
        return path

    async def test_custom_home_is_parent_of_gemini_and_agents_skill_roots_without_scanning_extensions(self):
        gemini = self.write(self.gemini_home / ".gemini/skills/review/SKILL.md", "Gemini review", "Current native review")
        alias = self.write(self.gemini_home / ".agents/skills/check/SKILL.md", "Shared alias", "Current alias workflow")
        project = self.write(self.ws / ".gemini/skills/project/SKILL.md", "Project procedure", "Current project workflow")
        self.write(self.home / ".gemini/skills/wrong-home/SKILL.md", "Wrong home", "Not the configured Gemini home")
        self.write(self.gemini_home / "skills/wrong-level/SKILL.md", "Wrong level", "GEMINI_CLI_HOME is not .gemini")
        self.write(self.gemini_home / ".gemini/extensions/unverified/skills/hidden/SKILL.md",
                   "Unverified extension", "Directory existence does not establish an active extension")

        catalog = skills.discover(str(self.ws))

        self.assertEqual({entry.path for entry in catalog.entries}, {gemini.resolve(), alias.resolve(), project.resolve()})
        self.assertEqual({entry.source for entry in catalog.entries},
                         {"native-user:gemini", "native-user:gemini-agents", "native-project:gemini"})
        self.assertEqual(catalog.warnings, ())

    async def test_default_home_and_shared_alias_are_discovered_once_without_duplicate_catalog_entries(self):
        gemini = self.write(self.home / ".gemini/skills/review/SKILL.md", "Default Gemini", "Default home workflow")
        alias = self.write(self.home / ".agents/skills/shared/SKILL.md", "Default alias", "Shared native workflow")
        with mock.patch.dict(os.environ, {"GEMINI_CLI_HOME": ""}):
            catalog = skills.discover(str(self.ws))
        self.assertEqual({entry.path for entry in catalog.entries}, {gemini.resolve(), alias.resolve()})
        self.assertEqual(len(catalog.entries), 2)

    async def test_refresh_reloads_native_body_and_resources_and_exports_only_owned_cozter_sources(self):
        native = self.write(self.gemini_home / ".gemini/skills/review/SKILL.md", "Native Gemini", "Original native body")
        reference = native.parent / "checks.md"
        reference.write_text("Original checks", encoding="utf-8")
        project = self.write(self.ws / ".gemini/skills/local/SKILL.md", "Native project", "Foreign native instructions")
        flat = self.ws / ".cozter/skills/procedure.md"
        self.write(flat, "Cozter procedure", "Original Cozter body")

        async with skills.scope(str(self.ws)) as before:
            entry = next(entry for entry in before.entries if entry.path == native.resolve())
            native.write_text(native.read_text(encoding="utf-8").replace("Original native body", "Updated native body"),
                              encoding="utf-8")
            reference.write_text("Updated checks", encoding="utf-8")
            flat.write_text(flat.read_text(encoding="utf-8").replace("Original Cozter body", "Updated Cozter body"),
                            encoding="utf-8")

            refreshed = await skills.refresh(str(self.ws))

            current = next(row for row in refreshed.entries if row.id == entry.id)
            self.assertNotEqual(current.revision, entry.revision)
            self.assertIn("Updated native body", json.loads(skills.read_skill(str(self.ws), entry.id))["content"])
            self.assertEqual(json.loads(skills.read_skill(str(self.ws), entry.id, resource="checks.md"))["content"],
                             "Updated checks")
            self.assertIn("each new user request", skills.catalog_prompt(refreshed, cli=True))
            manifest = json.loads((self.ws / ".cozter/skill_bridges.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["files"]), 2)
            self.assertEqual({relative.split("/")[0] for relative in manifest["files"]}, {".agents", ".claude"})
            for relative in manifest["files"]:
                text = (self.ws / relative).read_text(encoding="utf-8")
                self.assertIn(json.dumps(str(skills._canonical(flat)), ensure_ascii=False), text)
                self.assertNotIn(json.dumps(str(skills._canonical(native)), ensure_ascii=False), text)
            self.assertIn("Foreign native instructions", project.read_text(encoding="utf-8"))
        self.assertEqual(len(skills.discover(str(self.ws)).entries), 3)

    async def test_project_gemini_roots_stop_at_repository_boundary_and_relative_home_does_not_fall_back(self):
        repo = self.ws / "repo"
        nested = repo / "src/component"
        nested.mkdir(parents=True)
        (repo / ".git").mkdir()
        parent = self.write(repo / ".gemini/skills/parent/SKILL.md", "Repo procedure", "Repository workflow")
        local = self.write(nested / ".gemini/skills/local/SKILL.md", "Local procedure", "Local workflow")
        self.write(self.ws / ".gemini/skills/outside/SKILL.md", "Outside repository", "Outside workflow")
        self.write(self.home / ".gemini/skills/wrong-home/SKILL.md", "Fallback home", "Wrong configured home")
        with mock.patch.dict(os.environ, {"GEMINI_CLI_HOME": "relative-home"}):
            catalog = skills.discover(str(nested))
        self.assertEqual({entry.path for entry in catalog.entries}, {parent.resolve(), local.resolve()})
