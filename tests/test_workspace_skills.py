"""Bounded skill discovery, source paging, and non-destructive native bridges."""

import asyncio
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from Cozter import skills


class WorkspaceSkillsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.ws = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.object(skills, "_native_roots", return_value=[]))
        self.source = self.ws / ".cozter" / "skills"

    def write(self, name: str, text: str) -> Path:
        self.source.mkdir(parents=True, exist_ok=True)
        path = self.source / name
        path.write_bytes(text.encode("utf-8"))
        return path

    async def test_discovery_is_read_only_and_every_flat_markdown_file_is_a_skill(self) -> None:
        self.assertEqual(skills.discover(str(self.ws)).entries, ())
        self.assertFalse(self.source.exists())
        self.assertEqual(skills.catalog_prompt(skills.discover(str(self.ws))), "")
        self.write("README.md", "# Workspace procedure\n\nRepeat this workflow.\n")
        self.write("other.MD", "---\nname: Explicit\ndescription: Narrow scope\n---\nBody")
        self.write("ignored.txt", "not a skill")
        (self.source / "nested").mkdir()
        (self.source / "nested" / "nested.md").write_text("not flat", encoding="utf-8")

        catalog = skills.discover(str(self.ws))

        self.assertEqual([entry.name for entry in catalog.entries], ["Explicit", "Workspace procedure"])
        self.assertEqual(catalog.entries[0].description, "Narrow scope")
        self.assertEqual(catalog.entries[1].description, "Repeat this workflow.")
        self.assertEqual(catalog.warnings, ())

    async def test_frontmatter_multiline_scalars_are_metadata_without_executable_fields(self) -> None:
        for marker in (">", ">-", "|", "|-"):
            with self.subTest(marker=marker):
                text = f"---\nname: 'Review'\ndescription: {marker}\n  Review the change\n  and verify it.\nallowed-tools: '*'\n---\nBody"
                name, description = skills._metadata(text, "fallback")
                self.assertEqual(name, "Review")
                self.assertEqual(description, "Review the change and verify it.")

    async def test_ids_stay_stable_when_metadata_changes_and_names_are_unambiguous(self) -> None:
        first = self.write("first.md", "---\nname: Same\n---\nFirst body")
        self.write("second.md", "---\nname: Same\n---\nSecond body")
        before = skills.discover(str(self.ws))
        ambiguous = json.loads(skills.read_skill(str(self.ws), "same"))
        self.assertIn("Ambiguous", ambiguous["error"])
        self.assertEqual(len(ambiguous["matches"]), 2)
        entry = before.entries[0]
        self.assertEqual(json.loads(skills.read_skill(str(self.ws), entry.id))["content"], first.read_text())
        first.write_text("---\nname: Renamed\n---\nUpdated body", encoding="utf-8")
        after = skills.discover(str(self.ws)).entries[0]
        self.assertEqual(after.id, entry.id)
        self.assertNotEqual(after.revision, entry.revision)

    async def test_main_body_pages_reassemble_exact_unicode_and_pin_revisions(self) -> None:
        body = "# Unicode procedure\n\n" + "한글🙂\n" * 70
        self.write("unicode.md", body)
        entry = skills.discover(str(self.ws)).entries[0]
        offset = 0
        parts = []
        while True:
            page = json.loads(skills.read_skill(str(self.ws), entry.id, offset=offset,
                                               max_chars=31, revision=entry.revision))
            parts.append(page["content"])
            if page["next_offset"] is None:
                break
            self.assertGreater(page["next_offset"], offset)
            offset = page["next_offset"]
        self.assertEqual("".join(parts), body)
        changed = json.loads(skills.read_skill(str(self.ws), entry.id, revision="old"))
        self.assertIn("revision changed", changed["error"])

    async def test_explicit_tools_refresh_files_inside_a_bound_prompt_scope(self) -> None:
        path = self.write("live.md", "# Original\n\nOld body")
        async with skills.scope(str(self.ws)) as catalog:
            previous = catalog.entries[0]
            path.write_text("# Updated\n\nFresh body", encoding="utf-8")
            listing = json.loads(skills.list_skills(str(self.ws)))
            self.assertNotEqual(listing["skills"][0]["revision"], previous.revision)
            self.assertIn("Fresh body", json.loads(skills.read_skill(str(self.ws), previous.id))["content"])
            pinned = json.loads(skills.read_skill(str(self.ws), previous.id, revision=previous.revision))
            self.assertIn("revision changed", pinned["error"])

    async def test_refresh_replaces_bound_body_only_revision_and_owned_bridges(self) -> None:
        header = "---\nname: Stable\ndescription: Stable metadata\n---\n"
        path = self.write("stable.md", header + "Old body")
        async with skills.scope(str(self.ws)) as original:
            before = original.entries[0]
            stamp = path.stat()
            path.write_text(header + "New body", encoding="utf-8")
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            refreshed = await skills.refresh(str(self.ws))
            self.assertIs(skills.current_catalog(str(self.ws)), refreshed)
            self.assertEqual(refreshed.entries[0].id, before.id)
            self.assertEqual(refreshed.entries[0].description, before.description)
            self.assertNotEqual(refreshed.entries[0].revision, before.revision)
            self.assertIn("New body", json.loads(skills.read_skill(str(self.ws), before.id))["content"])
            manifest = json.loads((self.ws / ".cozter" / "skill_bridges.json").read_text())
            for relative in manifest["files"]:
                self.assertIn(refreshed.entries[0].revision, (self.ws / relative).read_text())
        self.assertIsNone(skills._CURRENT.get())

    async def test_refresh_adds_and_removes_sources_without_touching_foreign_bridges(self) -> None:
        path = self.write("old.md", "# Old\n\nOld body")
        async with skills.scope(str(self.ws)):
            manifest = json.loads((self.ws / ".cozter" / "skill_bridges.json").read_text())
            owned = [self.ws / relative for relative in manifest["files"]]
            foreign = self.ws / ".agents" / "skills" / "foreign" / "SKILL.md"
            foreign.parent.mkdir(parents=True)
            foreign.write_text("foreign instructions", encoding="utf-8")
            path.unlink()
            self.write("new.md", "# New\n\nNew body")
            refreshed = await skills.refresh(str(self.ws))
            self.assertEqual([entry.name for entry in refreshed.entries], ["New"])
            self.assertTrue(all(not old.exists() for old in owned))
            self.assertEqual(foreign.read_text(), "foreign instructions")
            self.assertIn("New", skills.catalog_prompt(skills.current_catalog(str(self.ws))))
            (self.source / "new.md").unlink()
            empty = await skills.refresh(str(self.ws))
            self.assertEqual(empty.entries, ())
            self.assertEqual(skills.catalog_prompt(empty), "")

    async def test_refresh_reloads_native_sources_and_needed_resources(self) -> None:
        native_root = self.ws / "personal-native"
        native_folder = native_root / "review"
        native_folder.mkdir(parents=True)
        source = native_folder / "SKILL.md"
        source.write_text("---\nname: Review\ndescription: Stable\n---\nOld workflow", encoding="utf-8")
        resource = native_folder / "checks.md"
        resource.write_text("Old checks", encoding="utf-8")
        with mock.patch.object(skills, "_native_roots", return_value=[(native_root, "native-user:fixture")]):
            async with skills.scope(str(self.ws)) as original:
                first = original.entries[0]
                source.write_text("---\nname: Review\ndescription: Stable\n---\nNew workflow", encoding="utf-8")
                refreshed = await skills.refresh(str(self.ws))
                self.assertEqual(refreshed.entries[0].id, first.id)
                self.assertNotEqual(refreshed.entries[0].revision, first.revision)
                self.assertIn("New workflow", json.loads(skills.read_skill(str(self.ws), first.id))["content"])
                unchanged_catalog = skills._catalog_revision(refreshed)
                resource.write_text("New checks", encoding="utf-8")
                references_refreshed = await skills.refresh(str(self.ws))
                self.assertEqual(skills._catalog_revision(references_refreshed), unchanged_catalog)
                self.assertEqual(json.loads(skills.read_skill(str(self.ws), first.id, resource="checks.md"))["content"], "New checks")
                self.assertIn("supporting references", skills.refresh_instructions(cli=True))

    async def test_refresh_outside_scope_or_for_another_workspace_does_not_leak(self) -> None:
        self.write("first.md", "First instructions")
        outside = await skills.refresh(str(self.ws))
        self.assertTrue(outside.entries)
        self.assertIsNone(skills._CURRENT.get())
        other = Path(self.enterContext(tempfile.TemporaryDirectory()))
        async with skills.scope(str(self.ws)) as bound:
            refreshed_other = await skills.refresh(str(other))
            self.assertEqual(refreshed_other.workspace_path, str(skills._canonical(other)))
            self.assertIs(skills.current_catalog(str(self.ws)), bound)
        self.assertIsNone(skills._CURRENT.get())

    async def test_concurrent_refresh_updates_each_callers_scope_only(self) -> None:
        self.write("first.md", "# First\n\nOriginal")
        other = Path(self.enterContext(tempfile.TemporaryDirectory()))
        directory = other / ".cozter" / "skills"
        directory.mkdir(parents=True)
        second = directory / "second.md"
        second.write_text("# Second\n\nOriginal", encoding="utf-8")
        ready = [asyncio.Event(), asyncio.Event()]

        async def use(path, source, index):
            async with skills.scope(str(path)) as original:
                source.write_text(source.read_text() + f"\nchanged-{index}", encoding="utf-8")
                ready[index].set()
                await ready[1 - index].wait()
                refreshed = await skills.refresh(str(path))
                await asyncio.sleep(0)
                self.assertIs(skills.current_catalog(str(path)), refreshed)
                self.assertNotEqual(original.entries[0].revision, refreshed.entries[0].revision)
                return refreshed.workspace_path

        first_path, second_path = await asyncio.gather(use(self.ws, self.source / "first.md", 0), use(other, second, 1))
        self.assertNotEqual(first_path, second_path)
        self.assertIsNone(skills._CURRENT.get())

    async def test_cancelled_refresh_settles_owned_bridge_work_without_rebinding(self) -> None:
        self.write("first.md", "Original body")
        async with skills.scope(str(self.ws)) as original:
            started = asyncio.Event()
            release = threading.Event()
            finished = threading.Event()
            loop = asyncio.get_running_loop()
            sync = skills._sync_bridges

            def blocked(catalog):
                loop.call_soon_threadsafe(started.set)
                try:
                    if not release.wait(2):
                        raise RuntimeError("Test did not release bridge preparation")
                    return sync(catalog)
                finally:
                    finished.set()

            with mock.patch.object(skills, "_sync_bridges", side_effect=blocked):
                refreshing = asyncio.create_task(skills.refresh(str(self.ws)))
                try:
                    await asyncio.wait_for(started.wait(), 1)
                    refreshing.cancel()
                    await asyncio.sleep(0)
                    refreshing.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(refreshing.done())
                    self.assertIs(skills.current_catalog(str(self.ws)), original)
                finally:
                    release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(refreshing, 1)
                self.assertTrue(finished.is_set())
                self.assertIs(skills.current_catalog(str(self.ws)), original)

    def test_refresh_rule_is_explicit_for_current_request_api_and_native_cli(self) -> None:
        catalog = skills.SkillCatalog(str(self.ws), (), ("Discovery warning",))
        for cli in (False, True):
            with self.subTest(cli=cli):
                rule = skills.refresh_instructions(cli=cli)
                self.assertIn("each new user request", rule)
                self.assertIn("worker starting or resuming", rule)
                self.assertIn("unchanged", rule)
                self.assertIn("historical context", rule)
                self.assertIn("permissions", rule)
                self.assertIn(rule, skills.catalog_prompt(catalog, cli=cli))
                self.assertIn("native CLI SKILL.md" if cli else "read_skill", rule)

    async def test_native_roots_deduplicate_paths_and_share_relative_resource_pages(self) -> None:
        root = self.ws / "native"
        folder = root / "review"
        (folder / "references").mkdir(parents=True)
        source = folder / "SKILL.md"
        source.write_text("---\nname: Native review\ndescription: Review code\n---\nRead references/check.md", encoding="utf-8")
        reference = folder / "references" / "check.md"
        reference.write_text("check" * 50, encoding="utf-8")
        with mock.patch.object(skills, "_native_roots", return_value=[(root, "native-user:test"), (root, "other")]):
            catalog = skills.discover(str(self.ws))
            self.assertEqual(len(catalog.entries), 1)
            entry = catalog.entries[0]
            self.assertEqual(entry.root, skills._canonical(folder))
            first = json.loads(skills.read_skill(str(self.ws), entry.id, resource="references/check.md", max_chars=20))
            self.assertEqual(first["content"], "check" * 4)
            self.assertEqual(first["next_offset"], 20)
            self.assertNotEqual(first["revision"], first["skill_revision"])
            second = json.loads(skills.read_skill(str(self.ws), entry.id, resource="references/check.md",
                                                offset=20, revision=first["revision"]))
            self.assertEqual(first["content"] + second["content"], reference.read_text())
            reference.write_text("changed", encoding="utf-8")
            self.assertIn("revision changed", json.loads(skills.read_skill(
                str(self.ws), entry.id, resource="references/check.md", revision=first["revision"],
            ))["error"])

    async def test_resource_paths_cannot_read_outside_selected_skill(self) -> None:
        self.write("procedure.md", "Read a reference")
        for resource in ("../settings.json", "/secret", "C:\\secret", "file:stream", "a/../../secret", "a\x00b"):
            with self.subTest(resource=resource), self.assertRaises(ValueError):
                skills.read_skill(str(self.ws), "procedure", resource=resource)
        sibling = self.ws / "outside.md"
        sibling.write_text("private", encoding="utf-8")
        try:
            (self.source / "reference.md").symlink_to(sibling)
        except OSError:
            return
        result = json.loads(skills.read_skill(str(self.ws), "procedure", resource="reference.md"))
        self.assertIn("leaves", result["error"])

    async def test_non_utf8_oversized_and_escaping_skill_files_report_warnings(self) -> None:
        self.write("valid.md", "valid")
        (self.source / "binary.md").write_bytes(b"\xff\xfe")
        (self.source / "large.md").write_bytes(b"x" * (skills.MAX_SKILL_BYTES + 1))
        catalog = skills.discover(str(self.ws))
        self.assertEqual(len(catalog.entries), 1)
        self.assertEqual(len(catalog.warnings), 2)
        self.assertTrue(any("exceeds" in warning for warning in catalog.warnings))

    async def test_catalog_and_list_paging_preserve_complete_metadata_and_cursors(self) -> None:
        for index in range(80):
            self.write(f"skill-{index:03}.md", f"---\nname: Skill{index}\ndescription: " + "description " * 80 + "\n---\nBody")
        catalog = skills.discover(str(self.ws))
        prompt = skills.catalog_prompt(catalog)
        self.assertLessEqual(len(prompt), skills.MAX_CATALOG_CHARS)
        self.assertIn("Showing", prompt)
        self.assertIn("list_skills(offset=", prompt)
        cli = skills.catalog_prompt(catalog, cli=True)
        self.assertIn("python -m Cozter.agent_tools.builtin.list_skills", cli)
        self.assertNotIn("Use list_skills(offset=", cli)
        found = []
        offset = 0
        while True:
            raw = skills.list_skills(str(self.ws), offset=offset, max_results=200)
            self.assertLessEqual(len(raw), skills.MAX_RESPONSE_CHARS)
            page = json.loads(raw)
            found.extend(row["id"] for row in page["skills"])
            if page["next_offset"] is None:
                break
            self.assertGreater(page["next_offset"], offset)
            offset = page["next_offset"]
        self.assertEqual(found, [entry.id for entry in catalog.entries])

    async def test_read_response_escaping_keeps_cursor_under_tool_cap(self) -> None:
        self.write("escaped.md", "\t" * 48_000)
        raw = skills.read_skill(str(self.ws), "escaped", max_chars=48_000)
        self.assertLessEqual(len(raw), skills.MAX_RESPONSE_CHARS)
        page = json.loads(raw)
        self.assertGreater(page["next_offset"], 0)
        self.assertEqual(len(page["content"]), page["next_offset"])

    async def test_scope_exports_valid_owned_native_wrappers_without_native_permission_fields(self) -> None:
        self.write("release.md", "---\nname: Release\ndescription: Release the project\nallowed-tools: '*'\nmodel: expensive\n---\nSteps")
        async with skills.scope(str(self.ws)) as catalog:
            self.assertIs(skills.current_catalog(str(self.ws)), catalog)
            manifest = json.loads((self.ws / ".cozter" / "skill_bridges.json").read_text())
            self.assertEqual(len(manifest["files"]), 2)
            for relative in manifest["files"]:
                text = (self.ws / relative).read_text(encoding="utf-8")
                self.assertEqual(text.splitlines()[0], "---")
                name, description = skills._metadata(text, "fallback")
                self.assertTrue(name.startswith("cozter-"))
                self.assertEqual(description, "Release the project")
                self.assertIn(str(catalog.entries[0].path), text.replace("\\\\", "\\"))
                self.assertNotIn("allowed-tools", text)
                self.assertNotIn("model:", text)
                self.assertTrue(skills._is_bridge(text))
        # Generated native wrappers must not be imported as additional skills.
        with mock.patch.object(skills, "_native_roots", return_value=[(self.ws / ".agents" / "skills", "native")]):
            self.assertEqual(len(skills.discover(str(self.ws)).entries), 1)

    async def test_bridge_updates_and_removes_only_owned_unchanged_files(self) -> None:
        source = self.write("release.md", "# Release\n\nFirst instructions")
        async with skills.scope(str(self.ws)):
            pass
        manifest_path = self.ws / ".cozter" / "skill_bridges.json"
        initial = json.loads(manifest_path.read_text())["files"]
        paths = [self.ws / relative for relative in initial]
        foreign = self.ws / ".agents" / "skills" / "user-skill" / "SKILL.md"
        foreign.parent.mkdir(parents=True)
        foreign.write_text("user content", encoding="utf-8")
        source.write_text("# Release\n\nChanged instructions", encoding="utf-8")
        async with skills.scope(str(self.ws)) as catalog:
            for path in paths:
                self.assertIn(catalog.entries[0].revision, path.read_text())
        paths[0].write_text("user modified", encoding="utf-8")
        source.unlink()
        async with skills.scope(str(self.ws)) as catalog:
            self.assertTrue(any("Preserved foreign or modified" in warning for warning in catalog.warnings))
        self.assertEqual(paths[0].read_text(), "user modified")
        self.assertFalse(paths[1].exists())
        self.assertEqual(foreign.read_text(), "user content")

    async def test_foreign_bridge_collision_and_untrusted_manifest_are_preserved(self) -> None:
        self.write("skill.md", "instructions")
        async with skills.scope(str(self.ws)):
            pass
        manifest = self.ws / ".cozter" / "skill_bridges.json"
        original = json.loads(manifest.read_text())["files"]
        first = self.ws / next(iter(original))
        first.write_text("foreign instructions", encoding="utf-8")
        async with skills.scope(str(self.ws)) as catalog:
            self.assertTrue(catalog.warnings)
        self.assertEqual(first.read_text(), "foreign instructions")
        manifest.write_text('{"version":999,"files":{}}', encoding="utf-8")
        async with skills.scope(str(self.ws)) as catalog:
            self.assertTrue(any("sync skipped" in warning for warning in catalog.warnings))
        self.assertEqual(first.read_text(), "foreign instructions")
        self.assertEqual(json.loads(manifest.read_text())["version"], 999)

    async def test_simultaneous_workspace_scopes_do_not_share_catalogs_or_sources(self) -> None:
        self.write("first.md", "First instructions")
        other = Path(self.enterContext(tempfile.TemporaryDirectory()))
        directory = other / ".cozter" / "skills"
        directory.mkdir(parents=True)
        (directory / "second.md").write_text("Second instructions", encoding="utf-8")

        async def use(path, expected):
            async with skills.scope(str(path)) as catalog:
                await asyncio.sleep(0)
                self.assertEqual([skill.name for skill in skills.current_catalog(str(path)).entries], [expected])
                self.assertIs(skills.current_catalog(str(path)), catalog)
                return catalog.workspace_path

        first, second = await asyncio.gather(use(self.ws, "first"), use(other, "second"))
        self.assertNotEqual(first, second)
        self.assertIsNone(skills._CURRENT.get())

    async def test_invalid_page_options_reject_bool_and_unbounded_requested_pages(self) -> None:
        self.write("skill.md", "instructions")
        for value in (True, -1, "0", 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                skills.list_skills(str(self.ws), offset=value)
        for value in (0, True, 201):
            with self.subTest(value=value), self.assertRaises(ValueError):
                skills.list_skills(str(self.ws), max_results=value)
        for value in (0, True, 48_001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                skills.read_skill(str(self.ws), "skill", max_chars=value)


if __name__ == "__main__":
    unittest.main()
