"""Build screenshots stay local; automatic output previews have an attempt budget."""

import os
import tempfile
import unittest
from unittest import mock

from Cozter import agent
from Cozter.backends_agent.base import AgentResult, ChatEvent


def image(root: str, relative: str, *, size: int = 32, order: int = 0) -> str:
    path = os.path.join(root, relative)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as output:
        output.write(b"\x89PNG\r\n\x1a\n")
        output.truncate(size)
    stamp = 1_700_000_000_000_000_000 + order * 1_000_000
    os.utime(path, ns=(stamp, stamp))
    return os.path.realpath(path)


class AutomaticImageDiscoveryTests(unittest.TestCase):
    def test_generated_build_test_and_cache_trees_are_not_scanned_but_outputs_are(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            for directory in (
                "__cmake/build/windows/Testing/smoke", "Build", "CMakeFiles", "Testing",
                "cmake-build-debug", "CMAKE-BUILD-release", "test-results", "test-artifacts",
                "test-output", "_deps", ".cache", ".GIT", "node_modules", "__pycache__",
            ):
                image(ws, os.path.join(directory, "fixture.png"))
            expected = {image(ws, os.path.join(directory, "result.png"))
                        for directory in ("artifacts", "output", "generated_images", ".cozter/generated_images")}
            expected.add(image(ws, "final.png"))
            self.assertEqual(set(agent._snapshot_attachment_images(ws)), expected)

    def test_file_symlink_alias_into_a_skipped_build_tree_is_not_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            source = image(ws, "build/fixture.png")
            alias = os.path.join(ws, "looks-like-output.png")
            try:
                os.symlink(source, alias)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"file symlinks unavailable: {exc}")
            self.assertEqual(agent._snapshot_attachment_images(ws), {})

    def test_file_count_cap_is_applied_before_attachment_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            paths = [image(ws, f"output/{index}.png", order=index) for index in range(12)]
            selection = agent._AutomaticImageSelection()
            with mock.patch.object(agent, "prepare_attachment_path", wraps=agent.prepare_attachment_path) as prepare:
                selected = agent._collect_new_attachment_images({}, ws, selection=selection)
            self.assertEqual(selected, paths[:8])
            self.assertEqual(prepare.call_count, 8)
            self.assertEqual(selection.omitted_sources, set(paths[8:]))
            self.assertEqual(selection.total_bytes, 8 * 32)

    def test_byte_cap_can_skip_an_oversized_candidate_and_keep_a_later_fitting_output(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            first = image(ws, "output/first.png", size=12 * 1024 * 1024, order=1)
            too_large = image(ws, "output/too-large.png", size=9 * 1024 * 1024, order=2)
            fits = image(ws, "output/fits.png", size=8 * 1024 * 1024, order=3)
            selection = agent._AutomaticImageSelection()
            with mock.patch.object(agent, "prepare_attachment_path", wraps=agent.prepare_attachment_path) as prepare:
                selected = agent._collect_new_attachment_images({}, ws, selection=selection)
            self.assertEqual(selected, [first, fits])
            self.assertEqual(selection.total_bytes, 20 * 1024 * 1024)
            self.assertEqual(selection.omitted_sources, {too_large})
            self.assertEqual([call.args[0] for call in prepare.call_args_list], [first, fits])

    def test_explicit_build_image_is_authorized_even_when_automatic_discovery_skips_it(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            path = image(ws, "__cmake/build/Testing/meaningful-proof.png")
            self.assertEqual(agent._collect_new_attachment_images({}, ws), [])
            self.assertEqual(agent.prepare_attachment_path(path, ws), path)
            text, sources = agent.extract_attachment_sources(f"Proof [[attach: {path}]]", ws)
            self.assertEqual(text, "Proof")
            self.assertEqual(sources, [path])


class AutomaticImageAttemptBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_file_budget_is_shared_across_phases_and_notice_does_not_change_judged_answer(self) -> None:
        with tempfile.TemporaryDirectory() as ws, agent.turn_budget(fresh=True) as budget:
            first_paths = [image(ws, f"output/first-{index}.png", order=index) for index in range(6)]
            first = AgentResult(text="Initial work complete")
            await agent._append_new_attachment_images(first, {}, ws)
            before_second = agent._snapshot_attachment_images(ws)
            second_paths = [image(ws, f"output/second-{index}.png", order=10 + index) for index in range(4)]
            second = AgentResult(text="Requested work verified")
            await agent._append_new_attachment_images(second, before_second, ws)
            self.assertEqual([event.content for event in first.events if event.kind == "attachment"], first_paths)
            self.assertEqual([event.content for event in second.events if event.kind == "attachment"], second_paths[:2])
            self.assertEqual(budget.auto_attachment_files, 8)
            self.assertEqual(budget.auto_attachment_bytes, 8 * 32)
            self.assertEqual(budget.auto_attachment_omitted_sources, set(second_paths[2:]))
            self.assertEqual(second.text, "Requested work verified")
            notices = [event.content for event in second.events
                       if event.kind == "text" and "Automatic image attachments" in event.content]
            self.assertEqual(len(notices), 1)
            self.assertIn("2 additional image(s) remain in the workspace", notices[0])
            self.assertNotIn("PARTIAL", notices[0])
            self.assertIn("Requested work verified", [event.content for event in second.events if event.kind == "text"])

    async def test_byte_budget_is_shared_across_phases(self) -> None:
        with tempfile.TemporaryDirectory() as ws, agent.turn_budget(fresh=True) as budget:
            image(ws, "output/initial.png", size=12 * 1024 * 1024)
            await agent._append_new_attachment_images(AgentResult(text="First"), {}, ws)
            before = agent._snapshot_attachment_images(ws)
            too_large = image(ws, "output/large.png", size=9 * 1024 * 1024, order=1)
            fits = image(ws, "output/fits.png", size=8 * 1024 * 1024, order=2)
            result = AgentResult(text="Final")
            await agent._append_new_attachment_images(result, before, ws)
            self.assertEqual([event.content for event in result.events if event.kind == "attachment"], [fits])
            self.assertEqual(budget.auto_attachment_bytes, 20 * 1024 * 1024)
            self.assertEqual(budget.auto_attachment_omitted_sources, {too_large})

    async def test_explicit_images_and_markers_do_not_consume_the_automatic_budget(self) -> None:
        with tempfile.TemporaryDirectory() as ws, agent.turn_budget(fresh=True) as budget:
            explicit = [image(ws, f"output/explicit-{index}.png", order=index) for index in range(10)]
            build_proof = image(ws, "__cmake/build/Testing/proof.png", size=21 * 1024 * 1024)
            automatic = image(ws, "output/automatic.png", order=20)
            text = "Requested explicit images:\n" + "\n".join(f"[[attach: {path}]]" for path in explicit)
            # Bare result.text markers must count as explicit even before text events exist.
            result = AgentResult(text=text, events=[ChatEvent(kind="attachment", content=build_proof)])
            await agent._append_new_attachment_images(result, {}, ws)
            self.assertEqual([event.content for event in result.events if event.kind == "attachment"],
                             [build_proof, automatic])
            self.assertEqual(result.text, text)
            self.assertEqual(budget.auto_attachment_files, 1)
            self.assertEqual(budget.auto_attachment_bytes, 32)
            self.assertEqual(budget.auto_attachment_omitted_sources, set())

    async def test_selected_sources_are_not_prepared_or_charged_twice_in_one_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as ws, agent.turn_budget(fresh=True) as budget:
            path = image(ws, "output/result.png")
            await agent._append_new_attachment_images(AgentResult(text="First"), {}, ws)
            with mock.patch.object(agent, "prepare_attachment_path", wraps=agent.prepare_attachment_path) as prepare:
                later = AgentResult(text="Later")
                await agent._append_new_attachment_images(later, {}, ws)
            prepare.assert_not_called()
            self.assertEqual(budget.auto_attachment_files, 1)
            self.assertEqual(budget.auto_attachment_sources, {path})
            self.assertFalse(any(event.kind == "attachment" for event in later.events))

    async def test_later_explicit_attachment_is_no_longer_reported_as_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as ws, agent.turn_budget(fresh=True) as budget:
            path = image(ws, "output/requested-large.png", size=21 * 1024 * 1024)
            await agent._append_new_attachment_images(AgentResult(text="First"), {}, ws)
            self.assertEqual(budget.auto_attachment_omitted_sources, {path})
            later = AgentResult(text="Explicitly selected output", events=[ChatEvent(kind="attachment", content=path)])
            await agent._append_new_attachment_images(later, {}, ws)
            self.assertEqual(budget.auto_attachment_omitted_sources, set())
            self.assertFalse(any("Automatic image attachments" in event.content
                                 for event in later.events if event.kind == "text"))
            self.assertEqual([event.content for event in later.events if event.kind == "attachment"], [path])

    async def test_fresh_attempt_resets_the_automatic_preview_allowance(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            for index in range(9):
                image(ws, f"output/{index}.png", order=index)
            for _ in range(2):
                with agent.turn_budget(fresh=True) as budget:
                    result = AgentResult(text="Complete")
                    await agent._append_new_attachment_images(result, {}, ws)
                    self.assertEqual(budget.auto_attachment_files, 8)
                    self.assertEqual(len(budget.auto_attachment_omitted_sources), 1)
                    self.assertEqual(sum(event.kind == "attachment" for event in result.events), 8)
