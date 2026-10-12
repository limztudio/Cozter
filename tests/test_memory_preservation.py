"""Durable memory must survive partially covered maintenance input."""

import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from Cozter import colony, compaction, session, workspace


class PartialMemoryPreservationTests(unittest.IsolatedAsyncioTestCase):
    async def test_colony_partial_pass_cannot_evict_full_shared_or_local_memory(self):
        with (
            tempfile.TemporaryDirectory() as ws,
            mock.patch.object(colony, "CONSOLIDATE_MAX_INPUT_CHARS", 1_200),
            mock.patch.object(colony, "CONSOLIDATE_COLONY_CHARS", 100),
        ):
            data = await asyncio.to_thread(session.create_session, ws)
            local = [f"Local fact {index}." for index in range(session.LONG_TERM_CAP)]
            # The huge first fact hides every later fact from consolidation.
            local[0] = "x" * colony.CONSOLIDATE_MAX_INPUT_CHARS * 2
            data["long_term"] = local
            await asyncio.to_thread(session.save_session, ws, data["id"], data)
            shared = [f"Shared fact {index}." for index in range(colony.COLONY_CAP)]
            await asyncio.to_thread(colony.set_items, ws, shared)
            output = (
                "[COLONY]\n- New shared fact.\n[/COLONY]\n"
                f"[SESSION:{data['id']}]\n- New local fact.\n[/SESSION]"
            )
            with mock.patch.object(colony, "run_internal_backend", new=mock.AsyncMock(return_value=output)):
                self.assertTrue(await colony.consolidate(ws, backend_name="codex"))
            self.assertEqual(colony.get_items(ws), shared)
            self.assertEqual(session.load_session(ws, data["id"])["long_term"], local)

    async def test_compaction_reserves_capacity_for_every_omitted_fact(self):
        with (
            tempfile.TemporaryDirectory() as ws,
            mock.patch.object(compaction, "MAX_SUMMARY_CHARS", 5_000),
        ):
            data = await asyncio.to_thread(session.create_session, ws)
            retained = [f"Retained fact {index}." for index in range(session.LONG_TERM_CAP)]
            retained[-1] = "x" * compaction.MAX_SUMMARY_CHARS
            data["long_term"] = retained
            data["messages"] = [{"role": "user", "content": "hello"} for _ in range(6)]
            await asyncio.to_thread(session.save_session, ws, data["id"], data)
            output = (
                "[SUMMARY]" + "s" * 100 + "[/SUMMARY]"
                "[LONG_TERM]\n- New fact.\n[/LONG_TERM]"
            )
            with (
                mock.patch.object(workspace, "get_compact_interval", return_value=1),
                mock.patch.object(compaction, "run_internal_backend", new=mock.AsyncMock(return_value=output)),
                mock.patch.object(colony, "maybe_trigger"),
            ):
                await compaction.maybe_compact(ws, data["id"], backend_name="codex")
            self.assertEqual(session.load_session(ws, data["id"])["long_term"], retained)

    async def test_partial_compaction_preserves_prior_summary_evidence_not_in_prompt(self):
        with (
            tempfile.TemporaryDirectory() as ws,
            mock.patch.object(compaction, "_previous_summary_budget", return_value=1_000),
        ):
            data = await asyncio.to_thread(session.create_session, ws)
            unseen = "Never discard this unresolved commitment."
            data["summary"] = "A" * 2_000 + unseen + "B" * 2_000
            data["messages"] = [{"role": "user", "content": "hello"} for _ in range(6)]
            await asyncio.to_thread(session.save_session, ws, data["id"], data)
            output = "[SUMMARY]" + "s" * 100 + "[/SUMMARY]"
            with (
                mock.patch.object(workspace, "get_compact_interval", return_value=1),
                mock.patch.object(compaction, "run_internal_backend", new=mock.AsyncMock(return_value=output)) as summarize,
                mock.patch.object(colony, "maybe_trigger"),
            ):
                await compaction.maybe_compact(ws, data["id"], backend_name="codex")
            prompt = summarize.await_args.args[2]
            saved = session.load_session(ws, data["id"])
            self.assertNotIn(unseen, prompt)
            self.assertIn(unseen, saved["summary"])
            self.assertTrue(saved["summary"].endswith("s" * 100))
            self.assertLess(len(saved["summary"]), len(data["summary"]))
            self.assertLess(len(prompt), compaction.MAX_SUMMARY_CHARS)

    async def test_compaction_retains_legacy_over_cap_memory_when_all_facts_are_omitted(self):
        with (
            tempfile.TemporaryDirectory() as ws,
            mock.patch.object(compaction, "MAX_SUMMARY_CHARS", 5_000),
        ):
            data = await asyncio.to_thread(session.create_session, ws)
            original = [f"Old fact {index}." for index in range(session.LONG_TERM_CAP + 1)]
            original[-1] = "x" * compaction.MAX_SUMMARY_CHARS
            data["long_term"] = original
            data["messages"] = [{"role": "user", "content": "hello"} for _ in range(6)]
            await asyncio.to_thread(session.save_session, ws, data["id"], data)
            output = (
                "[SUMMARY]" + "s" * 100 + "[/SUMMARY]"
                "[LONG_TERM]\n- New fact.\n[/LONG_TERM]"
            )
            with (
                mock.patch.object(workspace, "get_compact_interval", return_value=1),
                mock.patch.object(compaction, "run_internal_backend", new=mock.AsyncMock(return_value=output)),
                mock.patch.object(colony, "maybe_trigger"),
            ):
                await compaction.maybe_compact(ws, data["id"], backend_name="codex")
            saved = session.load_session(ws, data["id"])
            self.assertEqual(saved["long_term"], original)
            self.assertEqual(saved["summary"], "s" * 100)
            self.assertEqual(len(saved["messages"]), 5)

    async def test_repeated_partial_compaction_shrinks_summary_and_preserves_unseen_commitment(self):
        with tempfile.TemporaryDirectory() as ws:
            data = await asyncio.to_thread(session.create_session, ws)
            unseen = "Unresolved commitment retained through every pass."
            original = "A" * 3_500 + unseen + "B" * 3_500
            data["summary"] = original
            await asyncio.to_thread(session.save_session, ws, data["id"], data)
            output = "[SUMMARY]" + "s" * 100 + "[/SUMMARY]"
            lengths = [len(original)]
            with (
                mock.patch.object(compaction, "_previous_summary_budget", return_value=1_000),
                mock.patch.object(workspace, "get_compact_interval", return_value=1),
                mock.patch.object(compaction, "run_internal_backend", new=mock.AsyncMock(return_value=output)) as summarize,
                mock.patch.object(colony, "maybe_trigger"),
            ):
                for _ in range(3):
                    await asyncio.to_thread(session.append_messages, ws, data["id"], [{"role": "user", "content": "hello"} for _ in range(6)])
                    await compaction.maybe_compact(ws, data["id"], backend_name="codex")
                    saved = session.load_session(ws, data["id"])
                    self.assertIn(unseen, saved["summary"])
                    self.assertNotIn(unseen, summarize.await_args.args[2])
                    self.assertLess(len(saved["summary"]), lengths[-1])
                    lengths.append(len(saved["summary"]))
            self.assertEqual(colony.get_compact_count(ws), 3)

    async def test_retained_summary_cannot_make_too_short_generated_summary_acceptable(self):
        with tempfile.TemporaryDirectory() as ws:
            data = await asyncio.to_thread(session.create_session, ws)
            data["summary"] = "A" * 3_500
            data["messages"] = [{"role": "user", "content": "hello"} for _ in range(6)]
            await asyncio.to_thread(session.save_session, ws, data["id"], data)
            with (
                mock.patch.object(compaction, "_previous_summary_budget", return_value=1_000),
                mock.patch.object(workspace, "get_compact_interval", return_value=1),
                mock.patch.object(compaction, "run_internal_backend", new=mock.AsyncMock(return_value="[SUMMARY]short[/SUMMARY]")),
                self.assertLogs(compaction.logger, level="ERROR"),
            ):
                await compaction.maybe_compact(ws, data["id"], backend_name="codex")
            self.assertEqual(session.load_session(ws, data["id"]), data)
            self.assertEqual(colony.get_compact_count(ws), 0)

    def test_prior_summary_shown_and_omitted_text_reconstructs_exact_original_at_edge_budgets(self):
        marker = compaction._PREVIOUS_SUMMARY_TRUNCATION_MARKER
        original = "".join(chr(0x1_000 + index) for index in range(len(marker) + 50))
        for budget in (0, 1, len(marker), len(marker) + 1, len(marker) + 2, len(original)):
            with self.subTest(budget=budget), mock.patch.object(compaction, "_previous_summary_budget", return_value=budget):
                shown, omitted = compaction._previous_summary_with_omission(original)
            self.assertLessEqual(len(shown), budget)
            if marker in shown:
                head, tail = shown.split(marker)
                reconstructed = head + omitted + tail
            elif shown.endswith("…"):
                reconstructed = shown[:-1] + omitted
            else:
                reconstructed = shown + omitted
            self.assertEqual(reconstructed, original)


class SessionRecencyTests(unittest.TestCase):
    def test_one_missing_file_does_not_discard_other_session_recency(self):
        mtimes = {"old.json": 1, "new.json": 3, "middle.json": 2}

        def stat(path):
            name = os.path.basename(path)
            if name == "missing.json":
                raise FileNotFoundError(path)
            return SimpleNamespace(st_mtime_ns=mtimes[name])

        with mock.patch.object(session.os, "stat", side_effect=stat):
            self.assertEqual(
                session._newest_first("sessions", ["old.json", "missing.json", "new.json", "middle.json"], 2),
                ["new.json", "middle.json"],
            )
            self.assertEqual(
                session._newest_first("sessions", ["missing.json"], 2),
                ["missing.json"],
            )
