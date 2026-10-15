"""Memory discovery pages retain ordering and counts without all transcripts."""

import json
from pathlib import Path
import re
import tempfile
import tracemalloc
import unittest
from unittest import mock

from Cozter.agent_tools.plugins import memory


def _write_session(directory: Path, session_id: str, *, created: str = "2026-10-15", name: str = "Fixture", summary: str = "", long_term: list[str] | None = None, messages: list[str] | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    Path(directory, f"{session_id}.json").write_text(json.dumps({
        "id": session_id, "name": name, "created": created,
        "summary": summary, "long_term": long_term or [],
        "messages": [{"role": "assistant", "content": text} for text in messages or []],
    }), encoding="utf-8")


class StreamedMemorySearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_small_results_keep_legacy_format_and_unicode_excerpts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp, ".cozter", "sessions")
            _write_session(directory, "aaaaaaaa1111", name="Unicode", summary="Straße", messages=["STRASSE remembered"])
            result = await memory.MemoryTool().run(tmp, {"action": "search", "query": "strasse"})
            self.assertEqual(result, (
                "Found 2 match(es) for 'strasse':\n"
                "- [Unicode · 2026-10-15] Summary: Straße\n"
                "- [Unicode · 2026-10-15] Assistant: STRASSE remembered"
            ))

    async def test_created_order_filename_ties_section_order_and_colony_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp, ".cozter", "sessions")
            for session_id in ("zzzzzzzz1111", "bbbbbbbb1111", "aaaaaaaa1111"):
                _write_session(directory, session_id, name=session_id[0], summary="needle summary", long_term=["needle memory"], messages=["needle message"])
            _write_session(directory, "newer1111111", created="2027-01-01", name="newer", summary="needle latest")
            Path(tmp, ".cozter", "colony.json").write_text(json.dumps({"items": ["needle colony"]}), encoding="utf-8")
            result = await memory.MemoryTool().run(tmp, {"action": "search", "query": "needle", "limit": 20})
            self.assertIn("Found 11 match(es)", result)
            rows = [line for line in result.splitlines() if line.startswith("- ")]
            self.assertEqual(len(rows), 11)
            self.assertIn("[newer ·", rows[0])
            for start, name in ((1, "a"), (4, "b"), (7, "z")):
                self.assertIn(f"[{name} ·", rows[start])
                self.assertIn("Summary:", rows[start])
                self.assertIn("Long-term:", rows[start + 1])
                self.assertIn("Assistant:", rows[start + 2])
            self.assertTrue(rows[-1].startswith("- [Colony]"))

    async def test_search_pages_recover_all_matches_after_character_trimming(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp, ".cozter", "sessions")
            for index in range(4):
                _write_session(directory, f"{index:08x}aaaa", name="Session " + "n" * 80, messages=[
                    f"needle item-{index * 15 + number:03d} " + "x" * 220 for number in range(15)
                ])
            Path(tmp, ".cozter", "colony.json").write_text(json.dumps({"items": ["needle item-060"]}), encoding="utf-8")
            offset = 0
            seen: list[int] = []
            for _ in range(20):
                result = await memory.MemoryTool().run(tmp, {"action": "search", "query": "needle", "offset": offset, "limit": 20})
                self.assertLessEqual(len(result), memory._OUTPUT_BUDGET)
                self.assertIn("Found 61 match(es)", result)
                indices = [int(value) for value in re.findall(r"item-(\d+)", result)]
                self.assertTrue(indices, result)
                self.assertFalse(set(indices).intersection(seen))
                seen.extend(indices)
                cursor = re.search(r"\boffset=(\d+)\b", result)
                if cursor is None:
                    break
                self.assertIn("PARTIAL + remainder", result)
                self.assertEqual(int(cursor.group(1)), offset + len(indices))
                offset = int(cursor.group(1))
            else:
                self.fail("Search did not reach the final matching colony item")
            self.assertEqual(seen, list(range(61)))
            past = await memory.MemoryTool().run(tmp, {"action": "search", "query": "needle", "offset": 61})
            self.assertIn("No more memory matches", past)
            self.assertNotIn("offset=", past)

    async def test_search_peak_memory_is_bounded_by_individual_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp, ".cozter", "sessions")
            for index in range(40):
                _write_session(directory, f"{index:08x}aaaa", messages=[f"needle {index} " + "x" * (128 * 1024)])
            tracemalloc.start()
            try:
                result = await memory.MemoryTool().run(tmp, {"action": "search", "query": "needle", "limit": 1})
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            self.assertIn("Found 40 match(es)", result)
            # The fixture contains 5 MiB of message text. Eager loading kept
            # all of it; streaming plus one selected reread stays well below it.
            self.assertLess(peak, 2 * 1024 * 1024)

    async def test_final_search_page_has_no_artificial_remainder_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp, ".cozter", "sessions")
            _write_session(directory, "aaaaaaaa1111", messages=[f"needle item-{number:03d}" for number in range(21)])
            result = await memory.MemoryTool().run(tmp, {
                "action": "search", "query": "needle", "offset": 20, "limit": 20,
            })
            self.assertIn("showing 1 match(es) after offset 20", result)
            self.assertIn("needle item-020", result)
            self.assertNotIn("offset=", result)
            self.assertNotIn("PARTIAL", result)
            self.assertNotIn("remainder", result)

    async def test_selected_file_changed_after_counting_requires_fresh_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp, ".cozter", "sessions")
            _write_session(directory, "aaaaaaaa1111", summary="needle original")
            original = memory._load_session_metadata

            def change_after_counting(*args):
                metadata = original(*args)
                _write_session(directory, "aaaaaaaa1111", summary="needle different and longer")
                return metadata

            with mock.patch.object(memory, "_load_session_metadata", side_effect=change_after_counting):
                result = await memory.MemoryTool().run(tmp, {"action": "search", "query": "needle"})
            self.assertTrue(result.startswith("Error: memory changed"), result)
            self.assertNotIn("Found 1 match", result)


class StreamedMemoryListTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_pages_recover_older_sessions_after_budget_trimming(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp, ".cozter", "sessions")
            for index in reversed(range(65)):
                _write_session(directory, f"{index:08x}aaaa", name=f"Session-{index:03d} " + "n" * 90)
            offset = 0
            seen: list[int] = []
            for _ in range(10):
                result = await memory.MemoryTool().run(tmp, {"action": "list", "offset": offset})
                self.assertLessEqual(len(result), memory._OUTPUT_BUDGET)
                indices = [int(value) for value in re.findall(r"^- Session-(\d+)", result, re.MULTILINE)]
                self.assertTrue(indices, result)
                self.assertFalse(set(indices).intersection(seen))
                seen.extend(indices)
                cursor = re.search(r"\boffset=(\d+)\b", result)
                if cursor is None:
                    break
                self.assertEqual(int(cursor.group(1)), offset + len(indices))
                offset = int(cursor.group(1))
            else:
                self.fail("List did not reach all older sessions")
            self.assertEqual(seen, list(range(65)))
            past = await memory.MemoryTool().run(tmp, {"action": "list", "offset": 65})
            self.assertIn("No older sessions", past)
            self.assertNotIn("offset=", past)

    async def test_complete_small_list_preserves_legacy_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp, ".cozter", "sessions")
            _write_session(directory, "aaaaaaaa1111", name="Fixture", summary="Summary", long_term=["Fact"], messages=["saved"])
            result = await memory.MemoryTool().run(tmp, {"action": "list"})
            self.assertEqual(result, (
                "Sessions (newest first):\n"
                "- Fixture — id aaaaaaaa, 2026-10-15, 1 messages, summary, 1 long-term\n"
                "Colony: 0 items in .cozter/colony.json"
            ))

    async def test_invalid_offsets_do_not_load_transcripts(self) -> None:
        for action in ("list", "search"):
            for offset in (None, True, -1, 1.25, "1"):
                with self.subTest(action=action, offset=offset), mock.patch.object(
                    memory, "_load_session_metadata", side_effect=AssertionError("must validate before reading"),
                ):
                    result = await memory.MemoryTool().run(".", {"action": action, "query": "needle", "offset": offset})
                self.assertTrue(result.startswith("Error:"), result)

    async def test_single_oversized_result_does_not_advertise_stalled_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp, ".cozter", "sessions")
            _write_session(directory, "aaaaaaaa1111", name="x" * 4_000, summary="needle")
            for action in ("list", "search"):
                result = await memory.MemoryTool().run(tmp, {"action": action, "query": "needle"})
                self.assertTrue(result.startswith("Error:"), result[:100])
                self.assertNotIn("offset=", result)
                self.assertLessEqual(len(result), memory._OUTPUT_BUDGET)

    async def test_named_read_fallback_avoids_aggregate_transcript_retention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp, ".cozter", "sessions")
            for index in range(40):
                _write_session(directory, f"{index:08x}aaaa", name=f"Name {index}", messages=["x" * (128 * 1024)])
            tracemalloc.start()
            try:
                result = await memory.MemoryTool().run(tmp, {"action": "read", "session": "Name 39"})
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            self.assertIn("Session: Name 39", result)
            self.assertLess(peak, 2 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
