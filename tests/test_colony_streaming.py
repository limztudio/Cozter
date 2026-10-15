"""Colony maintenance retains memory projections and owns background writes."""

import asyncio
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import threading
import tracemalloc
import unittest
from unittest import mock

from Cozter import colony, session, workspace
from Cozter.utils import parse_bullets


class SessionMemoryProjectionTests(unittest.TestCase):
    def test_projection_matches_full_loader_normalization_and_tie_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp, ".cozter", "sessions")
            directory.mkdir(parents=True)
            for sid in ("a", "b", "c"):
                Path(directory, f"{sid}.json").write_text(json.dumps({
                    "id": sid, "name": 3, "created": "2027-01-01" if sid == "c" else "2026-01-01",
                    "long_term": ["Fact", 1, "", "  spaced fact  "],
                    "messages": [{"role": "assistant", "content": "private fixture transcript"}],
                }), encoding="utf-8")
            Path(directory, "corrupt.json").write_text("{", encoding="utf-8")
            Path(directory, "mismatch.json").write_text('{"id":"wrong"}', encoding="utf-8")
            order = ["b.json", "corrupt.json", "a.json", "mismatch.json", "c.json"]
            with mock.patch.object(session.os, "listdir", return_value=order):
                full = session.list_sessions_with_data(tmp)
                projected = session.list_session_memory(tmp)
            self.assertEqual(projected, [
                {key: data[key] for key in ("id", "name", "created", "long_term")}
                for data in full
            ])
            self.assertEqual([data["id"] for data in projected], ["c", "b", "a"])
            self.assertTrue(all("messages" not in data for data in projected))

    def test_projection_does_not_retain_every_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp, ".cozter", "sessions")
            directory.mkdir(parents=True)
            for number in range(40):
                sid = f"{number:012x}"
                Path(directory, f"{sid}.json").write_text(json.dumps({
                    "id": sid, "created": f"{number:04d}", "long_term": ["Retained fact"],
                    "messages": [{"role": "assistant", "content": "x" * (128 * 1024)}],
                }), encoding="utf-8")
            tracemalloc.start()
            try:
                projected = session.list_session_memory(tmp)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            self.assertEqual(len(projected), 40)
            self.assertLess(peak, 2 * 1024 * 1024)


class ColonyOutputParserTests(unittest.TestCase):
    def test_forward_parser_matches_original_nested_duplicate_and_malformed_semantics(self) -> None:
        original = re.compile(r"\[SESSION:([^\]\s]+)\](.*?)\[/SESSION\]", re.DOTALL)
        rng = random.Random(17)
        pieces = ["[SESSION:a]", "[SESSION:b]", "[/SESSION]", "[SESSION:bad id]", "\n- Fact\n", "text", "[SESSION:"]
        fixtures = [
            "[SESSION:a]\n- outer\n[SESSION:b]\n- inner\n[/SESSION][/SESSION]",
            "[SESSION:a]\n- first\n[/SESSION][SESSION:a]\n- last\n[/SESSION]",
            "[SESSION:x" * 50,
            "[SESSION:x" * 50 + "\n ]",
            "[SESSION:x" * 50 + "\u2003 ][SESSION:valid]\n- Fact\n[/SESSION]",
            *["".join(rng.choices(pieces, k=40)) for _ in range(150)],
        ]
        for value in fixtures:
            expected = {match.group(1).strip(): parse_bullets(match.group(2)) for match in original.finditer(value)}
            self.assertEqual(colony._parse_consolidate_output(value)[1], expected)

    def test_many_unclosed_blocks_finish_in_bounded_child(self) -> None:
        code = (
            "from Cozter import colony; "
            "assert colony._parse_consolidate_output('[SESSION:fixture]\\n' * 30000) == (None, {}); "
            "assert colony._parse_consolidate_output('[SESSION:x' * 30000) == (None, {}); "
            "assert colony._parse_consolidate_output('[SESSION:x' * 30000 + '\\n ]') == (None, {})"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class ColonyWriterOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_unchanged_shared_and_local_lists_do_not_rewrite_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = await asyncio.to_thread(session.create_session, tmp)
            data["long_term"] = ["Original local fact"]
            await asyncio.to_thread(session.save_session, tmp, data["id"], data)
            await asyncio.to_thread(colony.set_items, tmp, ["Original shared fact"])
            output = f"[COLONY]\n- Original shared fact\n[/COLONY][SESSION:{data['id']}]\n- Original local fact\n[/SESSION]"
            with mock.patch.object(colony, "run_internal_backend", new=mock.AsyncMock(return_value=output)), mock.patch.object(
                colony, "set_items", side_effect=AssertionError("unchanged shared memory must not rewrite"),
            ), mock.patch.object(session, "save_session", side_effect=AssertionError("unchanged local memory must not rewrite")), mock.patch.object(
                session, "load_session", side_effect=AssertionError("unchanged local memory needs no full transcript reload"),
            ):
                self.assertTrue(await colony.consolidate(tmp, backend_name="codex"))

    async def test_cancelled_apply_holds_writer_lock_until_real_save_settles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = await asyncio.to_thread(session.create_session, tmp)
            data["messages"] = [{"role": "user", "content": "Preserve this message"}]
            data["long_term"] = ["Original local fact"]
            await asyncio.to_thread(session.save_session, tmp, data["id"], data)
            output = f"[COLONY]\n- Shared fact\n[/COLONY][SESSION:{data['id']}]\n- Remaining local fact\n[/SESSION]"
            entered, release = threading.Event(), threading.Event()
            save = session.save_session

            def held_save(*args):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("test did not release colony save")
                return save(*args)

            with mock.patch.object(colony, "run_internal_backend", new=mock.AsyncMock(return_value=output)), mock.patch.object(
                session, "save_session", side_effect=held_save,
            ):
                task = asyncio.create_task(colony.consolidate(tmp, backend_name="codex"))
                contender = None
                try:
                    self.assertTrue(await asyncio.wait_for(asyncio.to_thread(entered.wait, 2), 3))
                    writer = workspace.get_lock(tmp)
                    self.assertTrue(writer.locked())
                    task.cancel()
                    await asyncio.sleep(0)
                    task.cancel()
                    await asyncio.sleep(0.02)
                    self.assertFalse(task.done())
                    self.assertTrue(writer.locked())
                    acquired = asyncio.Event()

                    async def next_writer():
                        async with writer:
                            acquired.set()

                    contender = asyncio.create_task(next_writer())
                    await asyncio.sleep(0)
                    self.assertFalse(acquired.is_set())
                    release.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(task, 3)
                    await asyncio.wait_for(contender, 3)
                    self.assertTrue(acquired.is_set())
                finally:
                    release.set()
                    await asyncio.gather(task, *([contender] if contender else []), return_exceptions=True)
            latest = session.load_session(tmp, data["id"])
            self.assertEqual(latest["long_term"], ["Remaining local fact"])
            self.assertEqual(latest["messages"], data["messages"])

    async def test_empty_workspace_clear_keeps_writer_lock_through_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            await asyncio.to_thread(colony.set_items, tmp, ["Obsolete shared fact"])
            entered, release = threading.Event(), threading.Event()
            save = colony.set_items

            def held_clear(*args):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("test did not release colony clear")
                return save(*args)

            with mock.patch.object(colony, "set_items", side_effect=held_clear):
                task = asyncio.create_task(colony.consolidate(tmp, backend_name="codex"))
                try:
                    self.assertTrue(await asyncio.wait_for(asyncio.to_thread(entered.wait, 2), 3))
                    task.cancel()
                    await asyncio.sleep(0.02)
                    self.assertFalse(task.done())
                    self.assertTrue(workspace.get_lock(tmp).locked())
                    release.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(task, 3)
                finally:
                    release.set()
                    await asyncio.gather(task, return_exceptions=True)
            self.assertEqual(colony.get_items(tmp), [])


if __name__ == "__main__":
    unittest.main()
