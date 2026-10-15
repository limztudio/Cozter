"""Memory maintenance preserves source facts and owns off-thread persistence."""

import asyncio
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from Cozter import colony, compaction, execution_state, session, titling, workspace


class ContinuationValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_numbers_and_deep_nesting_preserve_damaged_state(self):
        with tempfile.TemporaryDirectory() as ws:
            async with execution_state.execution_scope(ws, "job", "request"):
                record = await execution_state.record("plan")
                await record.save(evidence="keep")
                path = record.path
                valid = json.loads(path.read_bytes())
            for value in ("NaN", "Infinity", "-Infinity", "1e9999", "-1e9999",
                          "[" * 110 + "0" + "]" * 110,
                          "[" * 2000 + "0" + "]" * 2000):
                payload = json.dumps(valid).replace('"keep"', value).encode()
                with self.subTest(value=value[:30]):
                    path.write_bytes(payload)
                    async with execution_state.execution_scope(ws, "job", "request"):
                        with self.assertRaisesRegex(RuntimeError, "preserve and inspect"):
                            await execution_state.record("plan")
                    self.assertEqual(path.read_bytes(), payload)

    async def test_finite_numbers_still_roundtrip(self):
        with tempfile.TemporaryDirectory() as ws:
            async with execution_state.execution_scope(ws, "job", "request"):
                await (await execution_state.record("plan")).save(values=[1.25, 1e300])
            async with execution_state.execution_scope(ws, "job", "request"):
                self.assertEqual((await execution_state.record("plan")).get("values"),
                                 [1.25, 1e300])


class ColonyCapacityTests(unittest.IsolatedAsyncioTestCase):
    async def test_over_capacity_proposal_cannot_remove_promoted_source_fact(self):
        with tempfile.TemporaryDirectory() as ws:
            data = await asyncio.to_thread(session.create_session, ws)
            data["long_term"] = ["Promoted fact must survive."]
            await asyncio.to_thread(session.save_session, ws, data["id"], data)
            await asyncio.to_thread(colony.set_items, ws, ["Existing shared fact."])
            paths = [Path(colony._path(ws)), Path(ws, ".cozter", "sessions", data["id"] + ".json")]
            before = [p.read_bytes() for p in paths]
            proposed = ["Promoted fact must survive.", *[f"Shared {i}." for i in range(colony.COLONY_CAP)]]
            output = ("[COLONY]\n" + "\n".join("- " + item for item in proposed)
                      + f"\n[/COLONY]\n[SESSION:{data['id']}]\n[/SESSION]")
            with (mock.patch.object(colony, "run_internal_backend", new=mock.AsyncMock(return_value=output)),
                  self.assertLogs(colony.logger, level="WARNING")):
                self.assertFalse(await colony.consolidate(ws, backend_name="codex"))
            self.assertEqual([p.read_bytes() for p in paths], before)

    async def test_repeated_colony_items_are_deduplicated_before_capacity_check(self):
        with tempfile.TemporaryDirectory() as ws:
            data = await asyncio.to_thread(session.create_session, ws)
            data["long_term"] = ["Shared fact."]
            await asyncio.to_thread(session.save_session, ws, data["id"], data)
            output = ("[COLONY]\n" + "- Shared fact.\n" * (colony.COLONY_CAP + 1)
                      + f"[/COLONY]\n[SESSION:{data['id']}]\n[/SESSION]")
            with mock.patch.object(colony, "run_internal_backend", new=mock.AsyncMock(return_value=output)):
                self.assertTrue(await colony.consolidate(ws, backend_name="codex"))
            self.assertEqual(colony.get_items(ws), ["Shared fact."])
            self.assertEqual(session.load_session(ws, data["id"])["long_term"], [])


class MaintenancePersistenceOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def _check_cancelled_persistence(self, kind):
        entered = threading.Event()
        release = threading.Event()
        main_thread = threading.get_ident()
        worker_threads = []
        operation = session.set_summary if kind == "compact" else session.set_session_name

        def slow_save(*args, **kwargs):
            worker_threads.append(threading.get_ident())
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release persistence")
            return operation(*args, **kwargs)

        with tempfile.TemporaryDirectory() as ws:
            data = await asyncio.to_thread(session.create_session, ws)
            data["messages"] = [{"role": "assistant", "content": "reply"} for _ in range(6)]
            await asyncio.to_thread(session.save_session, ws, data["id"], data)
            with (mock.patch.object(workspace, "get_compact_interval", return_value=1),
                  mock.patch.object(compaction, "compact_session", new=mock.AsyncMock(return_value=("s" * 100, None, None, 6))),
                  mock.patch.object(titling, "generate", new=mock.AsyncMock(return_value="Saved Title")),
                  mock.patch.object(colony, "maybe_trigger"),
                  mock.patch.object(session, "set_summary" if kind == "compact" else "set_session_name", side_effect=slow_save)):
                action = compaction.maybe_compact if kind == "compact" else titling.maybe_auto_title
                task = asyncio.create_task(action(ws, data["id"], None, backend_name="codex"))
                try:
                    self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                    self.assertTrue(workspace.get_lock(ws).locked())
                    task.cancel()
                    await asyncio.sleep(0)
                    task.cancel()
                    await asyncio.sleep(0.02)
                    self.assertFalse(task.done())
                    self.assertTrue(workspace.get_lock(ws).locked())
                finally:
                    release.set()
                    await asyncio.gather(task, return_exceptions=True)
            self.assertTrue(task.cancelled())
            self.assertFalse(workspace.get_lock(ws).locked())
            self.assertNotEqual(worker_threads, [])
            self.assertNotIn(main_thread, worker_threads)
            saved = session.load_session(ws, data["id"])
            if kind == "compact":
                self.assertEqual(saved["summary"], "s" * 100)
            else:
                self.assertEqual(saved["name"], "Saved Title")

    async def test_compaction_persistence_is_responsive_and_owned_through_repeated_cancel(self):
        await self._check_cancelled_persistence("compact")

    async def test_title_persistence_is_responsive_and_owned_through_repeated_cancel(self):
        await self._check_cancelled_persistence("title")
