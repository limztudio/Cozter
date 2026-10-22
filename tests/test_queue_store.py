"""Queue acceptance and schedule receipts share a single durable transaction."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from Cozter.queue_store import AdmissionResult, QueueStore, ScheduledOccurrence


class QueueStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "queue_signal.sqlite3"
        self.legacy = self.root / "queue_signal.json"
        self.store = QueueStore(self.path)

    def occurrence(self, slot: str = "2026-10-17T09:00:00", owner: str = "owner", schedule: str = "daily") -> ScheduledOccurrence:
        return ScheduledOccurrence(str(self.root / "work"), owner, schedule, slot)

    def entry(self, occurrence: ScheduledOccurrence | None = None, entry_id: str = "ordinary") -> dict:
        entry = {"id": entry_id, "text": "Finish the work", "chat_id": "room"}
        if occurrence is not None:
            entry.update(id=occurrence.entry_id("signal"), ephemeral=True, job={
                "workspace_path": occurrence.workspace_path, "schedule_id": occurrence.schedule_id,
                "schedule_slot": occurrence.slot,
            })
        return entry

    def test_construction_does_no_io_and_empty_store_is_versioned(self) -> None:
        self.assertFalse(self.path.exists())
        self.assertEqual(self.store.read_entries(), {})
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)

    def test_new_admission_atomically_owns_entry_and_watermark(self) -> None:
        occurrence = self.occurrence()
        entry = self.entry(occurrence)
        self.assertEqual(self.store.admit("owner", entry, occurrence), AdmissionResult(entry, True))
        self.assertEqual(self.store.read_entries(), {"owner": [entry]})
        self.assertEqual(self.store.schedule_watermarks(), {occurrence.key(): occurrence.slot})

    def test_failed_queue_write_rolls_back_watermark_together(self) -> None:
        self.store.read_entries()
        occurrence = self.occurrence()
        with mock.patch.object(self.store, "_write_body", side_effect=OSError("disk failed")):
            with self.assertRaisesRegex(OSError, "disk failed"):
                self.store.admit("owner", self.entry(occurrence), occurrence)
        self.assertEqual(self.store.read_entries(), {})
        self.assertEqual(self.store.schedule_watermarks(), {})

    def test_failed_watermark_write_does_not_accept_entry(self) -> None:
        self.store.read_entries()
        occurrence = self.occurrence()
        with mock.patch.object(self.store, "_write_watermark", side_effect=OSError("receipt failed")):
            with self.assertRaisesRegex(OSError, "receipt failed"):
                self.store.admit("owner", self.entry(occurrence), occurrence)
        self.assertEqual(self.store.read_entries(), {})
        self.assertEqual(self.store.schedule_watermarks(), {})

    def test_retry_returns_existing_checkpoint_without_replacing_it(self) -> None:
        occurrence = self.occurrence()
        original = self.entry(occurrence)
        self.store.admit("owner", original, occurrence)
        body = self.store.read_entries()
        saved = body["owner"][0]
        saved["job"].update(request="User added guidance", state="running", session_id="unchanged-native-id")
        self.store.write_entries(body)
        result = self.store.admit("owner", original, occurrence)
        self.assertEqual(result, AdmissionResult(saved, False))
        self.assertEqual(self.store.read_entries(), body)

    def test_completed_and_cancelled_occurrences_never_reappear(self) -> None:
        occurrence = self.occurrence()
        entry = self.entry(occurrence)
        self.store.admit("owner", entry, occurrence)
        self.store.write_entries({})
        restarted = QueueStore(self.path)
        self.assertEqual(restarted.admit("owner", entry, occurrence), AdmissionResult(None, False, True))
        self.assertEqual(restarted.read_entries(), {})
        self.assertEqual(restarted.schedule_watermarks(), {occurrence.key(): occurrence.slot})

    def test_newer_watermark_suppresses_older_occurrence_without_regressing(self) -> None:
        newer = self.occurrence("2026-10-18T09:00:00")
        older = self.occurrence()
        self.store.admit("owner", self.entry(newer), newer)
        self.store.write_entries({})
        self.assertTrue(self.store.admit("owner", self.entry(older), older).already_consumed)
        self.assertEqual(self.store.schedule_watermarks(), {newer.key(): newer.slot})

    def test_distinct_workspaces_owners_and_schedules_have_independent_receipts(self) -> None:
        first = self.occurrence()
        other_owner = self.occurrence(owner="other")
        other_schedule = self.occurrence(schedule="weekly")
        other_workspace = ScheduledOccurrence(str(self.root / "another"), "owner", "daily", first.slot)
        occurrences = (first, other_owner, other_schedule, other_workspace)
        for occurrence in occurrences:
            self.assertTrue(self.store.admit(occurrence.owner_id, self.entry(occurrence), occurrence).created)
        self.assertEqual(len(self.store.schedule_watermarks()), 4)
        self.assertEqual(sum(len(rows) for rows in self.store.read_entries().values()), 4)

    def test_occurrence_identity_normalizes_paths_and_iso_slots(self) -> None:
        first = self.occurrence()
        alternate = ScheduledOccurrence(str(self.root / "work" / ".." / "work"), "owner", "daily", "2026-10-17T09:00")
        self.assertEqual(first, alternate)
        self.assertEqual(first.entry_id("signal"), alternate.entry_id("signal"))
        self.assertEqual(len(first.entry_id("signal")), 32)
        self.assertNotEqual(first.entry_id("signal"), first.entry_id("telegram"))

    def test_admission_rejects_identity_collisions_and_origin_mismatches(self) -> None:
        occurrence = self.occurrence()
        original = self.entry(occurrence)
        self.store.admit("owner", original, occurrence)
        changed = self.entry(occurrence)
        changed["text"] = "Different request"
        with self.assertRaisesRegex(ValueError, "collision"):
            self.store.admit("owner", changed, occurrence)
        wrong = self.entry(occurrence)
        wrong["job"]["workspace_path"] = str(self.root / "wrong")
        with self.assertRaisesRegex(ValueError, "occurrence"):
            self.store.admit("owner", wrong, occurrence)
        self.assertEqual(self.store.read_entries(), {"owner": [original]})

    def test_ordinary_admission_and_state_writes_keep_unknown_fields(self) -> None:
        entry = self.entry()
        entry["unknown"] = {"nested": [False, None, "한글"]}
        self.assertTrue(self.store.admit("owner", entry).created)
        self.assertFalse(self.store.admit("owner", entry).created)
        self.assertEqual(self.store.schedule_watermarks(), {})
        self.assertEqual(self.store.read_entries(), {"owner": [entry]})

    def test_json_sibling_blocks_initialization_without_reading_or_changing_it(self) -> None:
        for raw in (b'{"owner": [{"id": "saved-job"}]}', b"not JSON", b"[]", b'{"invalid": NaN}', b"\xff"):
            with self.subTest(raw=raw):
                self.legacy.write_bytes(raw)
                for _ in range(2):
                    with self.assertRaisesRegex(ValueError, "JSON queue storage is retired"):
                        self.store.read_entries()
                self.assertEqual(self.legacy.read_bytes(), raw)
                self.assertEqual(list(self.root.glob("*.migrated.*")), [])
                with closing(sqlite3.connect(self.path)) as connection:
                    self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 0)
                    self.assertEqual(connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [])

    def test_json_sibling_directory_blocks_initialization(self) -> None:
        self.legacy.mkdir()
        retained = self.legacy / "retained-source"
        retained.write_bytes(b"saved queue evidence")
        with self.assertRaisesRegex(ValueError, "JSON queue storage is retired"):
            self.store.read_entries()
        self.assertEqual(retained.read_bytes(), b"saved queue evidence")
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [])

    def test_broken_json_sibling_symlink_blocks_initialization(self) -> None:
        target = self.root / "missing-json-source"
        try:
            self.legacy.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("Creating symlinks is unavailable")
        with self.assertRaisesRegex(ValueError, "JSON queue storage is retired"):
            self.store.read_entries()
        self.assertTrue(self.legacy.is_symlink())
        self.assertFalse(target.exists())
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [])

    def test_unreadable_json_sibling_cannot_be_treated_as_absent(self) -> None:
        self.legacy.write_bytes(b"saved queue")
        with mock.patch.object(Path, "lstat", side_effect=PermissionError("source inaccessible")):
            with self.assertRaisesRegex(PermissionError, "source inaccessible"):
                self.store.read_entries()
        self.assertEqual(self.legacy.read_bytes(), b"saved queue")
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [])

    def test_version_one_keeps_checkpoints_and_receipts_without_archiving_old_sources(self) -> None:
        occurrence = self.occurrence()
        entry = self.entry(occurrence)
        entry["job"].update(state="running", request="exact saved request", session_id="saved-native-session")
        entry["job"]["checkpoint"] = {"reports": ["verified output"], "pending": ["remaining task"]}
        self.store.admit("owner", entry, occurrence)
        raw = b"retained obsolete source bytes"
        self.legacy.write_bytes(raw)
        backup = self.root / "queue_signal.json.migrated.saved"
        backup.write_bytes(raw)
        stat = self.legacy.stat()
        fingerprint = json.dumps([stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns])
        rows = [
            (str(self.legacy), fingerprint, str(backup), "retired", 0),
            (str(self.root / "archived-source.json"), "old", str(backup), "archived", 1),
        ]
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("CREATE TABLE queue_migration (source TEXT NOT NULL, fingerprint TEXT NOT NULL, backup TEXT NOT NULL, digest TEXT NOT NULL, archived INTEGER NOT NULL DEFAULT 0)")
            connection.executemany("INSERT INTO queue_migration VALUES (?, ?, ?, ?, ?)", rows)
            connection.commit()
        restarted = QueueStore(self.path)
        self.assertEqual(restarted.read_entries(), {"owner": [entry]})
        self.assertEqual(restarted.schedule_watermarks(), {occurrence.key(): occurrence.slot})
        self.assertEqual(restarted.admit("owner", self.entry(occurrence), occurrence), AdmissionResult(entry, False))
        self.assertEqual(self.legacy.read_bytes(), raw)
        self.assertEqual(backup.read_bytes(), raw)
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("SELECT * FROM queue_migration").fetchall(), rows)
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)

    def test_other_platform_json_and_archived_backups_are_not_adopted(self) -> None:
        obsolete = self.root / "queue_old_owner.json"
        backup = self.root / "queue_signal.json.migrated.saved"
        raw = b'{"old": [{"id": "unrelated-job"}]}'
        obsolete.write_bytes(raw)
        backup.write_bytes(raw)
        self.assertEqual(self.store.read_entries(), {})
        self.assertEqual(self.store.schedule_watermarks(), {})
        self.assertEqual(obsolete.read_bytes(), raw)
        self.assertEqual(backup.read_bytes(), raw)
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(set(connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()), {
                ("queue_state",), ("schedule_watermarks",),
            })

    def test_concurrent_initialization_admits_every_unique_job(self) -> None:
        def admit(index: int) -> AdmissionResult:
            occurrence = self.occurrence(schedule=f"schedule-{index}")
            return QueueStore(self.path).admit("owner", self.entry(occurrence), occurrence)

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(admit, range(24)))
        self.assertTrue(all(result.created for result in results))
        self.assertEqual(len(self.store.read_entries()["owner"]), 24)
        self.assertEqual(len(self.store.schedule_watermarks()), 24)
        self.assertEqual(list(self.root.glob("*.migrated.*")), [])

    def test_concurrent_json_guard_never_accepts_or_imports_jobs(self) -> None:
        raw = b'{"old": [{"id": "retained-native-job"}]}'
        self.legacy.write_bytes(raw)

        def admit(index: int) -> None:
            with self.assertRaisesRegex(ValueError, "JSON queue storage is retired"):
                QueueStore(self.path).admit("owner", self.entry(entry_id=f"job-{index}"))

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(admit, range(24)))
        self.assertEqual(self.legacy.read_bytes(), raw)
        self.assertEqual(list(self.root.glob("*.migrated.*")), [])
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [])

    def test_unknown_version_and_unversioned_schema_fail_closed(self) -> None:
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("PRAGMA user_version=99")
        with self.assertRaisesRegex(ValueError, "version: 99"):
            self.store.read_entries()
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("PRAGMA user_version=0")
            connection.execute("CREATE TABLE foreign_data (body TEXT)")
        with self.assertRaisesRegex(ValueError, "unknown schema"):
            self.store.read_entries()

    def test_corrupt_database_fails_closed_without_touching_json_sibling(self) -> None:
        self.path.write_bytes(b"This is not a SQLite database")
        self.legacy.write_text('{"owner": []}', encoding="utf-8")
        with self.assertRaises(sqlite3.DatabaseError):
            self.store.read_entries()
        self.assertTrue(self.legacy.exists())

    def test_concurrent_duplicate_occurrence_has_one_commit_owner(self) -> None:
        occurrence = self.occurrence()
        entry = self.entry(occurrence)

        def admit(_index: int) -> AdmissionResult:
            return QueueStore(self.path).admit("owner", entry, occurrence)

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(admit, range(24)))
        self.assertEqual(sum(result.created for result in results), 1)
        self.assertEqual(self.store.read_entries(), {"owner": [entry]})


if __name__ == "__main__":
    unittest.main()
