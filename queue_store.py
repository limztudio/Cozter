"""Transactional ownership of a platform's queued work and schedule receipts.

The queue body keeps its existing owner-to-entries JSON representation. SQLite
owns both that body and the schedule high-water marks, so accepting a scheduled
occurrence has one commit point. Call these synchronous methods off the loop.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
from collections.abc import Iterator
import uuid

from . import schedules, workspace

_VERSION = 1
_BUSY_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class ScheduledOccurrence:
    workspace_path: str
    owner_id: str
    schedule_id: str
    slot: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (
            self.workspace_path, self.owner_id, self.schedule_id, self.slot,
        )):
            raise ValueError("Scheduled occurrences require an origin and slot")
        parsed = schedules.parse_iso(self.slot)
        if parsed is None:
            raise ValueError("Invalid scheduled occurrence slot")
        object.__setattr__(self, "workspace_path", workspace.canonicalize_workspace_path(self.workspace_path))
        object.__setattr__(self, "slot", parsed.isoformat())

    def key(self) -> str:
        return json.dumps([self.workspace_path, self.owner_id, self.schedule_id], separators=(",", ":"))

    def entry_id(self, platform_id: str) -> str:
        identity = [str(platform_id), self.workspace_path, self.owner_id, self.schedule_id, self.slot]
        return uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(identity, separators=(",", ":"))).hex


@dataclass(frozen=True)
class AdmissionResult:
    entry: dict | None
    created: bool
    already_consumed: bool = False


class QueueStore:
    """A per-platform store; construction itself performs no filesystem I/O."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = os.fspath(path)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_SECONDS, isolation_level=None)
        try:
            connection.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_SECONDS * 1000}")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            self._initialize(connection)
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self, connection: sqlite3.Connection) -> None:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version == _VERSION:
            return
        if version != 0:
            raise ValueError(f"Unsupported queue database version: {version}")
        if connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone() is not None:
            raise ValueError("Unversioned queue database contains an unknown schema")
        source = Path(self.path).with_suffix(".json")
        try:
            source.lstat()
        except FileNotFoundError:
            pass
        else:
            raise ValueError(f"JSON queue storage is retired; preserve {source} and restore its SQLite queue")
        connection.execute("CREATE TABLE queue_state (id INTEGER PRIMARY KEY CHECK (id = 1), body TEXT NOT NULL)")
        connection.execute("CREATE TABLE schedule_watermarks (occurrence_key TEXT PRIMARY KEY, slot TEXT NOT NULL)")
        self._write_body(connection, {})
        connection.execute(f"PRAGMA user_version={_VERSION}")

    @staticmethod
    def _encode(data: dict) -> str:
        if not isinstance(data, dict):
            raise ValueError("Queue entries must be an owner-to-entries object")
        return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    @staticmethod
    def _read_body(connection: sqlite3.Connection) -> dict:
        row = connection.execute("SELECT body FROM queue_state WHERE id = 1").fetchone()
        if row is None:
            raise ValueError("Queue database has no state row")
        body = json.loads(row[0])
        if not isinstance(body, dict):
            raise ValueError("Queue database body is not an object")
        return body

    @classmethod
    def _write_body(cls, connection: sqlite3.Connection, data: dict) -> None:
        connection.execute("INSERT INTO queue_state (id, body) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET body=excluded.body", (cls._encode(data),))

    @staticmethod
    def _write_watermark(connection: sqlite3.Connection, occurrence: ScheduledOccurrence) -> None:
        connection.execute("INSERT INTO schedule_watermarks (occurrence_key, slot) VALUES (?, ?) ON CONFLICT(occurrence_key) DO UPDATE SET slot=excluded.slot", (occurrence.key(), occurrence.slot))

    def read_entries(self) -> dict:
        with self._transaction() as connection:
            return self._read_body(connection)

    def write_entries(self, data: dict) -> None:
        with self._transaction() as connection:
            self._write_body(connection, data)

    def schedule_watermarks(self) -> dict[str, str]:
        with self._transaction() as connection:
            return dict(connection.execute("SELECT occurrence_key, slot FROM schedule_watermarks"))

    @staticmethod
    def _validate_origin(owner: str, entry: dict, occurrence: ScheduledOccurrence) -> None:
        job = entry.get("job")
        if (owner != occurrence.owner_id or entry.get("ephemeral") is not True
                or not isinstance(job, dict)
                or job.get("schedule_id") != occurrence.schedule_id
                or not isinstance(job.get("workspace_path"), str)
                or workspace.canonicalize_workspace_path(job["workspace_path"]) != occurrence.workspace_path
                or schedules.parse_iso(job.get("schedule_slot")) != schedules.parse_iso(occurrence.slot)):
            raise ValueError("Scheduled queue entry does not match its occurrence")

    def admit(self, owner: str, entry: dict, occurrence: ScheduledOccurrence | None = None) -> AdmissionResult:
        if (not isinstance(owner, str) or not owner or not isinstance(entry, dict)
                or not isinstance(entry.get("id"), str) or not entry["id"]
                or not isinstance(entry.get("text"), str) or not isinstance(entry.get("chat_id"), str)):
            raise ValueError("Invalid queue admission")
        if occurrence is not None:
            self._validate_origin(owner, entry, occurrence)
        with self._transaction() as connection:
            body = self._read_body(connection)
            entries = body.get(owner, [])
            if not isinstance(entries, list):
                raise ValueError("Queue owner does not contain an entries list")
            matches = [candidate for candidate in entries if isinstance(candidate, dict) and candidate.get("id") == entry["id"]]
            if len(matches) > 1:
                raise ValueError("Duplicate queue admission identity")
            if matches:
                existing = matches[0]
                if any(existing.get(key) != entry.get(key) for key in ("text", "chat_id", "ephemeral")):
                    raise ValueError("Queue admission identity collision")
                if occurrence is not None:
                    self._validate_origin(owner, existing, occurrence)
                result = AdmissionResult(existing, False)
            else:
                result = AdmissionResult(entry, True)
            if occurrence is not None:
                row = connection.execute("SELECT slot FROM schedule_watermarks WHERE occurrence_key = ?", (occurrence.key(),)).fetchone()
                previous = schedules.parse_iso(row[0]) if row is not None else None
                if row is not None and previous is None:
                    raise ValueError("Invalid persisted schedule watermark")
                current = schedules.parse_iso(occurrence.slot)
                assert current is not None
                if previous is not None and previous >= current and not matches:
                    return AdmissionResult(None, False, already_consumed=True)
                if previous is None or previous < current:
                    self._write_watermark(connection, occurrence)
            if result.created:
                entries.append(entry)
                body[owner] = entries
                self._write_body(connection, body)
            return result
