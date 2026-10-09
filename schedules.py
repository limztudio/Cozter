"""Workspace-level schedule store.

Schedules are stored in ``.cozter/schedules.json`` so a fired schedule
can run in its own ephemeral session without being tied to whichever
session happened to be current when the user created it.

File shape:
    {"<user_id>": [<schedule_dict>, ...], ...}

Each ``<schedule_dict>``:
    {id, days, time, command, created, chat_id, user_id, platform_id?, last_fired?}
"""

import copy
import logging
import uuid
from datetime import datetime, time as dt_time, timedelta

from . import workspace as workspace_mod
from .utils import load_json_object
from .utils import parse_decimal_int
from .utils import save_json_object
from .utils import stat_mtime_size
from .utils import try_parse_int

logger = logging.getLogger(__name__)

SCHEDULES_FILE = "schedules.json"

DAY_ABBREV: tuple[str, ...] = (
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
)


def _path(workspace: str) -> str:
    return workspace_mod.workspace_state_path(workspace, SCHEDULES_FILE)


_SCHEDULES_CACHE: dict[str, tuple[int | None, int | None, dict]] = {}


def _load_all(workspace: str) -> dict:
    """Return the schedule map for *workspace* (empty on failure).

    Cached by file mtime+size like the session/colony caches: the
    scheduler tick re-reads this file for every active (user, workspace)
    pair every 30s, and most ticks change nothing. Edits refresh
    mtime/size so the next read sees them.
    """
    path = _path(workspace)
    mtime_ns, size = stat_mtime_size(path)
    cached = _SCHEDULES_CACHE.get(path)
    if cached is not None and cached[0] == mtime_ns and cached[1] == size:
        return copy.deepcopy(cached[2])
    data = load_json_object(path, "schedules file", logger)
    _SCHEDULES_CACHE[path] = (mtime_ns, size, copy.deepcopy(data))
    return data


def _save_all(workspace: str, data: dict) -> None:
    """Persist the schedule map and refresh the read cache."""
    path = _path(workspace)
    save_json_object(path, data)
    mtime_ns, size = stat_mtime_size(path)
    if mtime_ns is None:
        _SCHEDULES_CACHE.pop(path, None)
        return
    _SCHEDULES_CACHE[path] = (mtime_ns, size, copy.deepcopy(data))


def _schedule_list(data: dict, user_id: str | int) -> list:
    schedules = data.get(str(user_id), [])
    return schedules if isinstance(schedules, list) else []


def add_schedule(
    workspace: str, user_id: str | int, schedule: dict,
) -> None:
    data = _load_all(workspace)
    key = str(user_id)
    schedules = _schedule_list(data, key)
    if isinstance(schedule, dict):
        raw_id = schedule.get("id")
        taken = {
            schedule.get("id")
            for schedule in schedules
            if isinstance(schedule, dict) and isinstance(schedule.get("id"), str)
        }
        # Duplicate ids strand a twin; mint fresh ids.
        if not isinstance(raw_id, str) or not raw_id or raw_id in taken:
            fresh = uuid.uuid4().hex[:12]
            while fresh in taken:
                fresh = uuid.uuid4().hex[:12]
            schedule["id"] = fresh
    schedules.append(schedule)
    data[key] = schedules
    _save_all(workspace, data)


def remove_schedule(
    workspace: str, user_id: str | int, schedule_id: str,
) -> bool:
    data = _load_all(workspace)
    key = str(user_id)
    schedules = _schedule_list(data, key)
    kept = [
        schedule for schedule in schedules
        if not (isinstance(schedule, dict) and schedule.get("id") == schedule_id)
    ]
    if len(kept) == len(schedules):
        return False
    if kept:
        data[key] = kept
    else:
        data.pop(key, None)
    _save_all(workspace, data)
    return True


def list_schedules(
    workspace: str, user_id: str | int,
) -> list[dict]:
    return [
        schedule for schedule in _schedule_list(_load_all(workspace), user_id)
        if isinstance(schedule, dict)
    ]


def list_schedule_user_ids(workspace: str) -> list[str]:
    """Return user keys that currently own schedules in *workspace*."""
    return [
        str(uid)
        for uid, entries in _load_all(workspace).items()
        if isinstance(entries, list)
    ]


def migrate_schedules(
    workspace: str,
    source_user_ids: list[str | int] | tuple[str | int, ...],
    target_user_id: str | int,
    *,
    source_chat_id: str = "",
    target_chat_id: str = "",
    source_platform_id: str = "",
) -> int:
    """Move legacy schedules to a new user key, returning moved count."""
    data = _load_all(workspace)
    target_key = str(target_user_id)
    target = data.get(target_key, [])
    if not isinstance(target, list):
        target = []
    # Persisted IDs may be hand-edited; never abort migration.
    seen_ids = {
        found_id
        for schedule in target
        if isinstance(schedule, dict)
        and isinstance((found_id := schedule.get("id")), str)
        and found_id
    }

    moved = 0
    changed = False
    for source_user_id in source_user_ids:
        source_key = str(source_user_id)
        if source_key == target_key:
            continue
        source = data.get(source_key)
        if not isinstance(source, list) or not source:
            continue

        remaining: list[dict] = []
        for sched in source:
            if not isinstance(sched, dict):
                remaining.append(sched)
                continue
            if source_platform_id and sched.get("platform_id") not in (
                None, "", source_platform_id,
            ):
                remaining.append(sched)
                continue
            sched_chat_id = str(sched.get("chat_id") or "")
            if source_chat_id and sched_chat_id != source_chat_id:
                remaining.append(sched)
                continue
            raw_sched_id = sched.get("id")
            sched_id = (
                raw_sched_id
                if isinstance(raw_sched_id, str) and raw_sched_id
                else None
            )
            if sched_id is not None and sched_id in seen_ids:
                changed = True
                continue

            migrated = dict(sched)
            migrated["user_id"] = target_key
            if source_platform_id:
                migrated["platform_id"] = source_platform_id
            if target_chat_id:
                migrated["chat_id"] = target_chat_id
            target.append(migrated)
            if sched_id is not None:
                seen_ids.add(sched_id)
            moved += 1
            changed = True

        if remaining:
            data[source_key] = remaining
        else:
            data.pop(source_key, None)

    if changed:
        if target:
            data[target_key] = target
        _save_all(workspace, data)
    return moved


def update_schedule_fired(
    workspace: str,
    user_id: str | int,
    schedule_id: str,
    fired_at: str,
) -> dict | None:
    """Persist a schedule's fired slot and return its current record.

    The scheduler snapshots due entries before it acquires the workspace
    lock.  A schedule can be removed in that gap, so a missing record is an
    unsuccessful claim rather than an unconditional fire of stale data.
    Returning a copy of the record we actually updated also makes callers
    queue the current command/chat metadata instead of their old snapshot.
    """
    data = _load_all(workspace)
    key = str(user_id)
    schedules = _schedule_list(data, key)
    claimed: dict | None = None
    for schedule in schedules:
        if not isinstance(schedule, dict):
            continue
        if schedule.get("id") == schedule_id:
            # Stamp twins: heal legacy duplicates.
            schedule["last_fired"] = fired_at
            if claimed is None:
                claimed = dict(schedule)
    if claimed is None:
        return None
    _save_all(workspace, data)
    return claimed


# Field parsers + time-slot computation

def parse_days(text: object) -> list[str]:
    """Parse a days spec into ordered, de-duplicated abbreviations.

    Accepts ``"all"``, comma-separated names (``"mon,wed,fri"``), or
    1-7 numbers (``"1,3,5"``). Returns ``[]`` on invalid input.
    """
    if not isinstance(text, str):
        return []
    text = text.strip().lower()
    if text == "all":
        return list(DAY_ABBREV)
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return []
    days: list[str] = []
    for part in parts:
        if part.isdecimal():
            day_number = parse_decimal_int(part)
            if day_number is None:
                return []
            if not (1 <= day_number <= 7):
                return []
            days.append(DAY_ABBREV[day_number - 1])
        else:
            abbr = part[:3]
            if abbr not in DAY_ABBREV:
                return []
            days.append(abbr)
    # dict.fromkeys dedups, preserving order.
    return list(dict.fromkeys(days))


def parse_time(text: object) -> str | None:
    """Parse ``"HH:MM"`` (24-hour) into ``"HH:MM"`` (zero-padded)."""
    if not isinstance(text, str):
        return None
    parts = text.split(":")
    if len(parts) != 2:
        return None
    hour = try_parse_int(parts[0])
    minute = try_parse_int(parts[1])
    if hour is None or minute is None:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return f"{hour:02d}:{minute:02d}"


def parse_iso(value: object) -> datetime | None:
    """Parse an ISO timestamp in the scheduler's local naive time basis.

    Cozter persists schedule timestamps with :meth:`datetime.isoformat` on
    naive local datetimes.  Hand-edited or migrated state can contain an
    offset-aware ISO timestamp, though.  Normalize that form to local naive
    time before the scheduler compares it with ``datetime.now()``; mixing the
    two directly raises ``TypeError`` and would abort the whole scheduler
    tick.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (OverflowError, ValueError):
        return None
    if parsed.tzinfo is not None and parsed.utcoffset() is not None:
        try:
            return parsed.astimezone().replace(tzinfo=None)
        except (OverflowError, OSError, ValueError):
            # Clamp edge timestamps.
            return None
    return parsed


def most_recent_slot(sched: dict, now: datetime) -> datetime | None:
    """Return the latest (day, time) match <= now, or None."""
    parsed = parse_time(sched.get("time", ""))
    if parsed is None:
        return None
    hour, minute = map(int, parsed.split(":"))
    target = dt_time(hour, minute)
    raw_days = sched.get("days", [])
    if not isinstance(raw_days, list):
        return None
    days = [
        day for day in raw_days
        if isinstance(day, str) and day in DAY_ABBREV
    ]
    if not days:
        return None
    # Walk back 7 days for the most recent match.
    for offset in range(8):
        candidate_date = (now - timedelta(days=offset)).date()
        day_name = DAY_ABBREV[candidate_date.weekday()]
        if day_name not in days:
            continue
        candidate = datetime.combine(candidate_date, target)
        if candidate <= now:
            return candidate
    return None
