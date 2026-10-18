"""Workspace-level schedule store.

Schedules are stored in ``.cozter/schedules.json`` so a fired schedule
can run in its own ephemeral session without being tied to whichever
session happened to be current when the user created it.

File shape:
    {"<user_id>": [<schedule_dict>, ...], ...}

Each ``<schedule_dict>``:
    {id, days, time, command, created, chat_id, user_id, platform_id, last_fired?}
"""

import copy
import logging
import uuid
from datetime import datetime, time as dt_time
from typing import TypeAlias

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
ScheduleDay: TypeAlias = str | int
_DAY_NAMES = dict(zip(
    ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"),
    DAY_ABBREV,
))


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
            schedule["last_fired"] = fired_at
            if claimed is None:
                claimed = dict(schedule)
    if claimed is None:
        return None
    _save_all(workspace, data)
    return claimed


# Field parsers + time-slot computation

def _parse_year_day(value: str) -> str | None:
    parts = value.split("/")
    if len(parts) != 2 or any(
        not (1 <= len(part) <= 2 and part.isdecimal()) for part in parts
    ):
        return None
    month, day = (parse_decimal_int(part) for part in parts)
    if month is None or day is None:
        return None
    try:
        # A leap reference year admits February 29 without admitting dates
        # that can never occur, such as February 30 or April 31.
        datetime(2000, month, day)
    except ValueError:
        return None
    return f"{month:02d}/{day:02d}"


def _parse_day(value: object) -> ScheduleDay | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1 <= value <= 31 else None
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    if value.isdecimal():
        number = parse_decimal_int(value)
        return number if number is not None and 1 <= number <= 31 else None
    if "/" in value:
        return _parse_year_day(value)
    if value in DAY_ABBREV:
        return value
    return _DAY_NAMES.get(value)


def normalize_days(value: object) -> list[ScheduleDay]:
    """Return valid weekly, monthly, and yearly rules, in input order.

    Integers and decimal strings mean dates of every month; month/day
    strings mean yearly dates. Invalid entries are ignored so a damaged
    entry does not hide valid occurrences.
    """
    if not isinstance(value, list):
        return []
    days: list[ScheduleDay] = []
    for raw_day in value:
        day = _parse_day(raw_day)
        if day is not None and day not in days:
            days.append(day)
    return days


def parse_days(text: object) -> list[ScheduleDay]:
    """Parse comma-separated weekdays, monthly dates, and yearly dates.

    ``"mon,wed"`` runs weekly, ``"1,15"`` runs on those dates every month,
    and ``"12/25"`` runs yearly. Rules can be combined. ``"all"`` runs
    daily. A malformed token rejects the entire input rather than creating
    a different schedule.
    """
    if not isinstance(text, str):
        return []
    text = text.strip().lower()
    if text == "all":
        return list(DAY_ABBREV)
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return []
    days: list[ScheduleDay] = []
    for part in parts:
        day = _parse_day(part)
        if day is None:
            return []
        days.append(day)
    # dict.fromkeys dedups, preserving order.
    return list(dict.fromkeys(days))


def format_days(value: object) -> str:
    """Describe persisted schedule days without confusing dates and weekdays."""
    days = normalize_days(value)
    weekdays = [day for day in days if isinstance(day, str) and day in DAY_ABBREV]
    monthdays = [str(day) for day in days if isinstance(day, int)]
    yeardays = [day for day in days if isinstance(day, str) and day not in DAY_ABBREV]
    if len(weekdays) == len(DAY_ABBREV) and not monthdays and not yeardays:
        return "Daily"
    labels = []
    if weekdays:
        labels.append(f"Weekly: {', '.join(weekdays)}")
    if monthdays:
        labels.append(f"Monthly: {', '.join(monthdays)}")
    if yeardays:
        labels.append(f"Yearly: {', '.join(yeardays)}")
    return "; ".join(labels) or "?"


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
            return None
    return parsed


def most_recent_slot(sched: dict, now: datetime) -> datetime | None:
    """Return the latest weekly, monthly, or yearly occurrence <= *now*.

    Dates absent from a month or year are skipped. Looking back across
    calendar boundaries preserves catch-up after a missed real occurrence.
    """
    parsed = parse_time(sched.get("time", ""))
    if parsed is None:
        return None
    hour, minute = map(int, parsed.split(":"))
    target = dt_time(hour, minute)
    days = normalize_days(sched.get("days", []))
    if not days:
        return None
    latest: datetime | None = None
    for day in days:
        candidate: datetime | None
        if isinstance(day, str) and day in DAY_ABBREV:
            offset = (now.weekday() - DAY_ABBREV.index(day)) % 7
            ordinal = now.toordinal() - offset
            if ordinal < 1:
                continue
            candidate = datetime.combine(datetime.fromordinal(ordinal), target)
            if candidate > now:
                ordinal -= 7
                if ordinal < 1:
                    continue
                candidate = datetime.combine(datetime.fromordinal(ordinal), target)
        elif isinstance(day, int):
            year, month = now.year, now.month
            while True:
                try:
                    candidate = datetime(year, month, day, hour, minute)
                except ValueError:
                    candidate = None
                if candidate is not None and candidate <= now:
                    break
                if month > 1:
                    month -= 1
                elif year > 1:
                    year -= 1
                    month = 12
                else:
                    break
            if candidate is None or candidate > now:
                continue
        else:
            month, monthday = map(int, day.split("/"))
            year = now.year
            while True:
                try:
                    candidate = datetime(year, month, monthday, hour, minute)
                except ValueError:
                    candidate = None
                if candidate is not None and candidate <= now:
                    break
                if year == 1:
                    break
                year -= 1
            if candidate is None or candidate > now:
                continue
        if latest is None or candidate > latest:
            latest = candidate
    return latest
