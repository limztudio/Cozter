"""Workspace-level schedule store.

Schedules are stored in ``.cozter/schedules.json`` so a fired schedule
can run in its own ephemeral session without being tied to whichever
session happened to be current when the user created it.

File shape:
    {"<user_id>": [<schedule_dict>, ...], ...}

Each ``<schedule_dict>``:
    {id, days, time, command, created, chat_id, user_id, last_fired?}
"""

import logging
import os
from datetime import datetime, time as dt_time, timedelta

from .utils import COZTER_DIR
from .utils import load_json_object
from .utils import save_json_object

logger = logging.getLogger(__name__)

SCHEDULES_FILE = "schedules.json"

DAY_ABBREV: tuple[str, ...] = (
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
)


def _path(workspace: str) -> str:
    return os.path.join(workspace, COZTER_DIR, SCHEDULES_FILE)


def _load_all(workspace: str) -> dict:
    return load_json_object(_path(workspace), "schedules file", logger)


def _save_all(workspace: str, data: dict) -> None:
    save_json_object(_path(workspace), data)


def _schedule_list(data: dict, user_id: str | int) -> list:
    schedules = data.get(str(user_id), [])
    return schedules if isinstance(schedules, list) else []


def add_schedule(
    workspace: str, user_id: str | int, schedule: dict,
) -> None:
    data = _load_all(workspace)
    key = str(user_id)
    schedules = _schedule_list(data, key)
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
        s for s in schedules
        if not (isinstance(s, dict) and s.get("id") == schedule_id)
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
        s for s in _schedule_list(_load_all(workspace), user_id)
        if isinstance(s, dict)
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
) -> int:
    """Move legacy schedules to a new user key, returning moved count."""
    data = _load_all(workspace)
    target_key = str(target_user_id)
    target = data.get(target_key, [])
    if not isinstance(target, list):
        target = []
    seen_ids = {
        s.get("id") for s in target
        if isinstance(s, dict) and s.get("id")
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
            sched_chat_id = str(sched.get("chat_id") or "")
            if source_chat_id and sched_chat_id != source_chat_id:
                remaining.append(sched)
                continue
            sched_id = sched.get("id")
            if sched_id and sched_id in seen_ids:
                changed = True
                continue

            migrated = dict(sched)
            migrated["user_id"] = target_key
            if target_chat_id:
                migrated["chat_id"] = target_chat_id
            target.append(migrated)
            if sched_id:
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
) -> None:
    data = _load_all(workspace)
    key = str(user_id)
    schedules = _schedule_list(data, key)
    changed = False
    for s in schedules:
        if not isinstance(s, dict):
            continue
        if s.get("id") == schedule_id:
            s["last_fired"] = fired_at
            changed = True
            break
    if changed:
        _save_all(workspace, data)


# ---------------------------------------------------------------------------
# Schedule field parsers and time-slot computation. Pure functions used by
# both the user-input wizard (``cmd_reserve``) and the scheduler tick.
# ---------------------------------------------------------------------------

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
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return []
    days: list[str] = []
    for p in parts:
        if p.isdigit():
            n = int(p)
            if not (1 <= n <= 7):
                return []
            days.append(DAY_ABBREV[n - 1])
        else:
            abbr = p[:3]
            if abbr not in DAY_ABBREV:
                return []
            days.append(abbr)
    # dict.fromkeys dedups while preserving insertion order (Python 3.7+).
    return list(dict.fromkeys(days))


def parse_time(text: object) -> str | None:
    """Parse ``"HH:MM"`` (24-hour) into ``"HH:MM"`` (zero-padded)."""
    if not isinstance(text, str):
        return None
    parts = text.split(":")
    if len(parts) != 2:
        return None
    try:
        h = int(parts[0])
        m = int(parts[1])
    except ValueError:
        return None
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return None
    return f"{h:02d}:{m:02d}"


def parse_iso(value: object) -> datetime | None:
    """Parse an ISO timestamp; return None for missing or malformed input."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def most_recent_slot(sched: dict, now: datetime) -> datetime | None:
    """Return the latest (day, time) match <= now, or None."""
    parsed = parse_time(sched.get("time", ""))
    if parsed is None:
        return None
    h, m = map(int, parsed.split(":"))
    target = dt_time(h, m)
    raw_days = sched.get("days", [])
    if not isinstance(raw_days, list):
        return None
    days = [
        day for day in raw_days
        if isinstance(day, str) and day in DAY_ABBREV
    ]
    if not days:
        return None
    # Walk back up to 7 days; the first day-match whose datetime
    # is <= now is the most recent slot.
    for offset in range(8):
        candidate_date = (now - timedelta(days=offset)).date()
        day_name = DAY_ABBREV[candidate_date.weekday()]
        if day_name not in days:
            continue
        candidate = datetime.combine(candidate_date, target)
        if candidate <= now:
            return candidate
    return None
