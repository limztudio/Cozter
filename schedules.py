"""Workspace-level schedule store.

Schedules used to be embedded inside each session's JSON. They are now
managed independently in ``.cozter/schedules.json`` so a fired schedule
can run in its own ephemeral session without being tied to whichever
session happened to be current at the time the user created it.

File shape:
    {"<user_id>": [<schedule_dict>, ...], ...}

Each ``<schedule_dict>`` matches the previous embedded format:
    {id, days, time, command, created, chat_id, user_id, last_fired?}

``migrate_from_sessions`` lifts any pre-existing embedded schedules out
of session files into this store on startup, so users don't lose
schedules they created before this change.
"""

import json
import logging
import os

from .utils import atomic_write as _atomic_write

logger = logging.getLogger(__name__)

COZTER_DIR = ".cozter"
SCHEDULES_FILE = "schedules.json"
SESSIONS_DIR = "sessions"


def _path(workspace: str) -> str:
    return os.path.join(workspace, COZTER_DIR, SCHEDULES_FILE)


def _load_all(workspace: str) -> dict:
    path = _path(workspace)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError):
        logger.warning("Corrupt schedules file, ignoring: %s", path)
    return {}


def _save_all(workspace: str, data: dict) -> None:
    target_dir = os.path.join(workspace, COZTER_DIR)
    os.makedirs(target_dir, exist_ok=True)
    _atomic_write(_path(workspace), data, tmp_dir=target_dir)


def add_schedule(
    workspace: str, user_id: str | int, schedule: dict,
) -> None:
    data = _load_all(workspace)
    data.setdefault(str(user_id), []).append(schedule)
    _save_all(workspace, data)


def remove_schedule(
    workspace: str, user_id: str | int, schedule_id: str,
) -> bool:
    data = _load_all(workspace)
    key = str(user_id)
    schedules = data.get(key, [])
    kept = [s for s in schedules if s.get("id") != schedule_id]
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
    return list(_load_all(workspace).get(str(user_id), []))


def iter_all_schedules(
    workspace: str,
) -> list[tuple[str, dict]]:
    """Return [(user_id_str, schedule_dict)] for every schedule in *workspace*."""
    out: list[tuple[str, dict]] = []
    for uid, schedules in _load_all(workspace).items():
        for s in schedules:
            out.append((uid, s))
    return out


def update_schedule_fired(
    workspace: str,
    user_id: str | int,
    schedule_id: str,
    fired_at: str,
) -> None:
    data = _load_all(workspace)
    key = str(user_id)
    schedules = data.get(key, [])
    changed = False
    for s in schedules:
        if s.get("id") == schedule_id:
            s["last_fired"] = fired_at
            changed = True
            break
    if changed:
        _save_all(workspace, data)


def migrate_from_sessions(workspace: str) -> int:
    """Lift schedules embedded in session JSONs into the new store.

    Idempotent: a session whose ``schedules`` field has already been
    cleared is skipped. Returns the number of schedule entries moved.
    Called once on bot startup so users keep schedules they created
    before the split.
    """
    sessions_dir = os.path.join(workspace, COZTER_DIR, SESSIONS_DIR)
    if not os.path.isdir(sessions_dir):
        return 0

    moved = 0
    cozter_dir = os.path.join(workspace, COZTER_DIR)
    target = _load_all(workspace)
    target_dirty = False

    for fname in os.listdir(sessions_dir):
        if not fname.endswith(".json"):
            continue
        spath = os.path.join(sessions_dir, fname)
        try:
            with open(spath, encoding="utf-8") as f:
                sdata = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        embedded = sdata.get("schedules")
        if not embedded:
            continue
        # Hoist each schedule into the per-user list, keyed by the
        # schedule's stored user_id (falling back to the session's
        # name field is unsafe; if user_id is missing we drop it to
        # avoid orphaning into a wrong owner).
        for sched in embedded:
            uid = sched.get("user_id")
            if uid is None:
                continue
            target.setdefault(str(uid), []).append(sched)
            moved += 1
            target_dirty = True
        # Clear the field on the session so we don't re-migrate.
        sdata.pop("schedules", None)
        try:
            _atomic_write(spath, sdata, tmp_dir=sessions_dir)
        except OSError:
            logger.warning("Failed to rewrite session file %s", spath)

    if target_dirty:
        os.makedirs(cozter_dir, exist_ok=True)
        _save_all(workspace, target)
        logger.info(
            "Migrated %d schedule(s) from session files to %s",
            moved, _path(workspace),
        )
    return moved
