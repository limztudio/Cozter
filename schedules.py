"""Workspace-level schedule store.

Schedules are stored in ``.cozter/schedules.json`` so a fired schedule
can run in its own ephemeral session without being tied to whichever
session happened to be current when the user created it.

File shape:
    {"<user_id>": [<schedule_dict>, ...], ...}

Each ``<schedule_dict>``:
    {id, days, time, command, created, chat_id, user_id, last_fired?}
"""

import json
import logging
import os

from .utils import COZTER_DIR
from .utils import atomic_write as _atomic_write

logger = logging.getLogger(__name__)

SCHEDULES_FILE = "schedules.json"


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
