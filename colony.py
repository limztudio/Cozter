"""Workspace-level shared memory ("colony").

Colony is a workspace-shared layer that sits above each session's
``long_term`` list. Every Nth per-session compaction (``colony_interval``,
default 3, configurable in workspace settings), the agent looks across
all sessions in the workspace, finds long-term items that recur or
overlap, and promotes them into the colony — removing them from each
session's own list so there's a single canonical place for them.

When building the prompt for any turn, the colony is prepended to the
session's long-term list and summary so the agent sees both
workspace-shared and session-scoped knowledge.

File shape (``.cozter/colony.json``):
    {
        "items": ["fact 1", "fact 2", ...],
        "compact_count": <int>
    }

``compact_count`` is bumped after every successful per-session
compaction; when ``compact_count % colony_interval == 0`` we run the
consolidation pass.
"""

import json
import logging
import os

from .utils import atomic_write as _atomic_write

logger = logging.getLogger(__name__)

COZTER_DIR = ".cozter"
COLONY_FILE = "colony.json"

DEFAULT_COLONY_INTERVAL = 3
COLONY_CAP = 100  # hard cap on items so the prompt context stays bounded


def _path(workspace: str) -> str:
    return os.path.join(workspace, COZTER_DIR, COLONY_FILE)


def _load(workspace: str) -> dict:
    path = _path(workspace)
    if not os.path.exists(path):
        return {"items": [], "compact_count": 0}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                # Tolerate older / partial files.
                return {
                    "items": list(data.get("items") or []),
                    "compact_count": int(data.get("compact_count") or 0),
                }
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        logger.warning("Corrupt colony file, ignoring: %s", path)
    return {"items": [], "compact_count": 0}


def _save(workspace: str, data: dict) -> None:
    target_dir = os.path.join(workspace, COZTER_DIR)
    os.makedirs(target_dir, exist_ok=True)
    _atomic_write(_path(workspace), data, tmp_dir=target_dir)


def get_items(workspace: str) -> list[str]:
    return _load(workspace)["items"]


def set_items(workspace: str, items: list[str]) -> None:
    data = _load(workspace)
    cleaned = [s for s in (i.strip() for i in items if i) if s]
    if len(cleaned) > COLONY_CAP:
        dropped = len(cleaned) - COLONY_CAP
        logger.warning(
            "Colony rewrite exceeded cap (%d); dropping %d oldest item(s)",
            COLONY_CAP, dropped,
        )
        cleaned = cleaned[-COLONY_CAP:]
    data["items"] = cleaned
    _save(workspace, data)


def get_compact_count(workspace: str) -> int:
    return _load(workspace)["compact_count"]


def bump_compact_count(workspace: str) -> int:
    """Increment the workspace-wide compaction counter and return its new value.

    Caller is expected to hold the workspace lock so two concurrent
    compactions don't both observe the same count.
    """
    data = _load(workspace)
    data["compact_count"] = int(data.get("compact_count") or 0) + 1
    _save(workspace, data)
    return data["compact_count"]
