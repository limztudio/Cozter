"""Session management - stored in each workspace's .cozter/sessions/.

Sessions are no longer addressed by a "current session per user" — each
user turn is routed to the best-matching session by ``agent.select_or_create_session``.
"""

import json
import logging
import os
import uuid
from datetime import datetime

from .utils import atomic_write as _atomic_write

logger = logging.getLogger(__name__)

COZTER_DIR = ".cozter"
LONG_TERM_CAP = 50
DEFAULT_COMPACT_INTERVAL = 10

SESSIONS_DIR = "sessions"


def _sessions_dir(workspace: str) -> str:
    return os.path.join(workspace, COZTER_DIR, SESSIONS_DIR)


def _session_path(workspace: str, session_id: str) -> str:
    return os.path.join(_sessions_dir(workspace), f"{session_id}.json")


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

def total_message_count(data: dict) -> int:
    """Total messages in a session (compacted + currently stored)."""
    return data.get("compacted_count", 0) + len(data.get("messages", []))


def list_sessions(workspace: str) -> list[dict]:
    """Return [{id, name, created, message_count}] sorted by created desc."""
    sdir = _sessions_dir(workspace)
    if not os.path.isdir(sdir):
        return []
    sessions = []
    for fname in os.listdir(sdir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(sdir, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "id": data["id"],
                "name": data.get("name", data["id"][:8]),
                "created": data.get("created", ""),
                "message_count": total_message_count(data),
            })
        except Exception:
            continue
    sessions.sort(key=lambda s: s["created"], reverse=True)
    return sessions


def create_session(workspace: str, name: str | None = None) -> dict:
    """Create a new session and return its metadata.

    The auto-compaction interval is read from workspace settings at
    compaction time, so it isn't stored on the session itself.
    """
    sdir = _sessions_dir(workspace)
    os.makedirs(sdir, exist_ok=True)
    session_id = uuid.uuid4().hex[:12]
    now = datetime.now().isoformat()
    data = {
        "id": session_id,
        "name": name or f"Session {now[:10]}",
        "created": now,
        "messages": [],
        "summary": None,
        "long_term": [],
        "compacted_count": 0,
    }
    _atomic_write(_session_path(workspace, session_id), data, tmp_dir=sdir)
    return data


def load_session(workspace: str, session_id: str) -> dict | None:
    path = _session_path(workspace, session_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("Corrupt session file, ignoring: %s", path)
        return None


def save_session(workspace: str, session_id: str, data: dict) -> None:
    sdir = _sessions_dir(workspace)
    os.makedirs(sdir, exist_ok=True)
    _atomic_write(_session_path(workspace, session_id), data, tmp_dir=sdir)


def delete_session(workspace: str, session_id: str) -> bool:
    """Remove a session file. Returns True if a file was deleted.

    Used by the scheduler to clean up the ephemeral session it spins
    up to run a scheduled command.
    """
    path = _session_path(workspace, session_id)
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        logger.warning("Failed to delete session file: %s", path)
        return False


def is_default_name(name: str | None) -> bool:
    """True if *name* still matches the auto-generated ``Session YYYY-MM-DD``.

    Used as the trigger for auto-renaming: a session whose name has
    been changed (manually or by a previous title pass) is left alone
    so we don't overwrite a meaningful title with a stale draft.
    """
    if not name:
        return True
    if not name.startswith("Session "):
        return False
    rest = name[len("Session "):]
    if len(rest) != 10:
        return False
    parts = rest.split("-")
    return (
        len(parts) == 3
        and len(parts[0]) == 4
        and all(p.isdigit() for p in parts)
    )


def set_session_name(
    workspace: str, session_id: str, name: str,
) -> None:
    name = name.strip()
    if not name:
        return
    data = load_session(workspace, session_id)
    if data is None:
        return
    data["name"] = name
    save_session(workspace, session_id, data)


# ---------------------------------------------------------------------------
# Message persistence
# ---------------------------------------------------------------------------

def append_messages(
    workspace: str, session_id: str, messages: list[dict],
) -> int:
    """Append multiple messages in a single read+write. Returns total count."""
    data = load_session(workspace, session_id)
    if data is None:
        return 0
    data["messages"].extend(messages)
    save_session(workspace, session_id, data)
    return len(data["messages"])


# ---------------------------------------------------------------------------
# Summary and compaction
# ---------------------------------------------------------------------------

def set_summary(
    workspace: str,
    session_id: str,
    summary: str,
    keep_recent: int = 10,
    long_term_rewrite: list[str] | None = None,
    title: str | None = None,
) -> None:
    """Store a compacted summary, keeping only the last *keep_recent* messages.

    long_term_rewrite, when provided, replaces the long_term list entirely.
    None means the model did not emit a rewrite block; the existing list is
    kept unchanged. The list is capped at LONG_TERM_CAP after writing.

    title, when provided and non-empty, replaces the session name. The
    compaction prompt asks the model for a short title alongside the
    summary so the user-visible name reflects the latest topic.
    """
    data = load_session(workspace, session_id)
    if data is None:
        return
    msgs = data.get("messages", [])
    trimmed = max(0, len(msgs) - keep_recent)
    data["compacted_count"] = data.get("compacted_count", 0) + trimmed
    data["messages"] = msgs[-keep_recent:] if keep_recent else []
    data["summary"] = summary

    if long_term_rewrite is not None:
        long_term = [item for item in long_term_rewrite if item]
        if len(long_term) > LONG_TERM_CAP:
            dropped = len(long_term) - LONG_TERM_CAP
            logger.warning(
                "Long-term rewrite exceeded cap (%d); "
                "dropping %d oldest item(s)",
                LONG_TERM_CAP, dropped,
            )
            long_term = long_term[-LONG_TERM_CAP:]
        data["long_term"] = long_term

    if title:
        cleaned = title.strip()
        if cleaned:
            data["name"] = cleaned

    save_session(workspace, session_id, data)


# Schedules live in ``schedules.py`` (``.cozter/schedules.json``).
