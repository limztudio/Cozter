"""Session management - stored in each workspace's .cozter/sessions/."""

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
SESSION_INDEX = "session_state.json"


def _sessions_dir(workspace: str) -> str:
    return os.path.join(workspace, COZTER_DIR, SESSIONS_DIR)


def _session_path(workspace: str, session_id: str) -> str:
    return os.path.join(_sessions_dir(workspace), f"{session_id}.json")


def _state_path(workspace: str) -> str:
    return os.path.join(workspace, COZTER_DIR, SESSION_INDEX)


# ---------------------------------------------------------------------------
# State: tracks current session per user
# ---------------------------------------------------------------------------

def _load_state(workspace: str) -> dict:
    path = _state_path(workspace)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt session state file, ignoring: %s", path)
    return {}


def _save_state(workspace: str, state: dict) -> None:
    target_dir = os.path.join(workspace, COZTER_DIR)
    os.makedirs(target_dir, exist_ok=True)
    _atomic_write(_state_path(workspace), state, tmp_dir=target_dir)


def get_current_session_id(workspace: str, user_id: int) -> str | None:
    return _load_state(workspace).get(str(user_id))


def set_current_session_id(
    workspace: str, user_id: int, session_id: str,
) -> None:
    state = _load_state(workspace)
    state[str(user_id)] = session_id
    _save_state(workspace, state)


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
    """Create a new session and return its metadata."""
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
        "compact_interval": DEFAULT_COMPACT_INTERVAL,
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
# Ensure a session exists for a user in a workspace
# ---------------------------------------------------------------------------

def ensure_session_with_data(
    workspace: str, user_id: int,
) -> tuple[str, dict]:
    """Return (session_id, loaded data), creating the session if needed."""
    sid = get_current_session_id(workspace, user_id)
    if sid:
        data = load_session(workspace, sid)
        if data is not None:
            return (sid, data)
    data = create_session(workspace)
    set_current_session_id(workspace, user_id, data["id"])
    return (data["id"], data)


def ensure_session(workspace: str, user_id: int) -> str:
    """Return the current session ID, creating one if needed."""
    return ensure_session_with_data(workspace, user_id)[0]


# ---------------------------------------------------------------------------
# Summary and compaction
# ---------------------------------------------------------------------------

def set_summary(
    workspace: str,
    session_id: str,
    summary: str,
    keep_recent: int = 10,
    long_term_rewrite: list[str] | None = None,
) -> None:
    """Store a compacted summary, keeping only the last *keep_recent* messages.

    long_term_rewrite, when provided, replaces the long_term list entirely.
    None means the model did not emit a rewrite block; the existing list is
    kept unchanged. The list is capped at LONG_TERM_CAP after writing.
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

    save_session(workspace, session_id, data)


def set_compact_interval(
    workspace: str, session_id: str, interval: int,
) -> None:
    data = load_session(workspace, session_id)
    if data is None:
        return
    data["compact_interval"] = interval
    save_session(workspace, session_id, data)
