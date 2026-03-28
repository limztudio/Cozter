"""Session management — stored in each workspace's .cozter/sessions/."""

import json
import logging
import os
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

COZTER_DIR = ".cozter"
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
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_state(workspace: str, state: dict) -> None:
    os.makedirs(os.path.join(workspace, COZTER_DIR), exist_ok=True)
    with open(_state_path(workspace), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def get_current_session_id(workspace: str, user_id: int) -> str | None:
    return _load_state(workspace).get(str(user_id))


def set_current_session_id(workspace: str, user_id: int, session_id: str) -> None:
    state = _load_state(workspace)
    state[str(user_id)] = session_id
    _save_state(workspace, state)


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

def list_sessions(workspace: str) -> list[dict]:
    """Return list of {id, name, created, message_count} sorted by created desc."""
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
                "message_count": len(data.get("messages", [])),
            })
        except Exception:
            continue
    sessions.sort(key=lambda s: s["created"], reverse=True)
    return sessions


def create_session(workspace: str, name: str | None = None) -> dict:
    """Create a new session and return its metadata."""
    os.makedirs(_sessions_dir(workspace), exist_ok=True)
    session_id = uuid.uuid4().hex[:12]
    now = datetime.now().isoformat()
    data = {
        "id": session_id,
        "name": name or f"Session {now[:10]}",
        "created": now,
        "messages": [],
        "summary": None,
        "compact_interval": 20,
        "compacted_count": 0,
    }
    with open(_session_path(workspace, session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


def load_session(workspace: str, session_id: str) -> dict | None:
    path = _session_path(workspace, session_id)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_session(workspace: str, session_id: str, data: dict) -> None:
    os.makedirs(_sessions_dir(workspace), exist_ok=True)
    with open(_session_path(workspace, session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Message persistence
# ---------------------------------------------------------------------------

def append_message(workspace: str, session_id: str, message: dict) -> int:
    """Append a message to the session. Returns total message count."""
    data = load_session(workspace, session_id)
    if data is None:
        return 0
    data["messages"].append(message)
    save_session(workspace, session_id, data)
    return len(data["messages"])


def get_messages(workspace: str, session_id: str) -> list[dict]:
    data = load_session(workspace, session_id)
    if data is None:
        return []
    return data.get("messages", [])


def replace_messages(workspace: str, session_id: str, messages: list[dict]) -> None:
    """Replace all messages in a session (used after compaction)."""
    data = load_session(workspace, session_id)
    if data is None:
        return
    data["messages"] = messages
    save_session(workspace, session_id, data)


# ---------------------------------------------------------------------------
# Ensure a session exists for a user in a workspace
# ---------------------------------------------------------------------------

def ensure_session(workspace: str, user_id: int) -> str:
    """Return the current session ID, creating one if needed."""
    sid = get_current_session_id(workspace, user_id)
    if sid:
        data = load_session(workspace, sid)
        if data is not None:
            return sid
    # Create new session
    data = create_session(workspace)
    set_current_session_id(workspace, user_id, data["id"])
    return data["id"]


# ---------------------------------------------------------------------------
# Summary and compaction
# ---------------------------------------------------------------------------

def get_summary(workspace: str, session_id: str) -> str | None:
    data = load_session(workspace, session_id)
    if data is None:
        return None
    return data.get("summary")


def set_summary(
    workspace: str, session_id: str, summary: str, keep_recent: int = 10,
) -> None:
    """Store a compacted summary, keeping only the last *keep_recent* messages."""
    data = load_session(workspace, session_id)
    if data is None:
        return
    msgs = data.get("messages", [])
    trimmed = max(0, len(msgs) - keep_recent)
    data["compacted_count"] = data.get("compacted_count", 0) + trimmed
    data["messages"] = msgs[-keep_recent:] if keep_recent else []
    data["summary"] = summary
    save_session(workspace, session_id, data)


def get_compact_interval(workspace: str, session_id: str) -> int:
    data = load_session(workspace, session_id)
    if data is None:
        return 20
    return data.get("compact_interval", 20)


def set_compact_interval(workspace: str, session_id: str, interval: int) -> None:
    data = load_session(workspace, session_id)
    if data is None:
        return
    data["compact_interval"] = interval
    save_session(workspace, session_id, data)


def get_total_message_count(workspace: str, session_id: str) -> int:
    data = load_session(workspace, session_id)
    if data is None:
        return 0
    return data.get("compacted_count", 0) + len(data.get("messages", []))
