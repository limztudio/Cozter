"""Session management - stored in each workspace's .cozter/sessions/.

Each user has a "last session" pointer at
``.cozter/last_session.json`` so that conversations resume in the
same session across bot restarts. The router is only consulted when
that pointer is missing or stale (workspace just created, session
file deleted, user ran ``/newsession``).
"""

import json
import logging
import os
import uuid
from datetime import datetime

from .utils import COZTER_DIR, take_recent_lines
from .utils import load_json_object
from .utils import normalize_string_list
from .utils import save_json_object

logger = logging.getLogger(__name__)

LONG_TERM_CAP = 50
# Cap each message's content when including it in a prompt so a single
# long AI response cannot consume the entire token budget.
MSG_CONTENT_MAX = 800

SESSIONS_DIR = "sessions"
LAST_SESSION_FILE = "last_session.json"


def _sessions_dir(workspace: str) -> str:
    return os.path.join(workspace, COZTER_DIR, SESSIONS_DIR)


def _session_path(workspace: str, session_id: str) -> str:
    return os.path.join(_sessions_dir(workspace), f"{session_id}.json")


def _last_session_path(workspace: str) -> str:
    return os.path.join(workspace, COZTER_DIR, LAST_SESSION_FILE)


# ---------------------------------------------------------------------------
# Last-session pointer (per workspace, keyed by user)
# ---------------------------------------------------------------------------


def _load_last_session_map(workspace: str) -> dict:
    """Return the user_id -> session_id map for *workspace* (empty on failure).

    JSON corruption is logged and swallowed: this is a UX hint, not
    load-bearing state, so a broken file falls back to "no last session"
    rather than blocking the turn.
    """
    return load_json_object(
        _last_session_path(workspace), "last_session file", logger,
    )


def get_last_session(workspace: str, user_id: int | str) -> str | None:
    """Return the session id this user was last writing into, or None."""
    val = _load_last_session_map(workspace).get(str(user_id))
    return val if isinstance(val, str) and val else None


def set_last_session(
    workspace: str, user_id: int | str, session_id: str,
) -> None:
    """Record that *user_id* is now working in *session_id*."""
    data = _load_last_session_map(workspace)
    data[str(user_id)] = session_id
    save_json_object(_last_session_path(workspace), data)


def migrate_last_session(
    workspace: str,
    source_user_ids: list[int | str] | tuple[int | str, ...],
    target_user_id: int | str,
) -> bool:
    """Copy the first legacy last-session pointer to a new user key."""
    data = _load_last_session_map(workspace)
    target_key = str(target_user_id)
    if isinstance(data.get(target_key), str) and data[target_key]:
        return False

    for source_user_id in source_user_ids:
        source_key = str(source_user_id)
        if source_key == target_key:
            continue
        session_id = data.get(source_key)
        if not isinstance(session_id, str) or not session_id:
            continue
        data[target_key] = session_id
        save_json_object(_last_session_path(workspace), data)
        return True
    return False


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

def _safe_text(value: object, default: str = "") -> str:
    """Return a string for persisted free-text fields."""
    if isinstance(value, str):
        return value
    if value is None:
        return default
    return str(value)


def _normalize_message(value: object) -> dict | None:
    if not isinstance(value, dict):
        return None
    role = value.get("role", "?")
    if not isinstance(role, str) or not role:
        role = "?"
    content = _safe_text(value.get("content"))
    return {"role": role, "content": content}


def _normalize_session_data(value: object, *, path: str = "") -> dict | None:
    """Return a crash-safe session dict, or None if identity is missing."""
    if not isinstance(value, dict):
        if path:
            logger.warning("Ignoring non-object session file: %s", path)
        return None

    session_id = value.get("id")
    if not isinstance(session_id, str) or not session_id:
        if path:
            logger.warning("Ignoring session without valid id: %s", path)
        return None

    data = dict(value)
    name = data.get("name")
    data["name"] = name if isinstance(name, str) and name else session_id[:8]
    created = data.get("created")
    data["created"] = created if isinstance(created, str) else ""

    messages = data.get("messages", [])
    if isinstance(messages, list):
        data["messages"] = [
            msg for msg in (_normalize_message(m) for m in messages)
            if msg is not None
        ]
    else:
        data["messages"] = []

    summary = data.get("summary")
    data["summary"] = summary if isinstance(summary, str) and summary else None

    long_term = data.get("long_term", [])
    data["long_term"] = normalize_string_list(long_term, strip=False)

    compacted_count = data.get("compacted_count", 0)
    if (
        not isinstance(compacted_count, int)
        or isinstance(compacted_count, bool)
        or compacted_count < 0
    ):
        compacted_count = 0
    data["compacted_count"] = compacted_count

    return data


def total_message_count(data: dict) -> int:
    """Total messages in a session (compacted + currently stored)."""
    compacted_count = data.get("compacted_count", 0)
    if (
        not isinstance(compacted_count, int)
        or isinstance(compacted_count, bool)
        or compacted_count < 0
    ):
        compacted_count = 0
    messages = data.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    return compacted_count + len(messages)


def format_msg_line(msg: dict, cap: int | None = MSG_CONTENT_MAX) -> str:
    """Format a session message as ``Role: content``.

    Content is truncated with an ellipsis when it exceeds *cap*.
    Pass ``cap=None`` to disable truncation (used by compaction, where
    the per-call budget is large enough to afford full message text).
    """
    msg = msg if isinstance(msg, dict) else {}
    role = msg.get("role", "?")
    if not isinstance(role, str) or not role:
        role = "?"
    role = role.capitalize()
    content = _safe_text(msg.get("content"))
    if cap is not None and len(content) > cap:
        content = content[:cap] + "…"
    return f"{role}: {content}"


def take_recent_messages(
    messages: list[dict],
    budget: int,
    *,
    cap: int | None = MSG_CONTENT_MAX,
) -> list[str]:
    """Format the most recent messages that fit in *budget* chars.

    Each line is formatted via :func:`format_msg_line` with the given
    *cap*. Wrapper around :func:`utils.take_recent_lines`.
    """
    return take_recent_lines(
        messages, budget, lambda m: format_msg_line(m, cap=cap),
    )


def list_sessions_with_data(workspace: str) -> list[dict]:
    """Return every session's full data dict, sorted by created desc.

    Skips files that don't parse as JSON or lack an ``id``. Callers
    that only need lightweight metadata can use :func:`list_sessions`,
    which projects from this result.
    """
    sdir = _sessions_dir(workspace)
    if not os.path.isdir(sdir):
        return []
    out: list[dict] = []
    for fname in os.listdir(sdir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(sdir, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        data = _normalize_session_data(data, path=fpath)
        if data is None:
            continue
        out.append(data)
    out.sort(key=lambda d: d.get("created", ""), reverse=True)
    return out


def list_sessions(workspace: str) -> list[dict]:
    """Return [{id, name, created, message_count}] sorted by created desc."""
    return [
        {
            "id": d["id"],
            "name": d.get("name", d["id"][:8]),
            "created": d.get("created", ""),
            "message_count": total_message_count(d),
        }
        for d in list_sessions_with_data(workspace)
    ]


def create_session(workspace: str, name: str | None = None) -> dict:
    """Create a new session and return its metadata.

    The auto-compaction interval is read from workspace settings at
    compaction time, so it isn't stored on the session itself.
    """
    session_id = uuid.uuid4().hex[:12]
    now = datetime.now().isoformat()
    data: dict = {
        "id": session_id,
        "name": name or f"Session {now[:10]}",
        "created": now,
        "messages": [],
        "summary": None,
        "long_term": [],
        "compacted_count": 0,
    }
    save_json_object(_session_path(workspace, session_id), data)
    return data


def load_session(workspace: str, session_id: str) -> dict | None:
    path = _session_path(workspace, session_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("Corrupt session file, ignoring: %s", path)
        return None
    return _normalize_session_data(data, path=path)


def save_session(workspace: str, session_id: str, data: dict) -> None:
    save_json_object(_session_path(workspace, session_id), data)


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
    summarized_count: int | None = None,
) -> None:
    """Store a compacted summary, keeping only the last *keep_recent* messages.

    long_term_rewrite, when provided, replaces the long_term list entirely.
    None means the model did not emit a rewrite block; the existing list is
    kept unchanged. The list is capped at LONG_TERM_CAP after writing.

    title, when provided and non-empty, replaces the session name. The
    compaction prompt asks the model for a short title alongside the
    summary so the user-visible name reflects the latest topic.

    summarized_count is how many messages existed (and were thus covered by
    *summary*) when the compaction snapshot was taken. Compaction runs outside
    the workspace lock, so a concurrent turn can append messages while the
    summary is being computed; trimming against the *current* length would
    drop those newer, un-summarized messages. Trimming against the snapshot
    count instead keeps everything appended since. None (manual/legacy callers)
    falls back to the current length.
    """
    data = load_session(workspace, session_id)
    if data is None:
        return
    msgs = data.get("messages", [])
    basis = len(msgs) if summarized_count is None else min(
        summarized_count, len(msgs),
    )
    trimmed = max(0, basis - keep_recent)
    data["compacted_count"] = data.get("compacted_count", 0) + trimmed
    # Keep everything from the trim point onward: the last *keep_recent* of the
    # summarized prefix (raw context continuity) plus any messages appended
    # after the snapshot. msgs[trimmed:] reduces to msgs[-keep_recent:] in the
    # no-race case.
    data["messages"] = msgs[trimmed:]
    data["summary"] = summary

    if long_term_rewrite is not None:
        long_term = normalize_string_list(long_term_rewrite, strip=False)
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
