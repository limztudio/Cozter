"""Plugin: search and read the workspace's durable chat memory.

Every session persists its transcript, summary, and long-term notes to
``.cozter/sessions/<id>.json``, and the whole workspace shares
``.cozter/colony.json``. None of that is searchable by the built-in
tools (the discovery tools skip ``.cozter/``), and the prompt only ever
contains the *current* session's context - so anything decided in an
earlier session was effectively lost to the model. This plugin makes
that history queryable: full-text search across all sessions, colony,
summaries, and long-term notes, plus session listing and bounded reads.

The implementation is deliberately self-contained (plain JSON reads via
``resolve_inside_workspace``) so the plugin never imports app modules:
plugin files load during ``agent_tools`` package initialization, where
a root-package import (``session``/``workspace`` pull in
``backends_agent``) would be circular.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, ClassVar

from ..base import (
    AgentTool,
    coerce_int_arg,
    object_parameters,
    require_nonempty_string_arg,
    resolve_inside_workspace,
)

_SESSIONS_RELPATH = ".cozter/sessions"
_COLONY_RELPATH = ".cozter/colony.json"
_COLONY_ITEMS_CAP = 100
_ACTIONS = ("search", "list", "read")

# Bounds sized to execute_tool's 4,000-character result cap.
_MATCH_LIMIT_DEFAULT = 8
_MATCH_LIMIT_MAX = 20
_EXCERPT_CHARS = 160
_EXCERPT_CONTEXT = 40
_LINE_CAP = 300
_LIST_SESSIONS_CAP = 40
_READ_MESSAGES_DEFAULT = 20
_READ_MESSAGES_MAX = 100
_OUTPUT_BUDGET = 3_600

_NEWEST_ALIASES = frozenset({"last", "latest", "newest", "current"})


def _load_sessions(sessions_dir: str) -> list[dict]:
    """Load every well-formed session file, newest first.

    Mirrors the app-level session loader's crash-safety (skip corrupt
    files, require ``id`` to match the file name) without importing it.
    """
    try:
        names = os.listdir(sessions_dir)
    except OSError:
        return []
    sessions: list[dict] = []
    for fname in sorted(names):
        if not fname.endswith(".json"):
            continue
        session_id = fname[: -len(".json")]
        data = _load_session_file(os.path.join(sessions_dir, fname), session_id)
        if data is not None:
            sessions.append(data)
    # Missing timestamps sort as oldest rather than jumping to the front.
    sessions.sort(key=lambda d: d["created"] or "0000", reverse=True)
    return sessions


def _load_session_file(path: str, session_id: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("id") != session_id:
        return None

    name = raw.get("name")
    created = raw.get("created")
    summary = raw.get("summary")
    raw_messages = raw.get("messages")
    raw_long_term = raw.get("long_term")
    messages: list[dict] = []
    if isinstance(raw_messages, list):
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            messages.append({
                "role": role if isinstance(role, str) and role else "?",
                "content": content if isinstance(content, str) else "",
            })
    return {
        "id": session_id,
        "name": name if isinstance(name, str) and name else session_id[:8],
        "created": created if isinstance(created, str) else "",
        "summary": summary if isinstance(summary, str) and summary else "",
        "long_term": [
            item
            for item in (
                raw_long_term if isinstance(raw_long_term, list) else []
            )
            if isinstance(item, str)
        ],
        "messages": messages,
    }


def _colony_items(workspace: str) -> list[str]:
    try:
        path = resolve_inside_workspace(workspace, _COLONY_RELPATH)
    except ValueError:
        return []
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return []
    items = raw.get("items") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return []
    return [
        item for item in items if isinstance(item, str) and item.strip()
    ][:_COLONY_ITEMS_CAP]


def _sessions_dir(workspace: str) -> str | None:
    try:
        return resolve_inside_workspace(workspace, _SESSIONS_RELPATH)
    except ValueError:
        return None


def _session_label(data: dict) -> str:
    label = f"[{data['name']}"
    if data["created"]:
        label += f" · {data['created'][:10]}"
    return label + "]"


def _excerpt(text: str, index: int, needle_len: int) -> str:
    start = max(0, index - _EXCERPT_CONTEXT)
    end = min(len(text), index + needle_len + _EXCERPT_CHARS)
    prefix = "…" if start else ""
    suffix = "…" if end < len(text) else ""
    snippet = " ".join(text[start:end].split())
    return f"{prefix}{snippet}{suffix}"[:_EXCERPT_CHARS + 2]


def _iter_search_texts(data: dict):
    if data["summary"]:
        yield "Summary:", data["summary"]
    for item in data["long_term"]:
        yield "Long-term:", item
    for msg in data["messages"]:
        yield f"{msg['role'].capitalize()}:", msg["content"]


def _fit_output(lines: list[str], header: str) -> str:
    """Join *lines* under the result budget, dropping from the end."""
    dropped = 0
    while lines and len(header) + sum(len(line) + 1 for line in lines) > (
        _OUTPUT_BUDGET
    ):
        lines.pop()
        dropped += 1
    if dropped:
        lines.append(f"(…{dropped} more result(s) omitted to fit the limit)")
    return "\n".join([header, *lines])


class MemoryTool(AgentTool):
    name = "memory"
    order = 20  # utility tools group, next to notes/git_info
    description = (
        "Chat memory (sessions, summaries, long-term, colony)."
    )
    parameters: ClassVar[dict[str, Any]] = object_parameters(
        {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
            },
            "query": {
                "type": "string",
            },
            "session": {
                "type": "string",
                "description": "Name/id/prefix/'last'.",
            },
            "limit": {
                "type": "integer",
                "description": "Default 8/20, max 20/100.",
            },
        },
        ["action"],
    )

    async def run(self, workspace_path: str, args: dict) -> str:
        action = args.get("action")
        if not isinstance(action, str) or action not in _ACTIONS:
            return "Error: 'action' must be one of search, list, read"
        sessions_dir = _sessions_dir(workspace_path)
        if sessions_dir is None:
            return "Error: invalid workspace path"
        if action == "search":
            return await asyncio.to_thread(
                self._search, sessions_dir, workspace_path, args,
            )
        if action == "list":
            return await asyncio.to_thread(
                self._list, sessions_dir, workspace_path,
            )
        return await asyncio.to_thread(self._read, sessions_dir, args)

    def _search(self, sessions_dir: str, workspace: str, args: dict) -> str:
        query, error = require_nonempty_string_arg(
            args, "query", strip=True,
        )
        if error:
            return error
        assert query is not None
        limit = coerce_int_arg(
            args.get("limit") or _MATCH_LIMIT_DEFAULT,
            default=_MATCH_LIMIT_DEFAULT,
            minimum=1,
            maximum=_MATCH_LIMIT_MAX,
        )
        needle = query.casefold()

        matches: list[str] = []
        total = 0
        for data in _load_sessions(sessions_dir):
            label = _session_label(data)
            for kind, text in _iter_search_texts(data):
                index = text.casefold().find(needle)
                if index < 0:
                    continue
                total += 1
                if len(matches) < limit:
                    matches.append(
                        f"- {label} {kind} {_excerpt(text, index, len(needle))}"
                    )

        for item in _colony_items(workspace):
            index = item.casefold().find(needle)
            if index < 0:
                continue
            total += 1
            if len(matches) < limit:
                matches.append(f"- [Colony] {_excerpt(item, index, len(needle))}")

        if not total:
            return (
                f"No matches for '{query}' in any session, summary,"
                " long-term note, or colony item."
            )
        header = f"Found {total} match(es) for '{query}'"
        if total > len(matches):
            header += (
                f" — showing the {len(matches)} newest"
                " (raise *limit* to see more):"
            )
        else:
            header += ":"
        return _fit_output(matches, header)

    def _list(self, sessions_dir: str, workspace: str) -> str:
        sessions = _load_sessions(sessions_dir)
        lines: list[str] = []
        for data in sessions[:_LIST_SESSIONS_CAP]:
            flags = []
            if data["summary"]:
                flags.append("summary")
            if data["long_term"]:
                flags.append(f"{len(data['long_term'])} long-term")
            flag_text = f", {', '.join(flags)}" if flags else ""
            lines.append(
                f"- {data['name']} — id {data['id'][:8]},"
                f" {data['created'][:10] or 'unknown date'},"
                f" {len(data['messages'])} messages{flag_text}"
            )
        header = "Sessions (newest first):"
        omitted = len(sessions) - min(len(sessions), _LIST_SESSIONS_CAP)
        if omitted > 0:
            lines.append(f"(…and {omitted} older session(s))")
        lines.append(
            f"Colony: {len(_colony_items(workspace))}"
            " items in .cozter/colony.json"
        )
        if not sessions:
            return "No sessions recorded in this workspace yet."
        return _fit_output(lines, header)

    def _read(self, sessions_dir: str, args: dict) -> str:
        target = args.get("session")
        if not isinstance(target, str) or not target.strip():
            return (
                "Error: 'session' is required for read (a name, an id or"
                " unique prefix, or 'last')"
            )
        count = coerce_int_arg(
            args.get("limit") or _READ_MESSAGES_DEFAULT,
            default=_READ_MESSAGES_DEFAULT,
            minimum=1,
            maximum=_READ_MESSAGES_MAX,
        )
        data, error = _find_session(_load_sessions(sessions_dir), target)
        if error or data is None:
            return error or "Error: session not found"

        total = len(data["messages"])
        window = data["messages"][-count:]
        shown_from = total - len(window) + 1
        header = (
            f"Session: {data['name']} (id {data['id'][:8]},"
            f" {data['created'][:10] or 'unknown date'},"
            f" {total} message(s), showing last {len(window)})"
        )
        lines = [
            f"{shown_from + offset}. {msg['role'].capitalize()}:"
            f" {msg['content'][:_LINE_CAP]}"
            + ("…" if len(msg["content"]) > _LINE_CAP else "")
            for offset, msg in enumerate(window)
        ]
        return _fit_output(lines, header)


def _find_session(
    sessions: list[dict],
    target: str,
) -> tuple[dict | None, str | None]:
    """Match *target* by alias, exact id/name, or unique prefix."""
    if not sessions:
        return None, "Error: no sessions recorded in this workspace yet"
    key = target.strip().casefold()
    if key in _NEWEST_ALIASES:
        return sessions[0], None
    exact_id = [s for s in sessions if s["id"] == target.strip()]
    if exact_id:
        return exact_id[0], None
    exact_name = [s for s in sessions if s["name"].casefold() == key]
    if len(exact_name) == 1:
        return exact_name[0], None
    prefix = [
        s for s in sessions
        if s["id"].startswith(key) or s["name"].casefold().startswith(key)
    ]
    if len(prefix) == 1:
        return prefix[0], None
    if len(prefix) > 1:
        names = ", ".join(s["name"] for s in prefix[:5])
        return None, (
            f"Error: '{target}' matches {len(prefix)} sessions ({names});"
            " use a longer prefix or the full id"
        )
    return None, (
        f"Error: no session named '{target}'; use action=list to see"
        " sessions"
    )


if __name__ == "__main__":
    MemoryTool.run_as_script()
