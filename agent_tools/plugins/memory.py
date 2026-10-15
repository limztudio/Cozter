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
from collections.abc import Callable
import json
import os
from typing import Any, ClassVar

from ..base import (
    AgentTool,
    coerce_int_arg,
    object_parameters,
    require_nonempty_string_arg,
    resolve_inside_workspace,
    truncate_with_marker,
)

_SESSIONS_RELPATH = ".cozter/sessions"
_COLONY_RELPATH = ".cozter/colony.json"
_ACTIONS = ("search", "list", "read")

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


class _MemoryChangedError(Exception):
    """A transcript changed between counting and rendering its result page."""


def _iter_session_files(sessions_dir: str):
    try:
        names = sorted(os.listdir(sessions_dir))
    except OSError:
        return
    for filename in names:
        if not filename.endswith(".json"):
            continue
        try:
            path = resolve_inside_workspace(sessions_dir, filename)
        except ValueError:
            continue
        yield filename, path, filename[:-len(".json")]


def _stat_signature(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    # Windows 3.12 can report creation time through stat().st_ctime_ns while
    # fstat().st_ctime_ns reports change time. Use the explicitly named birth
    # time when available so path and handle snapshots describe the same data.
    # Older Windows versions expose birth time as ctime; POSIX keeps its real
    # change time. File identity also detects equal-size atomic replacements
    # whose modification timestamp was preserved.
    change_time = (
        getattr(metadata, "st_birthtime_ns", metadata.st_ctime_ns)
        if os.name == "nt" else metadata.st_ctime_ns
    )
    return (
        metadata.st_mtime_ns, change_time, metadata.st_size,
        metadata.st_dev, metadata.st_ino,
    )


def _file_signature(path: str) -> tuple[int, int, int, int, int] | None:
    try:
        metadata = os.stat(path)
    except OSError:
        return None
    return _stat_signature(metadata)


def _load_session_metadata(sessions_dir: str, needle: str | None = None) -> list[dict]:
    """Retain ordering/counts only; discard each transcript before the next."""
    sessions: list[dict] = []
    for filename, path, session_id in _iter_session_files(sessions_dir):
        data = _load_session_file(path, session_id, capture_signature=True)
        if data is None:
            continue
        metadata = {
            "id": data["id"], "name": data["name"], "created": data["created"],
            "has_summary": bool(data["summary"]),
            "long_term_count": len(data["long_term"]),
            "message_count": len(data["messages"]),
            "_filename": filename, "_signature": data["_signature"],
        }
        if needle is not None:
            # Counting never needs an excerpt or a Unicode source-offset map.
            metadata["match_count"] = sum(
                1 for _kind, text in _iter_search_texts(data)
                if needle in text.casefold()
            )
        sessions.append(metadata)
        del data
    # The input is filename-sorted. Stable sorting preserves filename ties,
    # exactly as the original eager transcript loader did.
    sessions.sort(key=lambda item: item["created"] or "0000", reverse=True)
    return sessions


def _reload_session(sessions_dir: str, metadata: dict) -> dict:
    try:
        path = resolve_inside_workspace(sessions_dir, metadata["_filename"])
    except ValueError as exc:
        raise _MemoryChangedError from exc
    if _file_signature(path) != metadata["_signature"]:
        raise _MemoryChangedError
    data = _load_session_file(path, metadata["id"])
    if data is None or _file_signature(path) != metadata["_signature"]:
        raise _MemoryChangedError
    return data


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
        try:
            path = resolve_inside_workspace(sessions_dir, fname)
        except ValueError:
            continue  # A child symlink must not expose files outside memory.
        data = _load_session_file(path, session_id)
        if data is not None:
            sessions.append(data)
    # Missing timestamps sort as oldest.
    sessions.sort(
        key=lambda session_data: session_data["created"] or "0000", reverse=True,
    )
    return sessions


def _load_session_file(path: str, session_id: str, *, capture_signature: bool = False) -> dict | None:
    try:
        with open(path, encoding="utf-8") as file_handle:
            # Handle metadata avoids two extra Windows path-stat round trips
            # per transcript. Atomic replacement leaves this read coherent;
            # a selected reread compares its saved signature with the path.
            before = os.fstat(file_handle.fileno()) if capture_signature else None
            raw = json.load(file_handle)
            after = os.fstat(file_handle.fileno()) if capture_signature else None
    except (OSError, ValueError, RecursionError):
        return None
    if not isinstance(raw, dict) or raw.get("id") != session_id:
        return None
    if before is not None and after is not None and _stat_signature(before) != _stat_signature(after):
        raise _MemoryChangedError

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
    result: dict[str, Any] = {
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
    if after is not None:
        result["_signature"] = _stat_signature(after)
    return result


def _colony_items(workspace: str) -> list[str]:
    try:
        path = resolve_inside_workspace(workspace, _COLONY_RELPATH)
    except ValueError:
        return []
    try:
        with open(path, encoding="utf-8") as file_handle:
            raw = json.load(file_handle)
    except (OSError, ValueError, RecursionError):
        return []
    items = raw.get("items") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return []
    return [
        item for item in items if isinstance(item, str) and item.strip()
    ]


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
    cap = _EXCERPT_CHARS + 2
    full = f"{prefix}{snippet}{suffix}"
    if len(full) <= cap:
        return full
    # Outer cap could drop the marker.
    if suffix:
        budget = max(0, cap - len(prefix) - len(suffix))
        return f"{prefix}{snippet[:budget]}{suffix}"
    budget = max(0, cap - len(prefix) - 1)
    return f"{prefix}{snippet[:budget]}…"


def _casefold_match(text: str, needle: str) -> tuple[int, int] | None:
    """Locate a casefolded query, returning offsets in the original text.

    Unicode casefolding can expand characters (ß -> ss), so indices from the
    folded copy cannot be used directly to select the visible excerpt.
    """
    index = text.casefold().find(needle)
    if index < 0:
        return None
    if text.isascii():
        return index, len(needle)
    folded_position = 0
    start = 0
    end = index + len(needle)
    for position, character in enumerate(text):
        next_position = folded_position + len(character.casefold())
        if folded_position <= index < next_position:
            start = position
        if next_position >= end:
            return start, position + 1 - start
        folded_position = next_position
    return None


def _iter_search_texts(data: dict):
    if data["summary"]:
        yield "Summary:", data["summary"]
    for item in data["long_term"]:
        yield "Long-term:", item
    for msg in data["messages"]:
        yield f"{msg['role'].capitalize()}:", msg["content"]


def _fit_output(
    lines: list[str], header: str, *, keep_newest: bool = False,
) -> str:
    """Join *lines* under the budget, retaining the useful end of the list."""
    lines = list(lines)
    # Track joined size incrementally.
    total = len(header) + sum(len(line) + 1 for line in lines)
    dropped = 0

    def omission_footer() -> str:
        return (
            f"(…{dropped} more result(s) omitted to fit the limit;"
            " raise *limit* to see more; never treat this preview"
            " as full coverage; say PARTIAL + remainder"
            " when coverage is unclear)"
        )

    while lines and total + (len(omission_footer()) + 1 if dropped else 0) > _OUTPUT_BUDGET:
        total -= len(lines.pop(0 if keep_newest else -1)) + 1
        dropped += 1
    if dropped:
        lines.append(omission_footer())
    result = "\n".join([header, *lines])
    if len(result) > _OUTPUT_BUDGET:
        # A model-supplied query/session name can exceed the whole budget.
        return truncate_with_marker(
            result, _OUTPUT_BUDGET,
            "narrow the query or shorten the session name",
        )
    return result


def _page_offset(args: dict) -> int | None:
    offset = args.get("offset", 0)
    return offset if isinstance(offset, int) and not isinstance(offset, bool) and offset >= 0 else None


def _render_result_page(
    rows: list[str], *, offset: int, total: int, limit: int, action: str,
    header: Callable[[int, bool], str], tail: str = "",
) -> str:
    """Fit complete ordered rows and advance by the rows actually emitted."""
    rows = list(rows)
    while rows:
        shown = len(rows)
        more = offset + shown < total
        parts = [header(shown, more), *rows]
        if more:
            parts.append(
                f"(More {'matches' if action == 'search' else 'older sessions'} remain;"
                f" repeat action={action} with the same query/workspace,"
                f" offset={offset + shown}, and limit={limit}."
                " Restart at offset=0 if memory changes.)"
            )
        if tail:
            parts.append(tail)
        result = "\n".join(parts)
        if len(result) <= _OUTPUT_BUDGET:
            return result
        rows.pop()
    return (
        "Error: no complete result fits the memory output budget; narrow the"
        " query or shorten the session name before retrying."
    )


class MemoryTool(AgentTool):
    name = "memory"
    order = 20
    parallel_safe = True
    description = "Chat memory (sessions, summaries, colony)."
    parameters: ClassVar[dict[str, Any]] = object_parameters(
        {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
            },
            "query": {
                "type": "string",
            },
            "session": {"type": "string"},
            "limit": {
                "type": "integer",
                "description": "Maximum: search 20, list 40, read 100.",
            },
            "offset": {
                "type": "integer", "minimum": 0,
                "description": (
                    "Results already consumed: search matches, listed sessions,"
                    " or newest read messages. Use the returned offset to page."
                ),
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
        try:
            if action == "search":
                return await asyncio.to_thread(
                    self._search, sessions_dir, workspace_path, args,
                )
            if action == "list":
                return await asyncio.to_thread(
                    self._list, sessions_dir, workspace_path, args,
                )
            return await asyncio.to_thread(self._read, sessions_dir, args)
        except _MemoryChangedError:
            return "Error: memory changed while preparing this page; retry with offset=0."

    def _search(self, sessions_dir: str, workspace: str, args: dict) -> str:
        query, error = require_nonempty_string_arg(
            args, "query", strip=True,
        )
        if error:
            return error
        assert query is not None
        offset = _page_offset(args)
        if offset is None:
            return "Error: 'offset' must be a non-negative integer for search"
        limit = coerce_int_arg(
            args.get("limit", _MATCH_LIMIT_DEFAULT),
            default=_MATCH_LIMIT_DEFAULT,
            minimum=1,
            maximum=_MATCH_LIMIT_MAX,
        )
        needle = query.casefold()

        sessions = _load_session_metadata(sessions_dir, needle)
        colony = _colony_items(workspace)
        session_matches = sum(item["match_count"] for item in sessions)
        total = session_matches + sum(needle in item.casefold() for item in colony)

        if not total:
            return (
                f"No matches for '{query}' in any session, summary,"
                " long-term note, or colony item."
            )
        if offset >= total:
            return "No more memory matches remain for this query."

        matches: list[str] = []
        consumed = 0
        for metadata in sessions:
            count = metadata["match_count"]
            if not count:
                continue
            if consumed + count <= offset:
                consumed += count
                continue
            if len(matches) == limit:
                break
            skip = max(0, offset - consumed)
            data = _reload_session(sessions_dir, metadata)
            label = _session_label(data)
            for kind, text in _iter_search_texts(data):
                match = _casefold_match(text, needle)
                if match is None:
                    continue
                if skip:
                    skip -= 1
                    continue
                matches.append(f"- {label} {kind} {_excerpt(text, *match)}")
                if len(matches) == limit:
                    break
            consumed += count
            del data
        if len(matches) < limit:
            skip = max(0, offset - session_matches)
            for item in colony:
                match = _casefold_match(item, needle)
                if match is None:
                    continue
                if skip:
                    skip -= 1
                    continue
                matches.append(f"- [Colony] {_excerpt(item, *match)}")
                if len(matches) == limit:
                    break

        def header(shown: int, more: bool) -> str:
            text = f"Found {total} match(es) for '{query}'"
            if offset or more:
                text += (
                    f" — showing the {shown} newest" if offset == 0 else
                    f" — showing {shown} match(es) after offset {offset}"
                )
            if more:
                text += (
                    " (never treat this preview as full coverage;"
                    " say PARTIAL + remainder when coverage is unclear)"
                )
            return text + ":"

        return _render_result_page(
            matches, offset=offset, total=total, limit=limit,
            action="search", header=header,
        )

    def _list(self, sessions_dir: str, workspace: str, args: dict | None = None) -> str:
        args = args or {}
        offset = _page_offset(args)
        if offset is None:
            return "Error: 'offset' must be a non-negative integer for list"
        limit = coerce_int_arg(
            args.get("limit", _LIST_SESSIONS_CAP), default=_LIST_SESSIONS_CAP,
            minimum=1, maximum=_LIST_SESSIONS_CAP,
        )
        sessions = _load_session_metadata(sessions_dir)
        colony_count = len(_colony_items(workspace))
        if not sessions:
            if colony_count:
                return (
                    "No sessions recorded in this workspace yet."
                    f" Colony: {colony_count} items in .cozter/colony.json"
                )
            return "No sessions recorded in this workspace yet."
        if offset >= len(sessions):
            return "No older sessions remain in this workspace."
        lines: list[str] = []
        for data in sessions[offset:offset + limit]:
            flags = []
            if data["has_summary"]:
                flags.append("summary")
            if data["long_term_count"]:
                flags.append(f"{data['long_term_count']} long-term")
            flag_text = f", {', '.join(flags)}" if flags else ""
            lines.append(
                f"- {data['name']} — id {data['id'][:8]},"
                f" {data['created'][:10] or 'unknown date'},"
                f" {data['message_count']} messages{flag_text}"
            )

        def header(_shown: int, more: bool) -> str:
            return "Sessions (newest first):" + (
                " (preview, not full coverage; say PARTIAL + remainder"
                " when coverage is unclear)" if more else ""
            )

        return _render_result_page(
            lines, offset=offset, total=len(sessions), limit=limit,
            action="list", header=header,
            tail=f"Colony: {colony_count} items in .cozter/colony.json",
        )

    def _read(self, sessions_dir: str, args: dict) -> str:
        target = args.get("session")
        if not isinstance(target, str) or not target.strip():
            return (
                "Error: 'session' is required for read (a name, an id or"
                " unique prefix, or 'last')"
            )
        offset = args.get("offset", 0)
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            return "Error: 'offset' must be a non-negative integer for read"
        count = coerce_int_arg(
            args.get("limit", _READ_MESSAGES_DEFAULT),
            default=_READ_MESSAGES_DEFAULT,
            minimum=1,
            maximum=_READ_MESSAGES_MAX,
        )
        # A full durable id already identifies its file. Reading every other
        # transcript first makes this bounded operation grow with all history.
        # Names, prefixes and newest aliases retain the existing resolver.
        data = None
        session_id = target.strip()
        if (
            session_id.casefold() not in _NEWEST_ALIASES
            and session_id not in {".", ".."}
            and not any(character in session_id for character in "\x00/\\:")
        ):
            try:
                path = resolve_inside_workspace(sessions_dir, f"{session_id}.json")
            except ValueError:
                pass
            else:
                data = _load_session_file(path, session_id)
        error = None
        if data is None:
            selected, error = _find_session(_load_session_metadata(sessions_dir), target)
            if selected is not None:
                data = _reload_session(sessions_dir, selected)
        if error or data is None:
            return error or "Error: session not found"

        total = len(data["messages"])
        end = max(0, total - offset)
        if offset and end == 0:
            return "No older messages remain in this session."
        window = data["messages"][max(0, end - count):end]
        shown_from = end - len(window) + 1
        lines: list[str] = []
        for index, msg in enumerate(window):
            content = msg["content"]
            if len(content) > _LINE_CAP:
                content = content[:_LINE_CAP] + "… [line clipped]"
            lines.append(f"{shown_from + index}. {msg['role'].capitalize()}: {content}")
        while True:
            shown = len(lines)
            shown_text = (
                f"showing last {shown}" if offset == 0 else
                f"showing {shown} older message(s) after offset {offset}"
            )
            header = (
                f"Session: {data['name']} (id {data['id'][:8]},"
                f" {data['created'][:10] or 'unknown date'},"
                f" {total} message(s), {shown_text})"
            )
            if total and not shown:
                # Oversized legacy names can consume the entire budget. Do
                # not return a continuation cursor that cannot make progress.
                return truncate_with_marker(
                    "Error: session metadata leaves no room for a message;"
                    " shorten the session name before retrying.\n" + header,
                    _OUTPUT_BUDGET, "shorten the session name before retrying",
                )
            parts = [header, *lines]
            next_offset = offset + shown
            if shown and next_offset < total:
                parts.append(
                    "(Older messages remain; continue action=read with the"
                    f" same session, offset={next_offset}, and limit={count}."
                    " Offsets count newest messages already consumed;"
                    " restart at offset=0 if messages change. Say"
                    " PARTIAL + remainder if needed history remains unread.)"
                )
            result = "\n".join(parts)
            if len(result) <= _OUTPUT_BUDGET:
                return result
            if not lines:
                return truncate_with_marker(
                    result, _OUTPUT_BUDGET,
                    "shorten the session name before retrying",
                )
            # Keep the newest complete rows in this page. Recompute both the
            # visible count and cursor after each removal, so no row is skipped.
            lines.pop(0)


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
    exact_id = [session for session in sessions if session["id"] == target.strip()]
    if exact_id:
        return exact_id[0], None
    exact_name = [session for session in sessions if session["name"].casefold() == key]
    if len(exact_name) == 1:
        return exact_name[0], None
    prefix = [
        session for session in sessions
        if session["id"].startswith(key) or session["name"].casefold().startswith(key)
    ]
    if len(prefix) == 1:
        return prefix[0], None
    if len(prefix) > 1:
        shown = ", ".join(session["name"] for session in prefix[:5])
        if len(prefix) > 5:
            shown += f", … [{len(prefix) - 5} more match(es) omitted]"
        return None, (
            f"Error: '{target}' matches {len(prefix)} sessions ({shown});"
            " use a longer prefix or the full id"
        )
    return None, (
        f"Error: no session named '{target}'; use action=list to see"
        " sessions"
    )


if __name__ == "__main__":
    MemoryTool.run_as_script()
