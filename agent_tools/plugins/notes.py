"""Plugin: persistent workspace notes the agent can read and append to.

Notes live at ``.cozter/notes.md`` inside the workspace and survive
compaction, ``/stop``, and restarts - unlike conversation memory, which
keeps only recent raw messages. Use them to jot down where a task
stands before ending a turn, so any later turn (this session, a resumed
one, or a fresh session in the same workspace) can pick the work back
up. The directory name matches Cozter's runtime-data convention and is
skipped by the discovery tools, so notes never pollute grep/glob/tree
output.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any, ClassVar

from ..base import (
    AgentTool,
    ensure_parent_dir,
    object_parameters,
    resolve_inside_workspace,
    resolve_workspace_entry,
    write_text_after_edit,
)
from ..owned_worker import run_owned_worker
from ...utils import clip_status_value

_NOTES_RELPATH = ".cozter/notes.md"

_NOTES_MAX_BYTES = 64 * 1024
_NOTES_KEEP_BYTES = 32 * 1024
_MAX_ENTRY_CHARS = 2_000
_READ_TAIL_CHARS = 3_500


class NotesTool(AgentTool):
    name = "notes"
    order = 20
    description = "Workspace notes (survive compaction)."
    parameters: ClassVar[dict[str, Any]] = object_parameters(
        {
            "action": {
                "type": "string",
                "enum": ["append", "read", "clear"],
            },
            "text": {"type": "string"},
        },
        ["action"],
    )

    async def run(self, workspace_path: str, args: dict) -> str:
        action = args.get("action")
        if not isinstance(action, str) or action not in {
            "append", "read", "clear",
        }:
            return "Error: 'action' must be one of append, read, clear"

        try:
            # Clearing deletes the named notes entry. Content operations may
            # follow an allowed symlink, but clear must preserve its target.
            resolver = (
                resolve_workspace_entry if action == "clear"
                else resolve_inside_workspace
            )
            target = resolver(
                workspace_path, _NOTES_RELPATH,
            )
        except ValueError as exc:
            return f"Error: {exc}"

        if action == "clear":
            return await run_owned_worker(self._clear, target)
        if action == "read":
            return await asyncio.to_thread(self._read, target)
        return await run_owned_worker(self._append, target, args)

    def _append(self, target: str, args: dict) -> str:
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            return "Error: 'text' must be a non-empty string for append"
        raw_text = text.strip()
        clipped = len(raw_text) > _MAX_ENTRY_CHARS
        text = raw_text[:_MAX_ENTRY_CHARS]
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"## {stamp}\n{text}\n\n"

        try:
            existing = _read_notes_text(target)
        except OSError as exc:
            # A failed read is not an empty file: replacing it would silently
            # discard durable notes that this call never inspected.
            return f"Error: could not read notes: {exc}"
        pre_fit = (existing + entry) if existing else entry
        combined = _fit_entries([existing, entry] if existing else [entry])
        trimmed = combined != pre_fit
        ensure_parent_dir(target)
        try:
            write_text_after_edit(target, combined, uses_crlf=False)
        except OSError as exc:
            return f"Error: could not write notes: {exc}"
        note = f"Noted ({len(text)} chars). Total notes: {len(combined)} chars."
        if clipped:
            note += (
                f" Entry clipped to {_MAX_ENTRY_CHARS} chars;"
                " remainder omitted — never treat this preview as full"
                " content; say PARTIAL + remainder when coverage is unclear."
            )
        if trimmed:
            note += " Oldest entries were trimmed to fit the budget."
        return note

    def _read(self, target: str) -> str:
        try:
            text = _read_notes_text(target)
        except OSError as exc:
            return f"Error: could not read notes: {exc}"
        if not text:
            return (
                "Notes are empty. Use action=append to record progress,"
                " findings, and next steps."
            )
        if len(text) > _READ_TAIL_CHARS:
            omitted = len(text) - _READ_TAIL_CHARS
            text = (
                f"[{omitted} older characters omitted — newest-tail"
                " preview only, not full coverage; say PARTIAL +"
                " remainder when coverage is unclear]\n"
                + text[-_READ_TAIL_CHARS:]
            )
        return text.rstrip() + "\n(End of notes.)"

    def _clear(self, target: str) -> str:
        try:
            os.unlink(target)
        except FileNotFoundError:
            return "Notes are already empty."
        except OSError as exc:
            return f"Error: could not clear notes: {exc}"
        return "Notes cleared."

    def summarize(self, args: dict) -> str:
        action = args.get("action") if isinstance(args, dict) else None
        text = args.get("text") if isinstance(args, dict) else None
        if action == "append" and isinstance(text, str) and text:
            return f"notes append: {clip_status_value(text.strip(), 80)}"
        return f"notes {clip_status_value(action or '?', 40)}"


def _read_notes_text(target: str) -> str:
    """Return the notes tail, tolerating a missing file.

    Reading from the end matters: the newest entries live at the tail, so
    an oversized legacy file must not push them out of the window that a
    later append sees (a head read would silently drop them).
    """
    try:
        with open(target, "rb") as file_handle:
            file_handle.seek(0, os.SEEK_END)
            size = file_handle.tell()
            file_handle.seek(max(0, size - (_NOTES_MAX_BYTES + 1)))
            # The file may grow after seek/tell; keep the read bounded too.
            raw = file_handle.read(_NOTES_MAX_BYTES + 1)
    except FileNotFoundError:
        return ""
    return raw.decode("utf-8", errors="replace")


def _fit_entries(parts: list[str]) -> str:
    """Combine *parts* and trim to the newest entries within budget.

    Entries are ``## <timestamp>`` headed blocks. Trimming walks the
    entries from newest to oldest and keeps as many whole entries as the
    keep budget allows, so a trim never severs an entry mid-line.
    Byte sizes are accumulated once while splitting (entries are
    substrings of *combined*, so each already-known size is reused in
    the keep pass instead of re-encoding every entry).
    """
    combined = "".join(parts)
    combined_bytes = combined.encode("utf-8")
    if len(combined_bytes) <= _NOTES_MAX_BYTES:
        return combined

    # Byte-split: one encode sizes every entry.
    entries: list[tuple[str, int]] = []
    current_lines: list[bytes] = []
    current_size = 0
    for raw_line in combined_bytes.splitlines(keepends=True):
        is_header = raw_line.startswith(b"## ")
        if is_header and current_lines:
            text = b"".join(current_lines).decode(
                "utf-8", errors="replace",
            )
            entries.append((text, current_size))
            current_lines = []
            current_size = 0
        current_lines.append(raw_line)
        current_size += len(raw_line)
    if current_lines:
        text = b"".join(current_lines).decode(
            "utf-8", errors="replace",
        )
        entries.append((text, current_size))

    kept: list[str] = []
    used = 0
    for text, size in reversed(entries):
        if kept and used + size > _NOTES_KEEP_BYTES:
            break
        kept.append(text)
        used += size
    kept.reverse()
    # Entries carry trailing blanks; concat reproduces append format.
    return "".join(kept)


if __name__ == "__main__":
    NotesTool.run_as_script()
