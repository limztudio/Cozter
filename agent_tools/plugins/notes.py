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
    write_text_after_edit,
)

_NOTES_RELPATH = ".cozter/notes.md"

# Total size ceiling. Appending past it trims the oldest entries, so the
# tool can never strand the model with a "notes full" dead end.
_NOTES_MAX_BYTES = 64 * 1024
# After a trim, keep roughly this much of the newest material.
_NOTES_KEEP_BYTES = 32 * 1024
# One appended entry is capped so a runaway argument cannot bypass the
# trim budget (UTF-8 worst case is 4 bytes per character).
_MAX_ENTRY_CHARS = 2_000
# read() returns the tail of the notes, sized so execute_tool's 4,000
# character result cap still shows the newest entry rather than hiding it.
_READ_TAIL_CHARS = 3_500


class NotesTool(AgentTool):
    name = "notes"
    order = 20  # pair with read_file in the model-facing ordering
    description = (
        "Persistent workspace notes (.cozter/notes.md) surviving"
        " compaction and restarts. append needs text; read shows the"
        " newest; clear deletes all. Record progress/next steps here."
    )
    parameters: ClassVar[dict[str, Any]] = object_parameters(
        {
            "action": {
                "type": "string",
                "enum": ["append", "read", "clear"],
                "description": (
                    "append (needs text), read, or clear."
                ),
            },
            "text": {
                "type": "string",
                "description": (
                    "Entry text (plain text or short markdown)."
                ),
            },
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
            target = resolve_inside_workspace(
                workspace_path, _NOTES_RELPATH,
            )
        except ValueError as exc:
            return f"Error: {exc}"

        if action == "clear":
            return await asyncio.to_thread(self._clear, target)
        if action == "read":
            return await asyncio.to_thread(self._read, target)
        return await asyncio.to_thread(self._append, target, args)

    def _append(self, target: str, args: dict) -> str:
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            return "Error: 'text' must be a non-empty string for append"
        text = text.strip()[:_MAX_ENTRY_CHARS]
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"## {stamp}\n{text}\n\n"

        existing = _read_notes_text(target)
        combined = _fit_entries([existing, entry] if existing else [entry])
        ensure_parent_dir(target)
        try:
            write_text_after_edit(target, combined, uses_crlf=False)
        except OSError as exc:
            return f"Error: could not write notes: {exc}"
        return f"Noted ({len(text)} chars). Total notes: {len(combined)} chars."

    def _read(self, target: str) -> str:
        text = _read_notes_text(target)
        if not text:
            return (
                "Notes are empty. Use action=append to record progress,"
                " findings, and next steps."
            )
        if len(text) > _READ_TAIL_CHARS:
            omitted = len(text) - _READ_TAIL_CHARS
            text = (
                f"[{omitted} older characters omitted;"
                " use action=clear once stale]\n" + text[-_READ_TAIL_CHARS:]
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
            preview = text.strip()[:80]
            return f"notes append: {preview}" + (
                "..." if len(text.strip()) > 80 else ""
            )
        return f"notes {action or '?'}"


def _read_notes_text(target: str) -> str:
    """Return the notes tail, tolerating a missing file.

    Reading from the end matters: the newest entries live at the tail, so
    an oversized legacy file must not push them out of the window that a
    later append sees (a head read would silently drop them).
    """
    try:
        with open(target, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - (_NOTES_MAX_BYTES + 1)))
            raw = f.read()
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace")


def _fit_entries(parts: list[str]) -> str:
    """Combine *parts* and trim to the newest entries within budget.

    Entries are ``## <timestamp>`` headed blocks. Trimming walks the
    entries from newest to oldest and keeps as many whole entries as the
    keep budget allows, so a trim never severs an entry mid-line.
    """
    combined = "".join(parts)
    if len(combined.encode("utf-8")) <= _NOTES_MAX_BYTES:
        return combined

    entries: list[str] = []
    current: list[str] = []
    for line in combined.split("\n"):
        if line.startswith("## ") and current:
            entries.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        entries.append("\n".join(current))

    kept: list[str] = []
    used = 0
    for entry in reversed(entries):
        size = len(entry.encode("utf-8", errors="replace")) + 1
        if kept and used + size > _NOTES_KEEP_BYTES:
            break
        kept.append(entry)
        used += size
    kept.reverse()
    # Entries already carry their trailing blank line, so plain
    # concatenation reproduces the append format exactly.
    return "".join(kept)


if __name__ == "__main__":
    NotesTool.run_as_script()
