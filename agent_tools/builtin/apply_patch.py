"""apply_patch: apply a unified diff to workspace files.

A dependency-free unified-diff applier. Supports modifying, creating
(``--- /dev/null``), and deleting (``+++ /dev/null``) files, with multiple
files and hunks per call. Hunk context is matched exactly first, then with
trailing-whitespace fuzz and a full-file scan, so slightly-stale line
numbers still apply. Each hunk that can't be located is reported rather
than silently dropped.
"""

from __future__ import annotations

import os
import re

from ..base import (
    AgentTool,
    create_text_file_atomically,
    ensure_parent_dir,
    object_parameters,
    resolve_inside_workspace,
    write_text_after_edit,
)


class _PatchError(Exception):
    """The patch text could not be parsed as a unified diff."""


class _Hunk:
    def __init__(
        self,
        start: int,
        old_count: int | None = None,
        new_count: int | None = None,
    ) -> None:
        self.start = start  # 1-based line in the old file (a hint)
        self.old_count = old_count
        self.new_count = new_count
        self.old: list[str] = []  # context + deleted lines (content only)
        self.new: list[str] = []  # context + added lines (content only)
        # A unified diff's "\\ No newline at end of file" marker applies to
        # the immediately preceding body line.  Remember which side of this
        # hunk is the output EOF so apply can preserve the requested newline
        # state instead of inheriting it blindly from the old file.
        self.last_marker: str | None = None
        self.new_ends_with_newline: bool | None = None

    @property
    def complete(self) -> bool:
        """Whether the declared old/new line counts have been consumed."""
        return (
            self.old_count is not None
            and self.new_count is not None
            and len(self.old) == self.old_count
            and len(self.new) == self.new_count
        )

    def validate(self) -> None:
        """Reject a hunk whose body does not match its declared line counts.

        A truncated diff must never be treated as a smaller valid edit: doing
        so can silently apply only the first part of a requested change.  Keep
        count-less fallback headers permissive for backwards compatibility,
        but validate every normal unified-diff hunk before it can reach the
        file writer.
        """
        if self.old_count is None or self.new_count is None:
            return
        if len(self.old) != self.old_count or len(self.new) != self.new_count:
            raise _PatchError(
                "hunk line counts do not match its header "
                f"(expected -{self.old_count}/+{self.new_count}, got "
                f"-{len(self.old)}/+{len(self.new)})"
            )


class _FilePatch:
    def __init__(self, old_path: str | None, new_path: str | None) -> None:
        self.old_path = old_path  # None == /dev/null (creation)
        self.new_path = new_path  # None == /dev/null (deletion)
        self.hunks: list[_Hunk] = []

    def validate(self) -> None:
        """Reject patch forms that cannot apply to their declared source."""
        if self.old_path is not None:
            return
        for hunk in self.hunks:
            # A creation diff's old side is /dev/null, so it cannot contain
            # a context or deletion line.  Previously those lines were simply
            # discarded while the new side was written, turning a malformed
            # diff into a different successful file creation.
            if hunk.old:
                raise _PatchError(
                    "creation hunk contains old/context lines for /dev/null",
                )
            if hunk.old_count not in (None, 0):
                raise _PatchError(
                    "creation hunk must declare zero old-file lines",
                )
            if hunk.old_count == 0 and hunk.start != 0:
                raise _PatchError(
                    "creation hunk must start at old-file line 0",
                )


class ApplyPatchTool(AgentTool):
    name = "apply_patch"
    file_action = "edit"
    order = 20  # group with the editing tools
    description = (
        "Apply a unified diff (as produced by `git diff` or `diff -u`) to"
        " the workspace. Prefer this for multi-hunk or multi-file edits."
        " Context lines are matched with small fuzz, so exact line numbers"
        " aren't required. Supports creating files (`--- /dev/null`) and"
        " deleting them (`+++ /dev/null`). Reports the outcome per file;"
        " a hunk whose context can't be found is reported, not skipped"
        " silently."
    )
    parameters = object_parameters(
        {"patch": {
            "type": "string",
            "description": "The unified diff text.",
        }},
        ["patch"],
    )

    async def run(self, workspace_path: str, args: dict) -> str:
        patch = args.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            return "Error: 'patch' must be a non-empty unified diff"
        try:
            file_patches = _parse_patch(patch)
        except _PatchError as exc:
            return f"Error: could not parse patch: {exc}"
        if not file_patches:
            return (
                "Error: no hunks found; expected unified-diff headers"
                " (--- / +++ / @@)"
            )
        return "\n".join(
            _apply_file_patch(workspace_path, fp) for fp in file_patches
        )

    def summarize(self, args: dict) -> str:
        patch = args.get("patch")
        if not isinstance(patch, str):
            return "apply_patch"
        n = patch.count("\n+++ ") + (1 if patch.startswith("+++ ") else 0)
        return f"apply_patch ({n} file{'s' if n != 1 else ''})"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _strip_git_prefix(path: str) -> str:
    return path[2:] if path.startswith(("a/", "b/")) else path


def _header_path(raw: str) -> str | None:
    raw = raw.split("\t", 1)[0].strip()
    return None if raw == "/dev/null" else _strip_git_prefix(raw)


def _parse_hunk_header(
    header: str,
) -> tuple[int, int | None, int | None]:
    """Return the old start and declared old/new counts from a hunk header."""
    match = re.match(
        r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,(\d+))? @@",
        header,
    )
    if match is None:
        return 1, None, None
    try:
        start = int(match.group(1))
        old_count = int(match.group(2) or "1")
        new_count = int(match.group(3) or "1")
    except ValueError as exc:
        raise _PatchError("hunk header contains an invalid line count") from exc
    return start, old_count, new_count


def _parse_patch(text: str) -> list[_FilePatch]:
    patches: list[_FilePatch] = []
    current: _FilePatch | None = None
    hunk: _Hunk | None = None
    pending_old: str | None = None

    for line in text.splitlines():
        if line == "\\ No newline at end of file":
            if hunk is None or hunk.last_marker is None:
                raise _PatchError(
                    "no-newline marker does not follow a hunk body line",
                )
            # The marker is emitted only for a final old/new line.  A marker
            # following a deletion means the replacement side ends with a
            # newline; one following an addition/context means it does not.
            hunk.new_ends_with_newline = hunk.last_marker == "-"
            hunk.last_marker = None
            continue
        # File-header-looking content is legal inside a hunk: deleting a line
        # that starts with ``--`` produces ``--- ...`` in the diff, and adding
        # one that starts with ``++`` produces ``+++ ...``. Only recognize the
        # next file header once the current hunk's declared counts are full.
        if hunk is not None and hunk.complete:
            if (
                line.startswith((" ", "+", "-"))
                and not line.startswith(("--- ", "+++ "))
            ):
                raise _PatchError(
                    "hunk contains more body lines than its header declares",
                )
            hunk = None
        if line.startswith("--- "):
            if hunk is not None:
                hunk.old.append(line[1:])
                hunk.last_marker = "-"
                continue
            pending_old = _header_path(line[4:])
            hunk = None
            continue
        if line.startswith("+++ "):
            if hunk is not None:
                hunk.new.append(line[1:])
                hunk.last_marker = "+"
                continue
            current = _FilePatch(pending_old, _header_path(line[4:]))
            patches.append(current)
            pending_old = None
            hunk = None
            continue
        if line.startswith("@@"):
            if current is None:
                raise _PatchError("hunk (@@) before any file header")
            if hunk is not None:
                hunk.validate()
            start, old_count, new_count = _parse_hunk_header(line)
            hunk = _Hunk(start, old_count, new_count)
            current.hunks.append(hunk)
            continue
        if hunk is None:
            continue  # preamble / "diff --git" / "index" lines
        if not line:
            # A bare empty line is an empty context line (some emitters drop
            # the leading space).
            hunk.old.append("")
            hunk.new.append("")
            hunk.last_marker = " "
            continue
        marker, content = line[0], line[1:]
        if marker == " ":
            hunk.old.append(content)
            hunk.new.append(content)
        elif marker == "-":
            hunk.old.append(content)
        elif marker == "+":
            hunk.new.append(content)
        else:
            hunk.validate()
            hunk = None  # a non-body line ends this hunk
            continue
        hunk.last_marker = marker

    if hunk is not None:
        hunk.validate()

    parsed = [p for p in patches if p.hunks]
    for file_patch in parsed:
        file_patch.validate()
    return parsed


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------


def _apply_file_patch(workspace_path: str, fp: _FilePatch) -> str:
    label = fp.new_path or fp.old_path or "?"

    # Deletion: +++ /dev/null
    if fp.new_path is None and fp.old_path is not None:
        try:
            target = resolve_inside_workspace(workspace_path, fp.old_path)
        except ValueError as exc:
            return f"{fp.old_path}: skipped ({exc})"
        if not os.path.isfile(target):
            return f"{fp.old_path}: already absent"
        try:
            file_lines, _had_nl, _crlf = _read_file_lines(target)
        except UnicodeDecodeError:
            return f"{fp.old_path}: not valid UTF-8 (binary?); not deleted"
        applied = _apply_hunks(file_lines, fp.hunks)
        if isinstance(applied, str):
            return f"{fp.old_path}: {applied}; not deleted"
        if applied:
            return f"{fp.old_path}: deletion patch did not remove all content"
        os.remove(target)
        return f"{fp.old_path}: deleted"

    if fp.new_path is None:
        return f"{label}: skipped (no target path)"

    try:
        target = resolve_inside_workspace(workspace_path, fp.new_path)
    except ValueError as exc:
        return f"{fp.new_path}: skipped ({exc})"

    # Creation: --- /dev/null
    if fp.old_path is None:
        new_lines: list[str] = []
        for h in fp.hunks:
            new_lines.extend(h.new)
        ensure_parent_dir(target)
        out = "\n".join(new_lines)
        if new_lines and _new_file_ends_with_newline(fp.hunks, default=True):
            out += "\n"
        # Do not turn a concurrent creator into a silent overwrite.  The
        # no-clobber helper also keeps failed writes from exposing a partial
        # new source file.
        if not create_text_file_atomically(target, out, uses_crlf=False):
            return f"{fp.new_path}: skipped (file already exists)"
        return f"{fp.new_path}: created ({len(new_lines)} lines)"

    # Modification
    if not os.path.isfile(target):
        return f"{fp.new_path}: skipped (file not found)"
    try:
        file_lines, had_nl, uses_crlf = _read_file_lines(target)
    except UnicodeDecodeError:
        return f"{fp.new_path}: not valid UTF-8 (binary?); not patched"

    applied = _apply_hunks(file_lines, fp.hunks)
    if isinstance(applied, str):
        return f"{fp.new_path}: {applied}"

    out = "\n".join(applied)
    if applied and _new_file_ends_with_newline(fp.hunks, default=had_nl):
        out += "\n"
    write_text_after_edit(target, out, uses_crlf=uses_crlf)
    return f"{fp.new_path}: applied {len(fp.hunks)} hunk(s)"


def _read_file_lines(path: str) -> tuple[list[str], bool, bool]:
    """Read a text file into patch lines; preserve final-newline & CRLF bits.

    Reads bytes and decodes strict UTF-8 so a binary/non-UTF-8 file raises
    ``UnicodeDecodeError`` (caught by the caller) instead of being silently
    rewritten with every undecodable byte replaced by U+FFFD. ``\\r\\n`` is
    normalized to ``\\n`` for hunk matching; the returned ``uses_crlf`` flag
    lets the writer restore the original line endings.
    """
    with open(path, "rb") as f:
        raw = f.read()
    content = raw.decode("utf-8")  # UnicodeDecodeError on binary -> caller
    uses_crlf = "\r\n" in content
    if uses_crlf:
        content = content.replace("\r\n", "\n")
    had_nl = content.endswith("\n")
    lines = content.split("\n")
    if had_nl and lines and lines[-1] == "":
        lines.pop()  # drop the trailing "" left by the final newline
    return lines, had_nl, uses_crlf


def _new_file_ends_with_newline(hunks: list[_Hunk], *, default: bool) -> bool:
    """Return the output EOF-newline state requested by the latest hunk."""
    for hunk in reversed(hunks):
        if hunk.new_ends_with_newline is not None:
            return hunk.new_ends_with_newline
    return default


def _apply_hunks(lines: list[str], hunks: list[_Hunk]) -> list[str] | str:
    result = list(lines)
    for idx, hunk in enumerate(hunks, 1):
        pos = _locate(result, hunk)
        if pos is None:
            return f"hunk {idx} did not apply (context not found)"
        result[pos:pos + len(hunk.old)] = hunk.new
    return result


def _locate(lines: list[str], hunk: _Hunk) -> int | None:
    old = hunk.old
    if not old:
        # Pure insertion: use the start hint, clamped in-range.
        return min(max(hunk.start - 1, 0), len(lines))
    n, m = len(lines), len(old)
    if m > n:
        return None
    hint = min(max(hunk.start - 1, 0), n - m)
    # Exact match: hint first, then a full scan.
    if lines[hint:hint + m] == old:
        return hint
    for p in range(n - m + 1):
        if lines[p:p + m] == old:
            return p
    # Fuzzy match: ignore trailing whitespace.
    old_stripped = [s.rstrip() for s in old]
    for p in [hint, *range(n - m + 1)]:
        if [s.rstrip() for s in lines[p:p + m]] == old_stripped:
            return p
    return None
