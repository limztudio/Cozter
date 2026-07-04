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

from ..base import (
    AgentTool,
    ensure_parent_dir,
    object_parameters,
    resolve_inside_workspace,
)


class _PatchError(Exception):
    """The patch text could not be parsed as a unified diff."""


class _Hunk:
    def __init__(self, start: int) -> None:
        self.start = start  # 1-based line in the old file (a hint)
        self.old: list[str] = []  # context + deleted lines (content only)
        self.new: list[str] = []  # context + added lines (content only)


class _FilePatch:
    def __init__(self, old_path: str | None, new_path: str | None) -> None:
        self.old_path = old_path  # None == /dev/null (creation)
        self.new_path = new_path  # None == /dev/null (deletion)
        self.hunks: list[_Hunk] = []


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


def _parse_hunk_start(header: str) -> int:
    # @@ -oldStart,oldCount +newStart,newCount @@
    try:
        old = header.split("-", 1)[1].split(" ", 1)[0]
        return int(old.split(",", 1)[0])
    except (IndexError, ValueError):
        return 1


def _parse_patch(text: str) -> list[_FilePatch]:
    patches: list[_FilePatch] = []
    current: _FilePatch | None = None
    hunk: _Hunk | None = None
    pending_old: str | None = None

    for line in text.splitlines():
        if line.startswith("--- "):
            pending_old = _header_path(line[4:])
            hunk = None
            continue
        if line.startswith("+++ "):
            current = _FilePatch(pending_old, _header_path(line[4:]))
            patches.append(current)
            pending_old = None
            hunk = None
            continue
        if line.startswith("@@"):
            if current is None:
                raise _PatchError("hunk (@@) before any file header")
            hunk = _Hunk(_parse_hunk_start(line))
            current.hunks.append(hunk)
            continue
        if hunk is None:
            continue  # preamble / "diff --git" / "index" lines
        if line.startswith("\\"):
            continue  # "\ No newline at end of file"
        if not line:
            # A bare empty line is an empty context line (some emitters drop
            # the leading space).
            hunk.old.append("")
            hunk.new.append("")
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
            hunk = None  # a non-body line ends this hunk

    return [p for p in patches if p.hunks]


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
        if os.path.isfile(target):
            os.remove(target)
            return f"{fp.old_path}: deleted"
        return f"{fp.old_path}: already absent"

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
        with open(target, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))
            if new_lines:
                f.write("\n")
        return f"{fp.new_path}: created ({len(new_lines)} lines)"

    # Modification
    if not os.path.isfile(target):
        return f"{fp.new_path}: skipped (file not found)"
    with open(target, encoding="utf-8", errors="replace") as f:
        content = f.read()
    had_nl = content.endswith("\n")
    file_lines = content.split("\n")
    if had_nl and file_lines and file_lines[-1] == "":
        file_lines.pop()  # drop the trailing "" left by the final newline

    applied = _apply_hunks(file_lines, fp.hunks)
    if isinstance(applied, str):
        return f"{fp.new_path}: {applied}"

    out = "\n".join(applied)
    if had_nl:
        out += "\n"
    with open(target, "w", encoding="utf-8") as f:
        f.write(out)
    return f"{fp.new_path}: applied {len(fp.hunks)} hunk(s)"


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
