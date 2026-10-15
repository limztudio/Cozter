"""Plugin: read-only Git inspection of the workspace repository.

HTTP backends (llama, meta, zai, ...) work through Cozter's typed tools
and have no shell outside ``full`` permission, so they cannot ask the
repository for its own state. This plugin exposes the read-only
questions agents ask most - what changed, what happened recently, what
does the patch look like, which branches/tags/stashes exist, who changed
a line, and what a revision contains - with a fixed, read-only argv the
model cannot extend.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

from ..base import (
    AgentTool,
    coerce_int_arg,
    object_parameters,
    resolve_workspace_entry,
)
from ...utils import clip_status_value
from ._git_common import (
    GitFailedError as _GitFailedError,
    MAX_OUTPUT_CHARS,
    _GitTextWindow,
    add_common_git_flags,
    append_stderr_note as _append_stderr_note,
    bounded as _bounded,
    check_ref as _checked_ref,
    finish_git_run_error as _finish_git_run_error,
    first_stderr_line,
    git_once as _git_once,
)

# Real-work cap; shared limits/validation live in ._git_common.
_ACTIONS = (
    "status",
    "log",
    "diff",
    "branches",
    "tags",
    "stashes",
    "show",
    "blame",
)
_PAGE_CONTENT_CHARS = MAX_OUTPUT_CHARS - 1000


def _output_offset(args: dict) -> int:
    offset = args.get("offset", 0)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("'offset' must be a non-negative integer character offset")
    return offset


def _page_output(window: _GitTextWindow, stderr: str) -> str:
    text = window.text()
    end = window.offset + len(text)
    more = end < window.total_chars
    if not window.offset and not more and not window.incomplete:
        return _bounded(_append_stderr_note(text.strip(), stderr))
    if window.offset > window.total_chars:
        location = f"offset {window.offset} is beyond {window.total_chars} output characters"
    else:
        location = f"characters {window.offset}:{end} of {window.total_chars}"
    footer = f"\n[git_info output {location}; "
    if more:
        footer += (
            f"truncated page; next_offset={end}. Repeat the same request"
            f" with offset={end} to fetch the remainder; never treat this"
            " preview as full content; say PARTIAL + remainder while output remains. "
        )
    elif window.incomplete:
        footer += "output is incomplete because a child kept a pipe open. "
    else:
        footer += "end of output. "
    if window.incomplete and more:
        footer += "A child kept a pipe open; captured output may be incomplete. "
    footer += (
        "Combine pages only while the repository is unchanged;"
        " restart at offset=0 after changes.]"
    )
    return _bounded(_append_stderr_note(text, stderr) + footer)


class GitInfoTool(AgentTool):
    name = "git_info"
    order = 20
    max_result_chars = MAX_OUTPUT_CHARS
    parallel_safe = True
    description = (
        "Git status/log/diff/branches/tags/stashes/show/blame. Read-only."
        " Long output is paged by character offset; follow next_offset with"
        " the same request, restarting at zero if the repository changes."
    )
    parameters: ClassVar[dict[str, Any]] = object_parameters(
        {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
            },
            "path": {"type": "string"},
            "ref": {
                "type": "string",
                "description": "Revision for show/blame/log (branch, tag, or SHA).",
            },
            "patch": {
                "type": "boolean",
                "description": "Full patch.",
            },
            "limit": {
                "type": "integer",
                "description": "max 50.",
            },
            "line": {
                "type": "integer",
                "description": "1-based line for blame; needs path.",
            },
            "line_count": {
                "type": "integer",
                "description": "Blame range length (1–200); starts at line or 1.",
            },
            "offset": {
                "type": "integer",
                "description": "Zero-based output character offset; use next_offset from the previous page.",
            },
        },
        ["action"],
    )

    async def run(self, workspace_path: str, args: dict) -> str:
        action = args.get("action")
        if not isinstance(action, str) or action not in _ACTIONS:
            return (
                "Error: 'action' must be one of "
                + ", ".join(_ACTIONS)
            )

        try:
            argv = self._build_argv(workspace_path, action, args)
            offset = _output_offset(args)
        except ValueError as exc:
            return f"Error: {exc}"

        try:
            window = _GitTextWindow(offset, _PAGE_CONTENT_CHARS)
            stdout, stderr = await self._run_git(
                argv, workspace_path, stdout_window=window,
            )
        except (FileNotFoundError, TimeoutError, _GitFailedError) as exc:
            return _finish_git_run_error(action, exc)

        text = stdout.strip()
        if not text and not offset and not window.total_chars and not window.incomplete:
            if action == "log":
                return "No commits yet."
            if action == "diff":
                return "No changes."
            if action == "tags":
                return "No tags."
            if action == "stashes":
                return "No stashes."
            return "(no output)"
        return _page_output(window, stderr)

    @staticmethod
    def _build_argv(
        workspace_path: str, action: str, args: dict,
    ) -> list[str]:
        """Build the fixed read-only argv; raise ValueError for bad args."""
        argv: list[str] = ["git", "-C", workspace_path]
        if action == "status":
            return argv + ["status", "--short", "--branch"]
        if action == "log":
            limit = coerce_int_arg(
                args.get("limit", 10), default=10, minimum=1, maximum=50,
            )
            tail: list[str] = [
                "log", "--no-ext-diff", "--no-textconv", "--oneline",
                f"-n{limit}",
            ]
            ref = args.get("ref")
            if isinstance(ref, str) and ref.strip():
                tail.append(_checked_ref(ref.strip()))
            return argv + tail
        if action == "diff":
            # Repository-configured diff/textconv commands can execute arbitrary
            # programs even for a read-only patch request. Use Git's own diff.
            tail = ["diff", "--no-ext-diff", "--no-textconv"]
            if args.get("patch") is True:
                tail.append("HEAD")
            else:
                # ``--stat`` covers staged and unstaged work.
                tail += ["--stat", "HEAD"]
            pathspec = args.get("path")
            if isinstance(pathspec, str) and pathspec.strip():
                tail += [
                    "--",
                    resolve_workspace_entry(
                        workspace_path, pathspec.strip(),
                    ),
                ]
            return argv + tail
        if action == "branches":
            return argv + ["branch", "-vv"]
        if action == "tags":
            limit = coerce_int_arg(
                args.get("limit", 50), default=50, minimum=1, maximum=50,
            )
            return argv + [
                "for-each-ref", f"--count={limit}", "--sort=-creatordate",
                "--format=%(refname:strip=2) %(contents:subject)", "refs/tags/",
            ]
        if action == "stashes":
            return argv + ["stash", "list"]
        if action == "show":
            ref = args.get("ref")
            revision = (
                _checked_ref(ref.strip())
                if isinstance(ref, str) and ref.strip()
                else "HEAD"
            )
            return argv + [
                "show", "--no-ext-diff", "--no-textconv", "--stat",
                "--oneline", revision,
            ]
        pathspec = args.get("path")
        if not isinstance(pathspec, str) or not pathspec.strip():
            raise ValueError("'blame' needs 'path' (a workspace file)")
        resolved = resolve_workspace_entry(workspace_path, pathspec.strip())
        line = args.get("line")
        ref = args.get("ref")
        revision = (
            _checked_ref(ref.strip())
            if isinstance(ref, str) and ref.strip()
            else "HEAD"
        )
        line_count = args.get("line_count")
        if line_count is not None:
            if (
                isinstance(line_count, bool) or not isinstance(line_count, int)
                or not 1 <= line_count <= 200
            ):
                raise ValueError("'line_count' must be an integer between 1 and 200")
        if (line is None or line is False) and line_count is None:
            return argv + ["blame", "--no-textconv", revision, "--", resolved]
        number = coerce_int_arg(line, default=1, minimum=1, maximum=1_000_000)
        last = number + (line_count or 1) - 1
        return argv + [
            "blame", "--no-textconv", "-L", f"{number},{last}",
            revision, "--", resolved,
        ]

    @staticmethod
    async def _run_git(
        argv: list[str], workspace: str,
        *, stdout_window: _GitTextWindow | None = None,
    ) -> tuple[str, str]:
        """Run a fixed read-only git argv and return decoded output.

        The argv is built here from constants plus the caller's workspace,
        never from model text, so no model-supplied flag can turn the
        snapshot into a mutation. ``GIT_OPTIONAL_LOCKS=0`` and the ``-c``
        switches keep the run strictly passive: no index-refresh lock, no
        fsmonitor daemon spawn, and no locale-dependent path quoting.
        """
        is_diff = len(argv) > 3 and argv[3] == "diff"
        env = {
            **os.environ, "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "0",
        }
        argv = add_common_git_flags(argv)
        capture = {"stdout_window": stdout_window} if stdout_window is not None else {}
        stdout, stderr, returncode = await _git_once(argv, workspace, env, **capture)
        if returncode != 0 and is_diff and "HEAD" in argv:
            head_failure = any(
                marker in stderr.casefold()
                for marker in ("unknown revision", "ambiguous argument 'head'")
            )
            if head_failure:
                # Zero-commit repo has no HEAD; diff against the index instead.
                rebuilt = [
                    "--cached" if arg == "HEAD" else arg for arg in argv
                ]
                stdout, stderr, returncode = await _git_once(rebuilt, workspace, env, **capture)
        if returncode != 0:
            raise _GitFailedError(first_stderr_line(stderr))
        return stdout, stderr

    def summarize(self, args: dict) -> str:
        action = args.get("action") if isinstance(args, dict) else None
        path = args.get("path") if isinstance(args, dict) else None
        ref = args.get("ref") if isinstance(args, dict) else None
        detail = path if isinstance(path, str) and path else (
            ref if isinstance(ref, str) and ref else ""
        )
        suffix = f" ({clip_status_value(detail)})" if detail else ""
        return f"git {clip_status_value(action or '?', 40)}{suffix}"


if __name__ == "__main__":
    GitInfoTool.run_as_script()
