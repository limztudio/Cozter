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

import asyncio
import os
import re
from contextlib import suppress
from typing import Any, ClassVar

from ..base import (
    AgentTool,
    coerce_int_arg,
    object_parameters,
    resolve_inside_workspace,
    truncate_with_marker,
)
from ...utils import clip_status_value

# Real-work cap: git runs up to 3600s (under the tool runner cap) or until cancelled.
_MAX_OUTPUT_CHARS = 12_000
_MAX_GIT_ERROR_CHARS = 500
# Read-only actions only: local writes live in git_ops, network sync in
# git_sync. The docstring order matches the action order below.
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


class _GitFailed(Exception):
    """Git exited non-zero; carries the model-facing stderr excerpt."""


_REF_RE = re.compile(r"^[A-Za-z0-9._\-/]+$")


def _checked_ref(name: str) -> str:
    """Validate a revision token so no model flag can enter the argv."""
    if len(name) > 200 or not _REF_RE.match(name):
        raise ValueError(f"invalid ref {name!r}")
    for marker in (
        "..", "@{", "~", "^", ":", "?", "*", "[", "\\",
        "'", '"', "`", "$", "(", ")", "|", ";", "&", "<", ">",
        "!", " ", "\t", "\n",
    ):
        if marker in name:
            raise ValueError(f"invalid ref {name!r}")
    if name.startswith(("-", "/", ".")) or name.endswith(("/", ".", ".lock")):
        raise ValueError(f"invalid ref {name!r}")
    return name


class GitInfoTool(AgentTool):
    name = "git_info"
    order = 20  # utility tools group
    description = (
        "Git status/log/diff/branches/tags/stashes/show/blame. Read-only."
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
        except ValueError as exc:
            return f"Error: {exc}"

        try:
            stdout, stderr = await self._run_git(argv, workspace_path)
        except FileNotFoundError:
            return "Error: git is not installed or not on PATH"
        except TimeoutError:
            return f"Error: git {action} timed out after 15s"
        except _GitFailed as exc:
            return f"Error: git {action}: {exc}"

        text = stdout.strip()
        if not text:
            if action == "log":
                return "No commits yet."
            if action == "diff":
                return "No changes."
            if action == "tags":
                return "No tags."
            if action == "stashes":
                return "No stashes."
            return "(no output)"
        if stderr.strip():
            clipped_err = stderr.strip()
            if len(clipped_err) > _MAX_GIT_ERROR_CHARS:
                clipped_err = (
                    clipped_err[:_MAX_GIT_ERROR_CHARS - len("… [stderr clipped]")]
                    + "… [stderr clipped]"
                )
            text += f"\n\ngit said:\n{clipped_err}"
        return _bounded(text)

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
            tail: list[str] = ["log", "--oneline", f"-n{limit}"]
            ref = args.get("ref")
            if isinstance(ref, str) and ref.strip():
                tail.append(_checked_ref(ref.strip()))
            return argv + tail
        if action == "diff":
            tail = ["diff"]
            if args.get("patch") is True:
                tail.append("HEAD")
            else:
                # ``--stat HEAD`` covers staged and unstaged work.
                tail += ["--stat", "HEAD"]
            pathspec = args.get("path")
            if isinstance(pathspec, str) and pathspec.strip():
                tail += [
                    "--",
                    resolve_inside_workspace(
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
            return argv + ["tag", "--list", "-n1", "--sort=-creatordate"]
        if action == "stashes":
            return argv + ["stash", "list"]
        if action == "show":
            ref = args.get("ref")
            revision = (
                _checked_ref(ref.strip())
                if isinstance(ref, str) and ref.strip()
                else "HEAD"
            )
            return argv + ["show", "--stat", "--oneline", revision]
        # action == "blame"
        pathspec = args.get("path")
        if not isinstance(pathspec, str) or not pathspec.strip():
            raise ValueError("'blame' needs 'path' (a workspace file)")
        resolved = resolve_inside_workspace(workspace_path, pathspec.strip())
        line = args.get("line")
        if line is None or line is False:
            ref = args.get("ref")
            revision = (
                _checked_ref(ref.strip())
                if isinstance(ref, str) and ref.strip()
                else "HEAD"
            )
            return argv + ["blame", revision, "--", resolved]
        number = coerce_int_arg(line, default=0, minimum=1, maximum=1_000_000)
        ref = args.get("ref")
        revision = (
            _checked_ref(ref.strip())
            if isinstance(ref, str) and ref.strip()
            else "HEAD"
        )
        return argv + [
            "blame", "-L", f"{number},{number}", revision, "--", resolved,
        ]

    @staticmethod
    async def _run_git(argv: list[str], workspace: str) -> tuple[str, str]:
        """Run a fixed read-only git argv and return decoded output.

        The argv is built here from constants plus the caller's workspace,
        never from model text, so no model-supplied flag can turn the
        snapshot into a mutation. ``GIT_OPTIONAL_LOCKS=0`` and the ``-c``
        switches keep the run strictly passive: no index-refresh lock, no
        fsmonitor daemon spawn, and no locale-dependent path quoting.
        """
        env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
        argv = [
            argv[0],
            "-c", "core.fsmonitor=false",
            "-c", "core.quotepath=false",
            *argv[1:],
        ]
        try:
            stdout, stderr, returncode = await _git_once(
                argv, workspace, env,
            )
        except asyncio.TimeoutError:
            # Same class as TimeoutError on 3.11+; run() reports it.
            raise
        if returncode != 0 and "HEAD" in argv:
            head_failure = any(
                marker in stderr.casefold()
                for marker in ("unknown revision", "ambiguous argument 'head'")
            )
            if head_failure:
                # A repository with zero commits has no HEAD revision; fall
                # back to diffing against the index (which git compares
                # with the empty tree there) instead of failing outright.
                rebuilt = [
                    "--cached" if arg == "HEAD" else arg for arg in argv
                ]
                stdout, stderr, returncode = await _git_once(
                    rebuilt, workspace, env,
                )
        if returncode != 0:
            # Keep the diagnostic to git's first stderr line; later lines
            # are usually usage text that would flood the tool result.
            first_line = stderr.strip().splitlines()
            raw_detail = first_line[0] if first_line else "?"
            if len(raw_detail) > _MAX_GIT_ERROR_CHARS:
                raw_detail = (
                    raw_detail[:_MAX_GIT_ERROR_CHARS - len("… [clipped]")]
                    + "… [clipped]"
                )
            detail = raw_detail
            raise _GitFailed(detail or "?")
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


async def _git_once(
    argv: list[str],
    workspace: str,
    env: dict[str, str],
) -> tuple[str, str, int]:
    """Run one git argv, reaping the process on cancellation."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=workspace,
        env=env,
    )
    try:
        stdout_b, stderr_b = await proc.communicate()
    except BaseException:
        # A turn stop cancelled us; never orphan
        # the git process inside Cozter's process group.
        if proc.returncode is None:
            proc.kill()
            with suppress(ProcessLookupError):
                await proc.wait()
        raise
    return (
        stdout_b.decode("utf-8", errors="replace"),
        stderr_b.decode("utf-8", errors="replace"),
        proc.returncode if proc.returncode is not None else -1,
    )


def _bounded(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return truncate_with_marker(
        text, _MAX_OUTPUT_CHARS, "say PARTIAL + remainder when coverage is unclear",
    )


if __name__ == "__main__":
    GitInfoTool.run_as_script()
