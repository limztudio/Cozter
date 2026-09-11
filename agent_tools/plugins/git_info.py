"""Plugin: read-only Git snapshot of the workspace repository.

HTTP backends (llama, meta, zai, ...) work through Cozter's typed tools
and have no shell outside ``full`` permission, so they cannot ask the
repository for its own state. This plugin exposes the three read-only
questions agents ask most - what changed, what happened recently, and
what does the patch look like - with a fixed, read-only argv the model
cannot extend.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from typing import Any, ClassVar

from ..base import (
    AgentTool,
    coerce_int_arg,
    object_parameters,
    resolve_inside_workspace,
)

_GIT_TIMEOUT_SECONDS = 15
_MAX_OUTPUT_CHARS = 12_000
_MAX_GIT_ERROR_CHARS = 500
_ACTIONS = ("status", "log", "diff")


class _GitFailed(Exception):
    """Git exited non-zero; carries the model-facing stderr excerpt."""


class GitInfoTool(AgentTool):
    name = "git_info"
    order = 20  # utility tools group
    description = "Git status/log/diff. Read-only."
    parameters: ClassVar[dict[str, Any]] = object_parameters(
        {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
            },
            "path": {"type": "string"},
            "patch": {
                "type": "boolean",
                "description": "Full patch.",
            },
            "limit": {
                "type": "integer",
                "description": "max 50.",
            },
        },
        ["action"],
    )

    async def run(self, workspace_path: str, args: dict) -> str:
        action = args.get("action")
        if not isinstance(action, str) or action not in _ACTIONS:
            return "Error: 'action' must be one of status, log, diff"

        argv: list[str] = ["git", "-C", workspace_path]
        if action == "status":
            argv += ["status", "--short", "--branch"]
        elif action == "log":
            limit = coerce_int_arg(
                args.get("limit") or 10, default=10, minimum=1, maximum=50,
            )
            argv += ["log", "--oneline", f"-n{limit}"]
        else:
            argv.append("diff")
            if args.get("patch") is True:
                argv.append("HEAD")
            else:
                # ``--stat HEAD`` covers staged and unstaged work.
                argv += ["--stat", "HEAD"]
            pathspec = args.get("path")
            if isinstance(pathspec, str) and pathspec.strip():
                try:
                    argv += [
                        "--",
                        resolve_inside_workspace(
                            workspace_path, pathspec.strip(),
                        ),
                    ]
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
            return "(no output)"
        if stderr.strip():
            clipped_err = stderr.strip()
            if len(clipped_err) > _MAX_GIT_ERROR_CHARS:
                clipped_err = (
                    clipped_err[:_MAX_GIT_ERROR_CHARS]
                    + "… [stderr clipped]"
                )
            text += f"\n\ngit said:\n{clipped_err}"
        return _bounded(text)

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
                for marker in ("head", "unknown revision")
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
                    raw_detail[:_MAX_GIT_ERROR_CHARS] + "… [clipped]"
                )
            detail = raw_detail
            raise _GitFailed(detail or "?")
        return stdout, stderr

    def summarize(self, args: dict) -> str:
        action = args.get("action") if isinstance(args, dict) else None
        path = args.get("path") if isinstance(args, dict) else None
        suffix = f" ({path})" if isinstance(path, str) and path else ""
        return f"git {action or '?'}{suffix}"


async def _git_once(
    argv: list[str],
    workspace: str,
    env: dict[str, str],
) -> tuple[str, str, int]:
    """Run one git argv, reaping the process on timeout or cancellation."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=workspace,
        env=env,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), _GIT_TIMEOUT_SECONDS,
        )
    except BaseException:
        # The outer tool timeout or a turn stop cancelled us; never orphan
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
    return (
        text[:_MAX_OUTPUT_CHARS]
        + f"\n… [truncated, {len(text)} chars total;"
        " never treat this preview as full content;"
        " say PARTIAL + remainder when coverage is unclear]"
    )


if __name__ == "__main__":
    GitInfoTool.run_as_script()
