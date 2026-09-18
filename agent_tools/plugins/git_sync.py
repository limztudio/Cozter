"""Plugin: Git network sync for the workspace repository.

Local reads live in ``git_info`` and local writes in ``git_ops``. This
plugin holds the three operations that touch the network (and any
configured credentials): ``fetch``, ``pull``, and ``push``, plus a
read-only ``remotes`` listing so the model can see where sync would go.

Network + credential use escapes the workspace-bounded safety model, so
this tool sets ``requires_full_permission``: HTTP agents in ``auto``
mode neither see nor run it. Local-only ``git_ops`` stays available in
``auto``; use that for add/commit/branch/stash/reset/merge/tag work.
"""

from __future__ import annotations

import asyncio
import os
import re
from contextlib import suppress
from typing import Any, ClassVar

from ..base import (
    AgentTool,
    object_parameters,
    truncate_with_marker,
)
from ...utils import clip_status_value

# Real-work cap: network sync runs under the tool runner cap
# (default 3600s) or until cancelled.
_MAX_OUTPUT_CHARS = 12_000
_MAX_GIT_ERROR_CHARS = 500
_ACTIONS = ("fetch", "pull", "push", "remotes")
_REF_RE = re.compile(r"^[A-Za-z0-9._\-/]+$")


class _GitFailed(Exception):
    """Git exited non-zero; carries the model-facing stderr excerpt."""


def _clean_name(value: object, *, what: str) -> tuple[str | None, str | None]:
    """Validate a remote/branch/ref token; return (value, error)."""
    if not isinstance(value, str) or not value.strip():
        return None, f"Error: '{what}' must be a non-empty string"
    name = value.strip()
    if len(name) > 200 or not _REF_RE.match(name):
        return None, f"Error: invalid {what} {name!r}"
    for marker in (
        "..", "@{", "~", "^", ":", "?", "*", "[", "\\",
        "'", '"', "`", "$", "(", ")", "|", ";", "&", "<", ">",
        "!", " ", "\t", "\n",
    ):
        if marker in name:
            return None, f"Error: invalid {what} {name!r}"
    if name.startswith(("-", "/", ".")) or name.endswith(("/", ".", ".lock")):
        return None, f"Error: invalid {what} {name!r}"
    return name, None


class GitSyncTool(AgentTool):
    name = "git_sync"
    order = 22  # utility tools group, next to git_info/git_ops
    requires_full_permission = True
    description = (
        "Git network sync: fetch/pull/push/remotes. "
        "Full permission only; local writes live in git_ops."
    )
    parameters: ClassVar[dict[str, Any]] = object_parameters(
        {
            "action": {"type": "string", "enum": list(_ACTIONS)},
            "remote": {"type": "string"},
            "branch": {"type": "string"},
            "prune": {"type": "boolean"},
            "upstream": {
                "type": "boolean",
                "description": "push: set upstream (-u).",
            },
            "force": {
                "type": "boolean",
                "description": "push: --force-with-lease to confirm.",
            },
        },
        ["action"],
    )

    async def run(self, workspace_path: str, args: dict) -> str:
        action = args.get("action")
        if not isinstance(action, str) or action not in _ACTIONS:
            return f"Error: 'action' must be one of {', '.join(_ACTIONS)}"
        argv_or_error = self._build_argv(action, args)
        if isinstance(argv_or_error, str):
            return argv_or_error
        _label, rest = argv_or_error
        argv = ["git", "-C", workspace_path, *rest]
        try:
            stdout, stderr = await _run_git(argv, workspace_path)
        except FileNotFoundError:
            return "Error: git is not installed or not on PATH"
        except TimeoutError:
            return f"Error: git {action} timed out"
        except _GitFailed as exc:
            return f"Error: git {action}: {exc}"
        text = stdout.strip()
        if not text:
            return "OK"
        if stderr.strip():
            clipped_err = stderr.strip()
            if len(clipped_err) > _MAX_GIT_ERROR_CHARS:
                clipped_err = (
                    clipped_err[:_MAX_GIT_ERROR_CHARS - len("… [stderr clipped]")]
                    + "… [stderr clipped]"
                )
            text += f"\n\ngit said:\n{clipped_err}"
        return _bounded(text)

    def _build_argv(
        self, action: str, args: dict,
    ) -> tuple[str, list[str]] | str:
        """Build (label, argv-tail); return model-facing error string."""
        remote_raw = args.get("remote")
        branch_raw = args.get("branch")
        remote: str | None = None
        branch: str | None = None
        if isinstance(remote_raw, str) and remote_raw.strip():
            cleaned, err = _clean_name(remote_raw.strip(), what="remote")
            if err is not None:
                return err
            assert cleaned is not None
            remote = cleaned
        if isinstance(branch_raw, str) and branch_raw.strip():
            cleaned, err = _clean_name(branch_raw.strip(), what="branch")
            if err is not None:
                return err
            assert cleaned is not None
            branch = cleaned

        if action == "remotes":
            return ("remotes", ["remote", "-v"])
        if action == "fetch":
            tail: list[str] = ["fetch"]
            if args.get("prune") is True:
                tail.append("--prune")
            if remote is not None:
                tail.append(remote)
            else:
                tail.append("--all")
                if "--prune" not in tail:
                    tail.append("--prune")
            return ("fetch", tail)
        if action == "pull":
            tail = ["pull", "--ff-only"]
            if remote is not None:
                tail.append(remote)
                if branch is not None:
                    tail.append(branch)
            elif branch is not None:
                return "Error: 'branch' needs 'remote' for pull"
            return ("pull", tail)
        # action == "push"
        tail = ["push"]
        if args.get("upstream") is True:
            tail.append("-u")
        if args.get("force") is True:
            # Never plain --force: --force-with-lease refuses when the
            # upstream moved since the last fetch instead of overwriting.
            tail.append("--force-with-lease")
        if remote is not None:
            tail.append(remote)
            if branch is not None:
                tail.append(branch)
        elif branch is not None:
            return "Error: 'branch' needs 'remote' for push"
        return ("push", tail)

    def summarize(self, args: dict) -> str:
        action = args.get("action") if isinstance(args, dict) else None
        remote = args.get("remote") if isinstance(args, dict) else None
        suffix = (
            f" ({clip_status_value(remote)})"
            if isinstance(remote, str) and remote else ""
        )
        return f"git {clip_status_value(action or '?', 40)}{suffix}"


async def _run_git(argv: list[str], workspace: str) -> tuple[str, str]:
    """Run a fixed network-sync argv; stdin stays closed, no prompts."""
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "1",
    }
    argv = [
        argv[0],
        "-c", "core.fsmonitor=false",
        "-c", "core.quotepath=false",
        *argv[1:],
    ]
    try:
        stdout, stderr, returncode = await _git_once(argv, workspace, env)
    except asyncio.TimeoutError:
        raise
    if returncode != 0:
        first_line = stderr.strip().splitlines()
        raw_detail = first_line[0] if first_line else "?"
        if len(raw_detail) > _MAX_GIT_ERROR_CHARS:
            raw_detail = (
                raw_detail[:_MAX_GIT_ERROR_CHARS - len("… [clipped]")]
                + "… [clipped]"
            )
        raise _GitFailed(raw_detail or "?")
    return stdout, stderr


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
    GitSyncTool.run_as_script()
