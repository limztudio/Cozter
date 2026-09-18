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

from typing import Any, ClassVar

from ..base import (
    AgentTool,
    object_parameters,
)
from ...utils import clip_status_value
from ._git_common import (
    GitFailed as _GitFailed,
    clean_ref as _clean_name,
    finish_git_output as _finish_git_output,
    finish_git_run_error as _finish_git_run_error,
    run_git as _run_git,
)

# Real-work cap: network sync runs under the tool runner cap
# (default 3600s) or until cancelled.
# Output/error limits, name validation, failure type, process runner,
# and output clipping live in ._git_common (shared with git_info/git_ops).
_ACTIONS = ("fetch", "pull", "push", "remotes")


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
        except (FileNotFoundError, TimeoutError, _GitFailed) as exc:
            return _finish_git_run_error(action, exc)
        return _finish_git_output(stdout, stderr)

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


if __name__ == "__main__":
    GitSyncTool.run_as_script()
