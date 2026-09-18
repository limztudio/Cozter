"""Plugin: local Git writes for the workspace repository.

HTTP backends (llama, meta, zai, ...) have no shell outside ``full``
permission, so without this tool they can read repo state (``git_info``)
but cannot stage, commit, branch, stash, reset, merge, or tag. This
plugin exposes those local-only operations with a fixed argv the model
cannot extend: every flag is chosen here from constants, model text only
supplies validated branch/tag/ref names, commit messages, and
workspace-bounded pathspecs.

Local-only: no ``fetch``/``pull``/``push``/``remote`` here. Network sync
lives in ``git_sync`` so workspaces can reason about network and
credential use separately.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, ClassVar

from ..base import (
    AgentTool,
    coerce_int_arg,
    object_parameters,
    resolve_inside_workspace,
)
from ...utils import clip_status_value
from ._git_common import (
    MAX_GIT_ERROR_CHARS as _MAX_GIT_ERROR_CHARS,
    GitFailed as _GitFailed,
    add_common_git_flags,
    bounded as _bounded,
    clean_ref as _clean_ref,
    first_stderr_line,
    git_once as _git_once,
)

# Real-work cap: git runs under the tool runner cap (default 3600s) or
# until cancelled.
# Output/error limits, ref validation, failure type, process runner, and
# output clipping live in ._git_common (shared with git_info/git_sync).
_MAX_MESSAGE_CHARS = 2_000
_ACTIONS = (
    "add",
    "unstage",
    "commit",
    "checkout",
    "branch-create",
    "branch-delete",
    "branch-rename",
    "stash-push",
    "stash-pop",
    "stash-apply",
    "stash-drop",
    "reset",
    "merge",
    "rebase",
    "rebase-abort",
    "tag-create",
    "tag-delete",
    "discard",
    "clean",
)
_RESET_MODES = ("soft", "mixed", "hard")
_STASH_RE = re.compile(r"^stash@\{\d+\}$")


def _clean_message(value: object) -> tuple[str | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return "Error: 'message' must be a non-empty string", None
    text = value.strip()
    if len(text) > _MAX_MESSAGE_CHARS:
        return f"Error: 'message' is too long (max {_MAX_MESSAGE_CHARS} chars)", None
    first = text.splitlines()[0] if text.splitlines() else text
    if not first.strip():
        return "Error: 'message' must be a non-empty string", None
    return None, text


def _collect_pathspecs(
    workspace_path: str, args: dict,
) -> tuple[list[str] | None, str | None]:
    """Resolve ``paths``/``path`` inside the workspace for ``--`` use."""
    raw_paths: list[str] = []
    paths = args.get("paths")
    if isinstance(paths, list):
        for entry in paths:
            if isinstance(entry, str) and entry.strip():
                raw_paths.append(entry.strip())
    single = args.get("path")
    if isinstance(single, str) and single.strip():
        raw_paths.append(single.strip())
    if not raw_paths:
        return None, "Error: 'paths' (or 'path') must list at least one file"
    if len(raw_paths) > 100:
        return None, "Error: too many paths (max 100)"
    resolved: list[str] = []
    for entry in raw_paths:
        try:
            resolved.append(resolve_inside_workspace(workspace_path, entry))
        except ValueError as exc:
            return None, f"Error: {exc}"
    return resolved, None


class GitOpsTool(AgentTool):
    name = "git_ops"
    order = 21  # utility tools group, next to git_info
    description = (
        "Git local writes: add/unstage/commit/checkout/branch/stash/"
        "reset/merge/rebase/tag/discard/clean. Local-only, no network."
    )
    parameters: ClassVar[dict[str, Any]] = object_parameters(
        {
            "action": {"type": "string", "enum": list(_ACTIONS)},
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Workspace files for add/unstage/discard/clean/stash-push.",
            },
            "path": {"type": "string"},
            "message": {"type": "string"},
            "branch": {"type": "string"},
            "new_branch": {"type": "string"},
            "tag": {"type": "string"},
            "ref": {"type": "string"},
            "stash": {"type": "string"},
            "mode": {"type": "string", "enum": list(_RESET_MODES)},
            "all": {"type": "boolean"},
            "create": {"type": "boolean", "description": "checkout: create the branch (-b)."},
            "force": {
                "type": "boolean",
                "description": "branch-delete: -D; branch-rename: -M; clean: confirm delete.",
            },
            "no_ff": {"type": "boolean"},
            "ff_only": {"type": "boolean"},
        },
        ["action"],
    )

    async def run(self, workspace_path: str, args: dict) -> str:
        action = args.get("action")
        if not isinstance(action, str) or action not in _ACTIONS:
            return f"Error: 'action' must be one of {', '.join(_ACTIONS)}"
        try:
            argv = await self._build_argv(workspace_path, action, args)
        except ValueError as exc:
            return str(exc)
        if isinstance(argv, str):
            return argv  # model-facing "Error: ..." from builder
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

    async def _build_argv(
        self, workspace_path: str, action: str, args: dict,
    ) -> list[str] | str:
        """Build the fixed git argv, or return a model-facing error."""
        base: list[str] = ["git", "-C", workspace_path]
        if action == "add":
            all_flag = args.get("all") is True
            has_paths = isinstance(args.get("paths"), list) or isinstance(
                args.get("path"), str,
            )
            if all_flag:
                argv = base + ["add", "-A"]
                if has_paths:
                    spec, err = _collect_pathspecs(workspace_path, args)
                    if err is not None:
                        return err
                    assert spec is not None
                    argv += ["--", *spec]
                return argv
            spec, err = _collect_pathspecs(workspace_path, args)
            if err is not None:
                return err
            assert spec is not None
            return base + ["add", "--", *spec]
        if action == "unstage":
            spec, err = _collect_pathspecs(workspace_path, args)
            if err is not None:
                return err
            assert spec is not None
            return base + ["restore", "--staged", "--", *spec]
        if action == "commit":
            err, message = _clean_message(args.get("message"))
            if err is not None:
                return err
            assert message is not None
            argv = base + ["commit", "-m", message]
            if args.get("all") is True:
                argv.append("-a")
            return argv
        if action == "checkout":
            branch, err = _clean_ref(args.get("branch"), what="branch")
            if err is not None:
                return err
            assert branch is not None
            if args.get("create") is True:
                argv = base + ["checkout", "-b", branch]
                ref = args.get("ref")
                if isinstance(ref, str) and ref.strip():
                    start, err = _clean_ref(ref.strip(), what="ref")
                    if err is not None:
                        return err
                    assert start is not None
                    argv.append(start)
                return argv
            return base + ["checkout", branch]
        if action == "branch-create":
            branch, err = _clean_ref(args.get("branch"), what="branch")
            if err is not None:
                return err
            assert branch is not None
            argv = base + ["branch", branch]
            ref = args.get("ref")
            if isinstance(ref, str) and ref.strip():
                start, err = _clean_ref(ref.strip(), what="ref")
                if err is not None:
                    return err
                assert start is not None
                argv.append(start)
            return argv
        if action == "branch-delete":
            branch, err = _clean_ref(args.get("branch"), what="branch")
            if err is not None:
                return err
            assert branch is not None
            flag = "-D" if args.get("force") is True else "-d"
            return base + ["branch", flag, branch]
        if action == "branch-rename":
            old, err = _clean_ref(args.get("branch"), what="branch")
            if err is not None:
                return err
            new, err = _clean_ref(args.get("new_branch"), what="new_branch")
            if err is not None:
                return err
            assert old is not None and new is not None
            flag = "-M" if args.get("force") is True else "-m"
            return base + ["branch", flag, old, new]
        if action == "stash-push":
            argv = base + ["stash", "push"]
            message = args.get("message")
            if isinstance(message, str) and message.strip():
                err, clean = _clean_message(message)
                if err is not None:
                    return err
                assert clean is not None
                argv += ["-m", clean]
            paths = args.get("paths")
            single = args.get("path")
            if isinstance(paths, list) or isinstance(single, str):
                spec, err = _collect_pathspecs(workspace_path, args)
                if err is not None:
                    return err
                assert spec is not None
                argv += ["--", *spec]
            return argv
        if action in ("stash-pop", "stash-apply", "stash-drop"):
            verb = action.split("-", 1)[1]
            argv = base + ["stash", verb]
            stash = args.get("stash")
            if isinstance(stash, str) and stash.strip():
                name = stash.strip()
                if not _STASH_RE.match(name):
                    return f"Error: invalid stash {name!r} (want stash@{{N}})"
                argv.append(name)
            return argv
        if action == "reset":
            mode = args.get("mode")
            if mode is None:
                mode = "mixed"
            if not isinstance(mode, str) or mode not in _RESET_MODES:
                return f"Error: 'mode' must be one of {', '.join(_RESET_MODES)}"
            ref = args.get("ref")
            if ref is None or (isinstance(ref, str) and not ref.strip()):
                ref_name = "HEAD"
            else:
                cleaned_ref, err = _clean_ref(ref, what="ref")
                if err is not None:
                    return err
                assert cleaned_ref is not None
                ref_name = cleaned_ref
            paths = args.get("paths")
            single = args.get("path")
            if isinstance(paths, list) or isinstance(single, str):
                spec, err = _collect_pathspecs(workspace_path, args)
                if err is not None:
                    return err
                assert spec is not None
                # Path-scoped reset unstages those paths to <ref>.
                return base + ["reset", ref_name, "--", *spec]
            return base + ["reset", f"--{mode}", ref_name]
        if action == "merge":
            branch, err = _clean_ref(args.get("branch"), what="branch")
            if err is not None:
                return err
            assert branch is not None
            ff_only = args.get("ff_only") is True
            no_ff = args.get("no_ff") is True
            if ff_only and no_ff:
                return "Error: 'ff_only' and 'no_ff' are mutually exclusive"
            argv = base + ["merge"]
            if ff_only:
                argv.append("--ff-only")
            elif no_ff:
                argv.append("--no-ff")
            argv.append(branch)
            return argv
        if action == "rebase":
            upstream, err = _clean_ref(args.get("branch"), what="branch")
            if err is not None:
                # Accept "ref" as an alias for the upstream.
                upstream, err = _clean_ref(args.get("ref"), what="ref")
            if err is not None:
                return err
            assert upstream is not None
            return base + ["rebase", upstream]
        if action == "rebase-abort":
            return base + ["rebase", "--abort"]
        if action == "tag-create":
            tag, err = _clean_ref(args.get("tag"), what="tag")
            if err is not None:
                return err
            assert tag is not None
            message = args.get("message")
            argv = base + ["tag"]
            if isinstance(message, str) and message.strip():
                err, clean = _clean_message(message)
                if err is not None:
                    return err
                assert clean is not None
                argv += ["-a", tag, "-m", clean]
            else:
                argv.append(tag)
            ref = args.get("ref")
            if isinstance(ref, str) and ref.strip():
                start, err = _clean_ref(ref.strip(), what="ref")
                if err is not None:
                    return err
                assert start is not None
                argv.append(start)
            return argv
        if action == "tag-delete":
            tag, err = _clean_ref(args.get("tag"), what="tag")
            if err is not None:
                return err
            assert tag is not None
            return base + ["tag", "-d", tag]
        if action == "discard":
            spec, err = _collect_pathspecs(workspace_path, args)
            if err is not None:
                return err
            assert spec is not None
            return base + [
                "restore", "--source=HEAD", "--staged", "--worktree",
                "--", *spec,
            ]
        if action == "clean":
            if args.get("force") is not True:
                return "Error: 'clean' deletes untracked files; pass force=true to confirm"
            paths = args.get("paths")
            single = args.get("path")
            argv = base + ["clean", "-fd"]
            if args.get("all") is True:
                return argv
            if isinstance(paths, list) or isinstance(single, str):
                spec, err = _collect_pathspecs(workspace_path, args)
                if err is not None:
                    return err
                assert spec is not None
                argv += ["--", *spec]
                return argv
            return "Error: 'clean' needs paths or all=true so it never wipes blindly"
        raise ValueError(f"Error: unknown action {action!r}")

    def summarize(self, args: dict) -> str:
        action = args.get("action") if isinstance(args, dict) else None
        branch = args.get("branch") if isinstance(args, dict) else None
        tag = args.get("tag") if isinstance(args, dict) else None
        extra = branch if isinstance(branch, str) and branch else (
            tag if isinstance(tag, str) and tag else ""
        )
        suffix = f" ({clip_status_value(extra)})" if extra else ""
        limit = coerce_int_arg(40, default=40, minimum=1, maximum=200)
        return f"git {clip_status_value(action or '?', limit)}{suffix}"


async def _run_git(argv: list[str], workspace: str) -> tuple[str, str]:
    """Run a fixed local-write git argv and return decoded output.

    The argv is built by :meth:`GitOpsTool._build_argv` from constants
    plus validated names/messages/paths, never from model-supplied flags.
    ``GIT_TERMINAL_PROMPT=0`` keeps a misconfigured repo from hanging on
    a credential prompt; stdin stays DEVNULL either way.
    """
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "1",
    }
    argv = add_common_git_flags(argv)
    try:
        stdout, stderr, returncode = await _git_once(argv, workspace, env)
    except asyncio.TimeoutError:
        raise
    if returncode != 0:
        raise _GitFailed(first_stderr_line(stderr))
    return stdout, stderr


if __name__ == "__main__":
    GitOpsTool.run_as_script()
