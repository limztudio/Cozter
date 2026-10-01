"""bash: run a single shell command inside the workspace."""

from __future__ import annotations

import asyncio
import os
import shutil
from typing import Any, ClassVar

from ..base import AgentTool
from ...utils import (
    clip_status_value,
    has_managed_process_group,
    kill_and_wait,
    mark_process_group_leader,
)

# Real-work cap (tool_timeout, default 3600s); cancel stops instantly.

# Output ceiling: only the first KBs reach the model, so stop reading and
# kill the tree past this instead of buffering a firehose whole.
_BASH_MAX_OUTPUT_BYTES = 4 * 1024 * 1024  # 4 MB


class BashTool(AgentTool):
    name = "bash"
    # Unrestricted shell (cwd is only the start dir): full-permission only.
    requires_full_permission = True
    description = (
        "Run a shell command (cwd = workspace, 3600s real-work cap — runs until done"
        " or cancelled). For build/test/verify: capture output to a file"
        " (`... 2>&1 | tee /tmp/build.log`), then grep the FULL file for"
        " error/warning/exception/traceback — never eyeball tail only;"
        " a truncated preview is not proof of clean."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
        },
        "required": ["command"],
    }

    async def run(self, workspace_path: str, args: dict) -> str:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return "Error: 'command' must be a non-empty string"
        # Extra args are ignored (real-work cap or turn cancel bounds the run).

        # Pipes/redirection need a real shell.
        shell = _find_shell()
        if shell is None:
            return "Error: no shell available to run bash commands"

        try:
            proc = await asyncio.create_subprocess_exec(
                *shell, command,
                # Never inherit interactive stdin (a `cat` could eat the next message).
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=workspace_path,
                start_new_session=os.name != "nt",
            )
        except FileNotFoundError:
            return "Error: shell not found"
        if os.name != "nt":
            mark_process_group_leader(proc)

        if proc.stdout is None:  # PIPE was requested, so this is defensive
            await _kill_command_tree(proc)
            return "Error: could not capture command output"

        truncated = False
        try:
            stdout, truncated = await _read_capped(
                proc.stdout, _BASH_MAX_OUTPUT_BYTES,
            )
            if truncated:
                # Reap the tree: a firehose must not hold memory or keep running.
                await _kill_command_tree(proc)
            else:
                await proc.wait()
        except asyncio.CancelledError:
            await _kill_command_tree(proc)  # /stop must not leak the shell.
            raise

        output = stdout.decode("utf-8", errors="replace")
        if truncated:
            note = (
                f"\n… [output truncated at {_BASH_MAX_OUTPUT_BYTES} bytes;"
                " command killed; never treat this preview as full content;"
                " say PARTIAL + remainder when work is left]"
            )
            return (output + note) if output else note.lstrip()
        rc = proc.returncode
        if rc == 0:
            return output or "(no output)"
        return f"$ exit {rc}\n{output}"

    def summarize(self, args: dict) -> str:
        cmd = args.get("command", "")
        if not isinstance(cmd, str):
            cmd = str(cmd)
        return f"$ {clip_status_value(cmd)}"


def _find_shell() -> list[str] | None:
    """Return an argv prefix that runs a single shell command."""
    if os.name == "nt":
        # Prefer bash; fall back to cmd.
        bash = shutil.which("bash")
        if bash:
            return [bash, "-c"]
        cmd = shutil.which("cmd.exe") or "cmd.exe"
        return [cmd, "/c"]
    sh = shutil.which("bash") or shutil.which("sh")
    if sh:
        return [sh, "-c"]
    return None


async def _read_capped(
    stream: asyncio.StreamReader, limit: int,
) -> tuple[bytes, bool]:
    """Read *stream* to EOF or *limit* bytes; return ``(data, truncated)``.

    A cut at the cap may split a UTF-8 sequence; the caller decodes
    with ``errors="replace"``.
    """
    chunks: list[bytes] = []
    total = 0
    truncated = False
    while True:
        chunk = await stream.read(64 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > limit:
            truncated = True
            break
    data = b"".join(chunks)
    return (data[:limit] if truncated else data), truncated


async def _kill_command_tree(proc: asyncio.subprocess.Process) -> None:
    """Terminate the shell and any children it spawned."""
    if proc.returncode is not None and not has_managed_process_group(proc):
        return
    # Process-group cleanup (POSIX) / taskkill (Windows): no orphaned children.
    await kill_and_wait(proc)
