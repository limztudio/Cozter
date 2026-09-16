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

# Bash runs under the real-work tool cap (tool_timeout, default 3600s):
# cancel (/stop, new user message, [[await]] pause) still stops instantly.
# A slow build/search keeps running up to the cap instead of timing out
# early and forcing a wasteful retry.

# Hard ceiling on captured output. A command like ``yes`` or ``cat /dev/zero``
# emits gigabytes; buffering it whole (as ``communicate()`` does) would OOM
# the bot. Only the first few KB reach the model anyway (execute_tool caps
# the result), so once we hit this we stop reading and kill the command tree.
_BASH_MAX_OUTPUT_BYTES = 4 * 1024 * 1024  # 4 MB


class BashTool(AgentTool):
    name = "bash"
    # This shell is intentionally unrestricted: ``cwd`` confines only the
    # starting directory, not paths, environment access, network access, or
    # child processes. Keep it out of HTTP agents' default ``auto`` mode.
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
        # Real-work cap: the command runs up to tool_timeout (default 3600s)
        # or until the turn is cancelled (/stop, new user message).
        # Extra args are ignored.

        # Use the shell so the model can use pipes, redirection, etc.
        shell = _find_shell()
        if shell is None:
            return "Error: no shell available to run bash commands"

        try:
            proc = await asyncio.create_subprocess_exec(
                *shell, command,
                # A tool must never inherit Cozter's interactive stdin: a
                # shell command such as ``cat`` could otherwise consume a
                # CLI user's next message (or a secret piped to the bot).
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
                # Runaway output - stop draining and reap the tree so a
                # firehose command can't hold memory or keep running.
                await _kill_command_tree(proc)
            else:
                await proc.wait()
        except asyncio.CancelledError:
            # /stop fired mid-command - kill the shell so we don't leak it.
            await _kill_command_tree(proc)
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
        # Prefer bash if available (matches what bash users expect); fall
        # back to cmd.
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
    """Read *stream* to EOF or until *limit* bytes, whichever comes first.

    Returns ``(data, truncated)``. ``data`` is at most *limit* bytes; a slice
    at the cap may split a multi-byte UTF-8 sequence, which the caller's
    ``decode(errors="replace")`` handles.
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
    # Shared cleanup uses a POSIX process group where available and
    # ``taskkill /T`` on Windows, so a timed-out shell cannot leave build or
    # test children behind.
    await kill_and_wait(proc)
