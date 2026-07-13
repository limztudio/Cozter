"""bash: run a single shell command inside the workspace."""

from __future__ import annotations

import asyncio
import os
import signal
import shutil
from typing import Any, ClassVar

from ..base import AgentTool, coerce_int_arg

# Bash tool default timeout (model can override via the ``timeout``
# argument up to this hard cap).
_BASH_DEFAULT_TIMEOUT = 30
_BASH_MAX_TIMEOUT = 120

# Hard ceiling on captured output. A command like ``yes`` or ``cat /dev/zero``
# emits gigabytes well within the timeout; buffering it whole (as
# ``communicate()`` does) would OOM the bot before the timeout ever fires.
# Only the first few KB reach the model anyway (execute_tool caps the result),
# so once we hit this we stop reading and kill the command tree.
_BASH_MAX_OUTPUT_BYTES = 4 * 1024 * 1024  # 4 MB


class BashTool(AgentTool):
    name = "bash"
    description = (
        "Run a shell command in the workspace. Use sparingly; prefer"
        " read_file/write_file/edit_file for file ops."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {
                "type": "integer",
                "description": (
                    f"Seconds (default {_BASH_DEFAULT_TIMEOUT},"
                    f" max {_BASH_MAX_TIMEOUT})."
                ),
            },
        },
        "required": ["command"],
    }

    async def run(self, workspace_path: str, args: dict) -> str:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return "Error: 'command' must be a non-empty string"
        timeout = coerce_int_arg(
            args.get("timeout") or _BASH_DEFAULT_TIMEOUT,
            default=_BASH_DEFAULT_TIMEOUT,
            minimum=1,
            maximum=_BASH_MAX_TIMEOUT,
        )

        # Use the shell so the model can use pipes, redirection, etc.
        shell = _find_shell()
        if shell is None:
            return "Error: no shell available to run bash commands"

        try:
            proc = await asyncio.create_subprocess_exec(
                *shell, command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=workspace_path,
                start_new_session=os.name != "nt",
            )
        except FileNotFoundError:
            return "Error: shell not found"

        if proc.stdout is None:  # PIPE was requested, so this is defensive
            await _kill_command_tree(proc)
            return "Error: could not capture command output"

        truncated = False
        try:
            async with asyncio.timeout(timeout):
                stdout, truncated = await _read_capped(
                    proc.stdout, _BASH_MAX_OUTPUT_BYTES,
                )
                if truncated:
                    # Runaway output - stop draining and reap the tree so a
                    # firehose command can't hold memory or keep running.
                    await _kill_command_tree(proc)
                else:
                    await proc.wait()
        except TimeoutError:
            await _kill_command_tree(proc)
            return f"Error: command timed out after {timeout}s"
        except asyncio.CancelledError:
            # /stop fired mid-command - kill the shell so we don't leak it.
            await _kill_command_tree(proc)
            raise

        output = stdout.decode("utf-8", errors="replace")
        if truncated:
            note = (
                f"\n... [output truncated at {_BASH_MAX_OUTPUT_BYTES} bytes;"
                " command killed]"
            )
            return (output + note) if output else note.lstrip()
        rc = proc.returncode
        if rc == 0:
            return output or "(no output)"
        return f"$ exit {rc}\n{output}"

    def summarize(self, args: dict) -> str:
        cmd = args.get("command", "")
        return f"$ {cmd[:200]}" + ("..." if len(cmd) > 200 else "")


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
    if proc.returncode is not None:
        return
    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
        await proc.wait()
    except ProcessLookupError:
        return
    except OSError:
        try:
            proc.kill()
            await proc.wait()
        except OSError:
            return
