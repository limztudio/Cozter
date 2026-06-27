"""bash: run a single shell command inside the workspace."""

from __future__ import annotations

import asyncio
import os
import shutil

from ..base import AgentTool, coerce_int_arg

# Bash tool default timeout (model can override via the ``timeout``
# argument up to this hard cap).
_BASH_DEFAULT_TIMEOUT = 30
_BASH_MAX_TIMEOUT = 120


class BashTool(AgentTool):
    name = "bash"
    description = (
        "Run a shell command in the workspace. Use sparingly; prefer"
        " read_file/write_file/edit_file for file ops."
    )
    parameters = {
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
            )
        except FileNotFoundError:
            return "Error: shell not found"

        try:
            async with asyncio.timeout(timeout):
                stdout, _ = await proc.communicate()
        except TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except OSError:
                pass
            return f"Error: command timed out after {timeout}s"
        except asyncio.CancelledError:
            # /stop fired mid-command - kill the shell so we don't leak it.
            try:
                proc.kill()
                await proc.wait()
            except OSError:
                pass
            raise

        output = stdout.decode("utf-8", errors="replace")
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
