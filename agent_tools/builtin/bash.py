"""bash: run a single shell command inside the workspace."""

from __future__ import annotations

import asyncio
import logging
import ntpath
import os
import shutil
from typing import Any, ClassVar

from ..base import AgentTool
from ...utils import (
    clip_status_value,
    close_subprocess_pipe,
    has_managed_process_group,
    kill_and_wait,
    mark_process_group_leader,
    wait_for_process_exit,
)

# Commands run until completion or cancellation.

_BASH_MAX_OUTPUT_BYTES = 4 * 1024 * 1024
# Only applies once the shell has exited, not to commands still doing work.
_BASH_POST_EXIT_DRAIN_TIMEOUT = 1.0
_BASH_PROGRESS_INTERVAL_SEC = 60.0
performance_logger = logging.getLogger("Cozter.performance")


class BashTool(AgentTool):
    name = "bash"
    # Unrestricted shell: full-permission only.
    requires_full_permission = True
    description = (
        "Run a host shell command using the shell stated in the system prompt"
        " (cwd = workspace; runs until done or cancelled)."
        " On Windows the shell may be PowerShell or cmd; use its syntax."
        " For build/test/verify: capture output to a workspace file using"
        " the selected shell's redirection, then grep the FULL file for"
        " error/warning/exception/traceback — never eyeball tail only;"
        " a truncated preview is not proof of clean."
        " Keep long validation resumable with full logs and per-case checkpoints."
        " For interrupted CTest on an unchanged build, inspect prior pass/fail"
        " evidence and use -F to resume; skipped or previously failed cases"
        " still need valid passing evidence before claiming completion."
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
        # Extra args are ignored.

        # Pipes/redirection need a real shell.
        shell = _find_shell()
        if shell is None:
            return "Error: no shell available to run bash commands"

        try:
            proc = await _spawn_command(shell, command, workspace_path)
        except FileNotFoundError:
            return "Error: shell not found"

        if proc.stdout is None:  # PIPE was requested, so this is defensive
            await _kill_command_tree(proc)
            return "Error: could not capture command output"

        try:
            stdout, truncated, incomplete = await _read_command_output(
                proc, _BASH_MAX_OUTPUT_BYTES,
            )
            if truncated or incomplete:
                # Reap the owned tree on excess output or an inherited pipe.
                await _kill_command_tree(proc)
            else:
                await wait_for_process_exit(proc)
        except BaseException:
            await _kill_command_tree(proc)  # /stop must not leak it.
            raise
        finally:
            close_subprocess_pipe(proc, 1)

        output = stdout.decode("utf-8", errors="replace")
        if truncated:
            note = (
                f"\n… [output truncated at {_BASH_MAX_OUTPUT_BYTES} bytes;"
                " command killed; never treat this preview as full content;"
                " say PARTIAL + remainder when work is left]"
            )
            return (output + note) if output else note.lstrip()
        if incomplete:
            output += (
                "\n… [output capture stopped after the shell exited while a"
                " child kept its output pipe open; output may be incomplete]"
            )
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
        # A WindowsApps/System32 bash.exe launches WSL, which may be absent or
        # prompt for setup. A real Bash installation remains usable.
        bash = shutil.which("bash")
        if bash and not _is_windows_bash_launcher(bash):
            return [bash, "-c"]
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell:
            return [
                powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
            ]
        cmd = shutil.which("cmd.exe") or "cmd.exe"
        return [cmd, "/d", "/c"]
    sh = shutil.which("bash") or shutil.which("sh")
    if sh:
        return [sh, "-c"]
    return None


async def _spawn_command(
    shell: list[str], command: str, workspace_path: str,
) -> asyncio.subprocess.Process:
    """Retain process ownership when cancellation races shell creation."""
    async def create() -> asyncio.subprocess.Process:
        proc = await asyncio.create_subprocess_exec(
            *shell, command,
            # Never inherit interactive stdin.
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=workspace_path,
            start_new_session=os.name != "nt",
        )
        if os.name != "nt":
            mark_process_group_leader(proc)
        return proc

    launch = asyncio.create_task(create())
    try:
        # wait() lets creation settle without forwarding cancellation to it.
        await asyncio.wait((launch,))
        return launch.result()
    except asyncio.CancelledError:
        cleanup = asyncio.create_task(_reap_cancelled_command_launch(launch))
        while not cleanup.done():
            try:
                await asyncio.wait((cleanup,))
            except asyncio.CancelledError:
                # Repeated /stop must not abandon a process being handed over.
                continue
        cleanup.result()
        raise


async def _reap_cancelled_command_launch(
    launch: asyncio.Task[asyncio.subprocess.Process],
) -> None:
    try:
        proc = await launch
    except (Exception, asyncio.CancelledError):
        return
    try:
        await _kill_command_tree(proc)
    finally:
        close_subprocess_pipe(proc, 1)


def _is_windows_bash_launcher(path: str) -> bool:
    normalized = ntpath.normpath(path).casefold()
    return (
        "\\microsoft\\windowsapps\\" in normalized
        or "\\windows\\system32\\" in normalized
        or "\\windows\\sysnative\\" in normalized
    )


def shell_context() -> str:
    """Describe the selected host shell so model commands use its dialect."""
    shell = _find_shell()
    if shell is None:
        return "Host shell: unavailable."
    name = ntpath.basename(shell[0]).casefold().removesuffix(".exe")
    if name in {"pwsh", "powershell"}:
        return "Host shell: PowerShell (noninteractive, without profiles). Use PowerShell syntax."
    if name == "cmd":
        return "Host shell: Windows cmd.exe. Use cmd syntax."
    return f"Host shell: {name}. Use POSIX shell syntax."


async def _read_command_output(
    proc: asyncio.subprocess.Process, limit: int,
) -> tuple[bytes, bool, bool]:
    """Capture output until EOF, the byte cap, or an exited parent's grace.

    Waiting for EOF alone can hang when a background child inherits stdout.
    Watch the shell's exit independently of its pipes and allow a fixed drain
    grace for buffered output. Keep captured bytes when abandoning that pipe.
    """
    stream = proc.stdout
    assert stream is not None
    chunks: list[bytes] = []
    total = 0
    truncated = False
    incomplete = False
    read_task = asyncio.create_task(stream.read(64 * 1024))
    exit_task = asyncio.create_task(wait_for_process_exit(proc))
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline: float | None = None
    try:
        while True:
            if deadline is None and exit_task.done():
                await exit_task
                deadline = loop.time() + _BASH_POST_EXIT_DRAIN_TIMEOUT
            if deadline is None:
                done, _ = await asyncio.wait(
                    (read_task, exit_task),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=_BASH_PROGRESS_INTERVAL_SEC,
                )
                if not done:
                    performance_logger.info(
                        "stage=tool_wait tool=bash pid=%s seconds=%.3f outcome=running",
                        getattr(proc, "pid", "?"), loop.time() - started,
                    )
                if not read_task.done():
                    continue
            else:
                remaining = deadline - loop.time()
                if remaining <= 0 and not read_task.done():
                    incomplete = True
                    break
                if not read_task.done():
                    done, _ = await asyncio.wait((read_task,), timeout=remaining)
                    if not done:
                        incomplete = True
                        break
            chunk = await read_task
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                truncated = True
                break
            if deadline is not None and loop.time() >= deadline:
                incomplete = True
                break
            read_task = asyncio.create_task(stream.read(64 * 1024))
    finally:
        for task in (read_task, exit_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(read_task, exit_task, return_exceptions=True)
    data = b"".join(chunks)
    return (data[:limit] if truncated else data), truncated, incomplete


async def _kill_command_tree(proc: asyncio.subprocess.Process) -> None:
    """Terminate the shell and any children it spawned."""
    if proc.returncode is not None and not has_managed_process_group(proc):
        return
    # Process-group/taskkill cleanup: no orphaned children.
    await kill_and_wait(proc)
