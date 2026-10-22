"""Gemini CLI headless JSONL adapter with exact native session reuse.

Verified against @google/gemini-cli 0.65.0 and its official headless and
policy references. Plan mode alone can transition into YOLO in headless
runs, so restricted launches also own a deny-by-default admin policy.
"""

import asyncio
import os
import re
import sys
import tempfile
import uuid

from .. import execution_state
from ..execution_state import native_session_id, record_missing_native_session
from ..process_launch import create_prompt_subprocess
from .base import (
    AgentResult, Backend, ChatEvent, ProcessResourceMap,
    attachment_image_paths, executable_command, record_backend_error, summarize_cli_tool,
    truncate_status_text,
)


# Official 0.65.0 models.ts aliases and cloud chat models; access varies by account.
_MODELS = (
    "auto", "pro", "flash", "flash-lite", "auto-gemini-3", "auto-gemini-2.5",
    "gemini-3.1-pro-preview", "gemini-3.1-pro-preview-customtools",
    "gemini-3-pro-preview", "gemini-3.8-flash", "gemini-3.5-flash",
    "gemini-3-flash-preview", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
)
_READ_TOOLS = ("read_file", "list_directory", "glob", "grep_search")


def _system_policy_directory() -> str:
    """Match Gemini Storage.getSystemPoliciesDir, not a configurable home."""
    if sys.platform == "win32":
        return r"C:\ProgramData\gemini-cli\policies"
    if sys.platform == "darwin":
        return "/Library/Application Support/GeminiCli/policies"
    return "/etc/gemini-cli/policies"


def _check_restricted_policy_available() -> None:
    try:
        with os.scandir(_system_policy_directory()) as entries:
            if any(entry.name.endswith(".toml") for entry in entries):
                raise RuntimeError(
                    "Gemini CLI restricted execution is unavailable because central admin policies "
                    "disable per-launch policy enforcement. Use another read-only backend."
                )
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeError("Cannot verify Gemini CLI restricted policy enforcement") from exc


def _restricted_policy(approval: str) -> str:
    policy = (
        '[[rule]]\ntoolName = "*"\ndecision = "deny"\npriority = 998\n'
        'denyMessage = "This Cozter worker cannot run this tool."\n'
        '\n[[rule]]\ntoolName = "*"\nmcpName = "*"\ndecision = "deny"\npriority = 999\n'
    )
    if approval != "deny":
        reads = ", ".join(f'"{name}"' for name in _READ_TOOLS)
        policy += f'\n[[rule]]\ntoolName = [{reads}]\ndecision = "allow"\npriority = 999\n'
    return policy


def _write_policy(approval: str) -> str:
    fd, path = tempfile.mkstemp(prefix="cozter-gemini-policy-", suffix=".toml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(_restricted_policy(approval))
    except BaseException:
        _remove_policy(path)
        raise
    return path


def _remove_policy(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


class GeminiCliBackend(Backend):
    name = "gemini_cli"
    executable = "gemini"
    available_models = _MODELS
    default_model = "auto"
    default_summary_model = "flash"
    tier_models = {"low": "flash-lite", "mid": "flash", "high": "pro"}
    agent_text_is_delta = True
    supports_vision = True
    supports_read_only_workers = True
    # The CLI has no documented per-run effort flag. Native model defaults apply.
    effort_levels: tuple[str, ...] = ()
    permission_arg_sets = {
        "full": ("--approval-mode", "yolo"),
        "auto": ("--approval-mode", "auto_edit"),
        "restricted": ("--approval-mode", "plan"),
    }

    def __init__(self) -> None:
        self._policy_files = ProcessResourceMap()

    async def launch(
        self, workspace_path: str, prompt: str, model: str | None, approval: str,
        *, compaction: bool = False, effort: int = 0,
    ) -> asyncio.subprocess.Process:
        effective = "deny" if compaction else approval
        restricted = effective not in {"full", "auto"}
        policy_path = None
        if restricted:
            await asyncio.to_thread(_check_restricted_policy_available)
            policy_path = _write_policy(effective)
        try:
            record = None if compaction else execution_state.current_run()
            native_id = native_session_id(record.get("native_session_id")) if record else None
            resuming = native_id is not None
            if native_id is None:
                native_id = str(uuid.uuid4())
            cmd = [*executable_command(self.executable), "--output-format", "stream-json", "--skip-trust"]
            self.append_launch_options(cmd, model, effort, effective)
            if policy_path is not None:
                cmd += ["--admin-policy", policy_path]
            cmd += ["--resume" if resuming else "--session-id", native_id]
            env = dict(os.environ)
            if effective == "full":
                env["GEMINI_SANDBOX"] = "false"
            if not compaction and attachment_image_paths(prompt, workspace_path):
                prompt += (
                    "\n\n[Vision: use your read_file tool on each saved image path above to see "
                    "its pixels before answering about image content.]"
                )
            if record is not None:
                await record.save(
                    native_session_id=native_id, native_resume_supported=True,
                    native_resume_attempted=resuming, native_tools_started=False,
                    native_resume_unavailable=False,
                )
            proc = await create_prompt_subprocess(cmd, prompt, cwd=workspace_path, env=env)
        except BaseException:
            if policy_path is not None:
                _remove_policy(policy_path)
            raise
        if policy_path is not None:
            self._policy_files.remember(proc, policy_path)
        return proc

    async def cleanup_process(self, proc: asyncio.subprocess.Process) -> None:
        path = self._policy_files.pop(proc)
        if path is not None:
            _remove_policy(path)

    def observe_process_exit(self, returncode: int | None, stderr: str) -> None:
        if returncode != 42:
            return
        record = execution_state.current_run()
        native_id = native_session_id(record.get("native_session_id")) if record else None
        if (record is None or native_id is None or not record.get("native_resume_attempted")
                or record.get("native_tools_started")):
            return
        lines = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", stderr).strip().splitlines()
        for index, line in enumerate(lines):
            if not line.startswith("Error resuming session: "):
                continue
            message = "\n".join(lines[index:]).removeprefix("Error resuming session: ")
            if message == "No previous sessions found for this project.":
                record_missing_native_session(f"Session not found: {native_id}")
            else:
                self._record_session_error(message)
            return

    @staticmethod
    def _record_session_error(message: str) -> None:
        record = execution_state.current_run()
        native_id = native_session_id(record.get("native_session_id")) if record else None
        if native_id is not None and re.fullmatch(
            rf'Invalid session identifier "{re.escape(native_id)}"\.\s+'
            r'Searched for sessions(?: in [^\r\n]+)?\.\s+'
            r'Use --list-sessions to see available sessions, then use --resume \{number\}, '
            r'--resume \{uuid\}, or --resume latest\.', message.strip(),
        ):
            record_missing_native_session(f"Session not found: {native_id}")
        else:
            record_missing_native_session(message)

    @staticmethod
    def _error_text(event: dict) -> str:
        error = event.get("error")
        message = error.get("message") if isinstance(error, dict) else event.get("message")
        return message if isinstance(message, str) and message.strip() else "Gemini CLI reported an error"

    def parse_event(self, event: dict, result: AgentResult) -> None:
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        record = execution_state.current_run()
        if kind == "init":
            execution_state.adopt_native_session_id(event.get("session_id"))
        elif kind == "message" and event.get("role") == "assistant":
            text = event.get("content")
            if isinstance(text, str) and text:
                if record is not None:
                    record.set(native_tools_started=True)
                result.text = result.text + text if event.get("delta") is True else text
                result.events.append(ChatEvent(kind="text", content=text))
        elif kind == "tool_use":
            if record is not None:
                record.set(native_tools_started=True)
            result.events.append(ChatEvent(
                kind="tool", content=summarize_cli_tool(event.get("tool_name"), event.get("parameters")),
            ))
        elif kind == "tool_result":
            if record is not None:
                record.set(native_tools_started=True)
            output = event.get("output")
            if event.get("status") == "error":
                output = self._error_text(event)
            if isinstance(output, str) and output:
                result.events.append(ChatEvent(kind="tool", content=truncate_status_text(output)))
        elif kind == "error":
            message = self._error_text(event)
            if event.get("severity") == "warning":
                result.events.append(ChatEvent(kind="tool", content=f"Warning: {truncate_status_text(message)}"))
            else:
                self._record_session_error(message)
                record_backend_error(result, message)
        elif kind == "result":
            stats = event.get("stats")
            if isinstance(stats, dict):
                result.usage = dict(stats)
            if event.get("status") == "error":
                message = self._error_text(event)
                self._record_session_error(message)
                record_backend_error(result, message)

    def extract_agent_text(self, event: dict) -> str | None:
        if isinstance(event, dict) and event.get("type") == "message" and event.get("role") == "assistant":
            text = event.get("content")
            return text if isinstance(text, str) and text else None
        return None
