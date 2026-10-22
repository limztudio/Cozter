"""Antigravity CLI streaming adapter with exact native conversation reuse."""

import asyncio
import json
import os
import re
import subprocess
import sys

from .. import execution_state
from ..execution_state import native_session_id, record_missing_native_session
from ..process_launch import (
    await_process_launch, create_piped_subprocess, create_prompt_subprocess, reap_failed_subprocess,
)
from .base import (
    AgentResult, Backend, CachedModelCatalog, ChatEvent, NativeSessionUnavailableError,
    fresh_model_catalog, record_backend_error,
    resolve_executable_prefix, summarize_cli_tool, truncate_status_text,
)


# Verified 1.4.5 account catalog; live discovery decides actual availability.
_MODELS = (
    "gemini-4-argon-medium", "gemini-4-argon-high", "gemini-4-argon-low",
    "gemini-3.8-flash-high", "gemini-3.8-flash-medium", "gemini-3.8-flash-low",
    "claude-opus-5-5-low", "claude-opus-5-5-medium", "claude-opus-5-5-high",
    "claude-sonnet-5-5-low", "claude-sonnet-5-5-medium", "claude-sonnet-5-5-high",
    "gpt-oss-120b-medium",
)
_EFFORTS = ("low", "medium", "high", "xhigh", "max")
_RESUME_STARTUP_TIMEOUT = 30
_STARTUP_STDERR_BYTES = 64 * 1024


class _StartupStderr(asyncio.StreamReader):
    """Replay diagnostics drained while checking the native conversation."""

    def __init__(self, prefix: bytes, source: asyncio.StreamReader) -> None:
        super().__init__()
        self._prefix = prefix
        self._source = source

    async def read(self, n: int = -1) -> bytes:
        if not self._prefix:
            return await self._source.read(n)
        if n < 0:
            prefix, self._prefix = self._prefix, b""
            return prefix + await self._source.read(n)
        chunk, self._prefix = self._prefix[:n], self._prefix[n:]
        return chunk


async def _stop_task(task: asyncio.Task) -> None:
    task.cancel()
    while not task.done():
        try:
            await asyncio.wait((task,))
        except asyncio.CancelledError:
            continue
    if not task.cancelled():
        task.exception()


def _command_prefix() -> list[str] | None:
    prefix = resolve_executable_prefix("agy")
    if prefix is not None:
        return prefix
    if sys.platform == "win32":
        path = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "AntigravityCLI", "agy.exe")
        if os.path.isfile(path):
            return [path]
    return None


def _model_family(model: str) -> str:
    base, _, suffix = model.rpartition("-")
    return base if suffix in _EFFORTS else model


def _parse_models(output: bytes) -> tuple[str, ...]:
    models = []
    for line in output.decode("utf-8", errors="replace").splitlines():
        slug, separator, label = line.partition("\t")
        if (separator and label.strip() and slug.casefold() not in {"model", "slug", "id"}
                and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", slug) and slug not in models):
            models.append(slug)
    return tuple(models)


class AntigravityCliBackend(CachedModelCatalog, Backend):
    name = "antigravity_cli"
    executable = "agy"
    default_model = "gemini-4-argon-medium"
    default_summary_model = "gemini-3.8-flash-medium"
    tier_models = {
        "low": "gemini-3.8-flash-low",
        "mid": "gemini-3.8-flash-medium",
        "high": "gemini-4-argon-high",
    }
    supports_internal_calls = False
    effort_levels = _EFFORTS
    permission_arg_sets = {
        "full": ("--dangerously-skip-permissions", "--sandbox=false"),
        "auto": ("--mode", "auto"),
    }

    def __init__(self) -> None:
        super().__init__()
        self._stdin_tasks: dict[int, asyncio.Task[None]] = {}

    @property
    def available_models(self) -> tuple[str, ...]:  # type: ignore[override]
        return self._live_model_catalog()

    def _fetch_models(self) -> tuple[str, ...]:
        prefix = _command_prefix()
        if prefix is None:
            return _MODELS
        try:
            proc = subprocess.run(
                [*prefix, "models"], stdin=subprocess.DEVNULL, capture_output=True,
                timeout=15, check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return _MODELS
        if proc.returncode != 0:
            return _MODELS
        return _parse_models(proc.stdout) or _MODELS

    @classmethod
    def permission_args(cls, approval: str) -> list[str]:
        if approval not in {"full", "auto"}:
            raise RuntimeError("Antigravity CLI has no verified per-launch restricted policy")
        return list(cls.permission_arg_sets[approval])

    def health_check(self) -> tuple[bool, str]:
        prefix = _command_prefix()
        if prefix is None:
            return False, "'agy' not found on PATH or in the Antigravity CLI install directory"
        return True, f"agy ({prefix[-1]})"

    def effort_levels_for_model(self, model: str | None) -> tuple[str, ...]:
        selected = _model_family((model or self.default_model).strip())
        # Effort confirmation must not probe a native process on the event loop.
        models = fresh_model_catalog(self._cached_models, self._catalog_expires_at)
        return tuple(level for level in _EFFORTS if f"{selected}-{level}" in (models if models is not None else _MODELS))

    def context_window_tokens(self, model: str | None) -> int | None:
        if _model_family((model or self.default_model).strip()) == "gemini-4-argon":
            return 1_048_576
        return None

    async def launch(
        self, workspace_path: str, prompt: str, model: str | None, approval: str,
        *, compaction: bool = False, effort: int = 0,
    ) -> asyncio.subprocess.Process:
        if compaction or approval not in {"full", "auto"}:
            raise RuntimeError(
                "Antigravity CLI has no verified per-launch restricted policy. "
                "Use a backend with enforced read-only permissions for this call."
            )
        prefix = _command_prefix()
        if prefix is None:
            raise FileNotFoundError("Antigravity CLI executable 'agy' was not found")
        record = execution_state.current_run()
        native_id = native_session_id(record.get("native_session_id")) if record else None
        cmd = [*prefix, "--input-format", "stream-json", "--output-format", "stream-json", "--print-timeout", "0"]
        self.append_launch_options(cmd, model, effort, approval)
        if native_id is not None:
            cmd += ["--conversation", native_id]
        if record is not None:
            await record.save(
                native_session_id=native_id,
                native_resume_supported=True, native_resume_attempted=native_id is not None,
                native_tools_started=False, native_resume_unavailable=False, native_stdin_failed=False,
            )
        wire = json.dumps({"event": "user", "message": {"content": prompt}}, ensure_ascii=False) + "\n"
        if native_id is not None:
            assert record is not None
            return await self._launch_resuming(cmd, wire, workspace_path, native_id, record)
        return await create_prompt_subprocess(cmd, wire, cwd=workspace_path)

    async def _launch_resuming(
        self, cmd: list[str], wire: str, workspace_path: str, native_id: str,
        record: execution_state.ExecutionRecord,
    ) -> asyncio.subprocess.Process:
        launch = asyncio.create_task(create_piped_subprocess(
            cmd, stdin=asyncio.subprocess.PIPE, cwd=workspace_path, start_new_session=os.name != "nt",
        ))
        proc = await await_process_launch(launch, reap_failed_subprocess)
        diagnostics = bytearray()
        original_stderr = proc.stderr

        async def capture_stderr() -> None:
            if original_stderr is not None:
                while chunk := await original_stderr.read(8192):
                    remaining = _STARTUP_STDERR_BYTES - len(diagnostics)
                    if remaining > 0:
                        diagnostics.extend(chunk[:remaining])

        stderr_task = asyncio.create_task(capture_stderr())
        try:
            if proc.stdout is None or proc.stdin is None:
                raise RuntimeError("Antigravity CLI did not open its resume streams")
            try:
                async with asyncio.timeout(_RESUME_STARTUP_TIMEOUT):
                    line = await proc.stdout.readline()
            except TimeoutError as exc:
                raise RuntimeError("Antigravity CLI timed out before confirming its native conversation") from exc
            await asyncio.sleep(0)
            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)
            if original_stderr is not None:
                proc.stderr = _StartupStderr(bytes(diagnostics), original_stderr)
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                detail = bytes(diagnostics).decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    "Antigravity CLI ended before confirming its native conversation"
                    + (f": {detail}" if detail else ""),
                ) from exc
            if not isinstance(event, dict) or event.get("event") != "init":
                raise RuntimeError("Antigravity CLI did not confirm its native conversation before input")
            confirmed = native_session_id(event.get("conversation_id"))
            if confirmed is None:
                raise RuntimeError("Antigravity CLI returned an invalid native conversation ID")
            if confirmed != native_id:
                await record.save(
                    native_session_id=None, native_resume_attempted=False, native_resume_unavailable=True,
                    native_unavailable_session_id=native_id, native_tools_started=False,
                )
                raise NativeSessionUnavailableError(self.name, native_id)
            await record.save(native_session_id=native_id)
            self._stdin_tasks[id(proc)] = asyncio.create_task(self._write_resumed_prompt(proc, wire, record))
            return proc
        except BaseException:
            async def cleanup_startup() -> None:
                try:
                    await _stop_task(stderr_task)
                finally:
                    await reap_failed_subprocess(proc)

            cleanup = asyncio.create_task(cleanup_startup())
            while not cleanup.done():
                try:
                    await asyncio.wait((cleanup,))
                except asyncio.CancelledError:
                    continue
            cleanup.result()
            raise

    @staticmethod
    async def _write_resumed_prompt(
        proc: asyncio.subprocess.Process, wire: str, record: execution_state.ExecutionRecord,
    ) -> None:
        assert proc.stdin is not None
        try:
            data = wire.encode("utf-8")
            for offset in range(0, len(data), 64 * 1024):
                proc.stdin.write(data[offset:offset + 64 * 1024])
                await proc.stdin.drain()
        except (OSError, RuntimeError):
            record.set(native_stdin_failed=True)
            await reap_failed_subprocess(proc)
        except asyncio.CancelledError:
            record.set(native_stdin_failed=True)
            raise
        finally:
            proc.stdin.close()

    async def cleanup_process(self, proc: asyncio.subprocess.Process) -> None:
        task = self._stdin_tasks.pop(id(proc), None)
        if task is not None:
            await _stop_task(task)

    @staticmethod
    def _matching_conversation(event: dict, result: AgentResult) -> bool:
        if "conversation_id" not in event:
            return True
        actual = native_session_id(event["conversation_id"])
        record = execution_state.current_run()
        expected = native_session_id(record.get("native_session_id")) if record is not None else None
        if actual is None:
            record_backend_error(result, "Antigravity CLI returned an invalid native conversation ID")
            return False
        if expected is not None and actual != expected:
            record_backend_error(result, "Antigravity CLI returned an event for another native conversation")
            return False
        return True

    def parse_event(self, event: dict, result: AgentResult) -> None:
        if not isinstance(event, dict):
            return
        kind = event.get("event")
        record = execution_state.current_run()
        if kind == "init":
            if self._matching_conversation(event, result):
                execution_state.adopt_native_session_id(event.get("conversation_id"))
        elif kind == "step_update":
            step = event.get("step_update")
            if not isinstance(step, dict) or not self._matching_conversation(step, result):
                return
            if record is not None and record.get("native_stdin_failed"):
                record_backend_error(result, "Antigravity CLI could not receive the resumed prompt")
                return
            if record is not None:
                record.set(native_tools_started=True)
            if step.get("step_type") == "agent_response":
                text = step.get("text_delta")
                if isinstance(text, str) and text:
                    result.text += text
                    result.events.append(ChatEvent(kind="text", content=text))
            elif step.get("step_type") == "tool":
                info = step.get("tool_info")
                if not isinstance(info, dict):
                    info = {}
                result.events.append(ChatEvent(
                    kind="tool", content=summarize_cli_tool(step.get("tool_name") or info.get("name"), info.get("parameters")),
                ))
                error = info.get("error")
                output = error.get("message") if isinstance(error, dict) else info.get("output")
                if isinstance(output, str) and output:
                    result.events.append(ChatEvent(kind="tool", content=truncate_status_text(output)))
        elif kind == "result":
            terminal = event.get("result")
            if not isinstance(terminal, dict):
                record_backend_error(result, "Antigravity CLI returned an invalid result")
                return
            result.terminal_result_received = True
            if not self._matching_conversation(terminal, result):
                return
            if record is not None and record.get("native_stdin_failed"):
                record_backend_error(result, "Antigravity CLI could not receive the resumed prompt")
                return
            execution_state.adopt_native_session_id(terminal.get("conversation_id"))
            response = terminal.get("response")
            if record is not None and (terminal.get("status") == "SUCCESS" or isinstance(response, str) and response):
                record.set(native_tools_started=True)
            if isinstance(response, str) and response:
                result.events[:] = [event for event in result.events if event.kind != "text"]
                result.events.append(ChatEvent(kind="text", content=response))
                result.text = response
            usage = terminal.get("usage")
            if isinstance(usage, dict):
                result.usage = dict(usage)
            if terminal.get("status") != "SUCCESS":
                error = terminal.get("error")
                if not isinstance(error, str) or not error.strip():
                    error = f"Antigravity CLI ended with status {terminal.get('status') or 'unknown'}"
                record_missing_native_session(error)
                record_backend_error(result, error)

    def finalize_result(self, result: AgentResult) -> None:
        record = execution_state.current_run()
        if result.error is None:
            if record is not None and record.get("native_stdin_failed"):
                record_backend_error(result, "Antigravity CLI could not receive the resumed prompt")
            elif not result.terminal_result_received:
                record_backend_error(result, "Antigravity CLI stream ended without a final result")

    def extract_agent_text(self, event: dict) -> str | None:
        if isinstance(event, dict) and event.get("event") == "result":
            terminal = event.get("result")
            if isinstance(terminal, dict):
                response = terminal.get("response")
                return response if isinstance(response, str) and response else None
        return None
