"""Grok Build CLI backend.

Grok Build's headless ``streaming-messages-json`` output follows the
Anthropic Messages-style JSONL shape.  It carries whole assistant messages,
tool uses, and a terminal ``result`` event, which makes it a better fit for
Cozter than the delta-oriented ``streaming-json`` format:

  - ``assistant.message.content`` contains ``text`` and ``tool_use`` blocks
  - ``user.message.content`` contains tool results (not shown as chat status)
  - ``result`` carries the final aggregate reply plus usage and cost metadata

The CLI receives its prompt through ``--prompt-file`` so Cozter's history
budget is not truncated by the platform argv limit.  Headless permission
modes map Cozter's four levels onto Grok flags that actually stick in
headless ``--prompt-file`` runs: always-approve for ``full``, always-approve
inside the workspace sandbox for ``auto``, and ``dontAsk`` plus a read-only
sandbox for ``confirm``/``deny``.  Grok's native ``auto`` classifier is a
TUI feature; this CLI currently ignores it in headless mode and falls back
to ask/default, which would hang or deny tool calls.  Restricted calls also
allowlist only non-mutating workspace tools and explicitly disallow the MCP
discovery tools that Grok otherwise keeps visible.
"""

import asyncio
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time

from .base import (
    MODEL_CATALOG_TTL_SEC,
    AgentResult,
    Backend,
    ChatEvent,
    append_text_result,
    create_captured_subprocess,
    executable_command,
    fresh_model_catalog,
    record_backend_error,
)

logger = logging.getLogger(__name__)

# ``grok models`` is account-aware, but the picker must remain useful when
# Grok is absent, unauthenticated, or its model-list command is unavailable.
# Keep only currently documented CLI models in the conservative fallback,
# with every capability beside its ID so picker, effort, and compaction
# cannot drift apart.  Grok rejects unsupported ``--effort`` values, so
# unpublished/custom IDs use the three-level subset both published models
# share rather than grok-4.6's extra ``xhigh``.
_COMMON_EFFORT_LEVELS = ("low", "medium", "high")
_FALLBACK_MODEL_SPECS = (
    ("grok-4.6", (*_COMMON_EFFORT_LEVELS, "xhigh"), 500_000),
    ("grok-4.5", _COMMON_EFFORT_LEVELS, 500_000),
)
_FALLBACK_MODELS = tuple(
    model for model, _efforts, _window in _FALLBACK_MODEL_SPECS
)
_FALLBACK_MODEL_EFFORT_LEVELS: dict[str, tuple[str, ...]] = {
    model: efforts for model, efforts, _window in _FALLBACK_MODEL_SPECS
}
_FALLBACK_MODEL_CONTEXT_WINDOWS = {
    model: window for model, _efforts, window in _FALLBACK_MODEL_SPECS
}
_MODEL_DISCOVERY_TIMEOUT_SEC = 15
_MODEL_LINE_RE = re.compile(r"^\s*[*-]\s+(\S+)\s*(?:\([^)]*\))?\s*$")
_PROMPT_FILE_PREFIX = "cozter-grok-prompt-"


def _parse_models_output(output: str | bytes) -> tuple[str, ...]:
    """Extract model IDs from the human-readable ``grok models`` listing.

    Current Grok Build releases intentionally expose this command as plain
    text rather than JSON.  Only accept entries after its ``Available models``
    heading so login banners and diagnostics cannot become selectable models.
    """
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    if not isinstance(output, str):
        return ()

    models: list[str] = []
    seen: set[str] = set()
    reading_models = False
    for line in output.splitlines():
        if line.strip().casefold() == "available models:":
            reading_models = True
            continue
        if not reading_models:
            continue
        match = _MODEL_LINE_RE.match(line)
        if match is None:
            # The list is a contiguous block. A blank separator or another
            # diagnostic heading ends it without interpreting later output.
            if not line.strip() or not line.startswith((" ", "\t")):
                break
            continue
        model = match.group(1).strip()
        if model and model not in seen:
            seen.add(model)
            models.append(model)
    return tuple(models)


def _write_prompt_file(prompt: str) -> str:
    """Write *prompt* to a private temp file and return its path."""
    fd, path = tempfile.mkstemp(prefix=_PROMPT_FILE_PREFIX)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(prompt.encode("utf-8"))
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


def _remove_prompt_file(path: str) -> None:
    """Delete a prompt file, ignoring a race with an already-reaped path."""
    try:
        os.unlink(path)
    except OSError:
        pass


class GrokBackend(Backend):
    name = "grok"
    executable = "grok"
    default_model = "grok-4.6"
    default_summary_model = "grok-4.6"
    # Default-model vocabulary. ``effort_levels_for_model`` narrows this for
    # grok-4.5 and unpublished IDs; effort=0 still means do not override.
    effort_levels = (*_COMMON_EFFORT_LEVELS, "xhigh")
    # Grok's model catalog is account-dependent, so do not route flexible's
    # low tier to a potentially unavailable pinned model before its picker
    # refreshes. Every unset tier uses the policy-safe default_model.
    tier_models: dict[str, str] = {}
    permission_arg_sets = {
        "full": ("--always-approve",),
        # Headless Grok currently ignores ``--permission-mode auto`` and
        # reports ask/default instead. Always-approve still auto-runs tools
        # so a chat turn cannot hang, while the workspace sandbox confines
        # writes to CWD / Grok home / temp.
        "auto": ("--always-approve", "--sandbox", "workspace"),
        "restricted": (
            "--permission-mode",
            "dontAsk",
            "--sandbox",
            "read-only",
            "--tools",
            "read_file,grep,list_dir",
            # ``--tools`` is an allowlist, but Grok still exposes its MCP
            # discovery/execution pair unless those names are denied.
            "--disallowed-tools",
            "search_tool,use_tool",
        ),
    }

    def __init__(self) -> None:
        self._cached_models: tuple[str, ...] | None = None
        self._catalog_expires_at = 0.0
        self._model_catalog_lock = threading.Lock()
        self._prompt_files: dict[int, str] = {}
        self._prompt_files_lock = threading.Lock()

    # ---- model discovery -----------------------------------------------

    @property
    def available_models(self) -> tuple[str, ...]:  # type: ignore[override]
        """Return Grok models available to the currently authenticated CLI."""
        return self._model_catalog()

    def _model_catalog(self) -> tuple[str, ...]:
        cached = fresh_model_catalog(
            self._cached_models, self._catalog_expires_at,
        )
        if cached is not None:
            return cached

        with self._model_catalog_lock:
            cached = fresh_model_catalog(
                self._cached_models, self._catalog_expires_at,
            )
            if cached is not None:
                return cached
            models = self._discover_models()
            self._cached_models = models
            self._catalog_expires_at = time.monotonic() + MODEL_CATALOG_TTL_SEC
            return models

    def _discover_models(self) -> tuple[str, ...]:
        if shutil.which(self.executable) is None:
            logger.debug("grok not on PATH; using fallback model list")
            return _FALLBACK_MODELS
        try:
            proc = subprocess.run(
                [*executable_command(self.executable), "models"],
                capture_output=True,
                timeout=_MODEL_DISCOVERY_TIMEOUT_SEC,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.debug("grok models probe failed (%s); using fallback", exc)
            return _FALLBACK_MODELS
        if proc.returncode != 0:
            logger.debug("grok models exited %s; using fallback", proc.returncode)
            return _FALLBACK_MODELS
        models = _parse_models_output(proc.stdout)
        if not models:
            logger.debug("grok models output was unrecognised; using fallback")
            return _FALLBACK_MODELS
        return models

    def effort_levels_for_model(
        self, model: str | None,
    ) -> tuple[str, ...]:
        """Return only the effort values the selected Grok model accepts.

        Grok exits the turn when ``--effort`` is not in the model's menu, so
        unpublished and custom IDs stay on the shared three-level subset.
        """
        selected = (model or self.default_model).strip()
        return _FALLBACK_MODEL_EFFORT_LEVELS.get(
            selected, _COMMON_EFFORT_LEVELS,
        )

    def context_window_tokens(self, model: str | None) -> int | None:
        """Return a published capacity for a known Grok CLI model ID."""
        selected = (model or self.default_model).strip()
        return _FALLBACK_MODEL_CONTEXT_WINDOWS.get(selected)

    # ---- launch ---------------------------------------------------------

    async def launch(
        self,
        workspace_path: str,
        prompt: str,
        model: str | None,
        approval: str,
        *,
        compaction: bool = False,
        effort: int = 0,
    ) -> asyncio.subprocess.Process:
        prompt_path = _write_prompt_file(prompt)
        cmd: list[str] = [
            *executable_command(self.executable),
            "--cwd",
            workspace_path,
            "--output-format",
            "streaming-messages-json",
        ]
        self.append_launch_options(cmd, model, effort, approval)
        # Keep the prompt argument last so model/permission flags cannot be
        # parsed as prompt text. ``--prompt-file`` avoids the platform argv
        # cap that ``-p`` inherits; Cozter's default history budget already
        # exceeds Windows' CreateProcess limit.
        cmd += ["--prompt-file", prompt_path]
        try:
            proc = await create_captured_subprocess(
                cmd,
                cwd=workspace_path,
                # Keep launched shell commands in an owned POSIX process group;
                # /stop and /inject can then stop the complete agent tree.
                start_new_session=os.name != "nt",
            )
        except BaseException:
            _remove_prompt_file(prompt_path)
            raise
        if isinstance(proc.pid, int):
            with self._prompt_files_lock:
                self._prompt_files[proc.pid] = prompt_path
        else:
            _remove_prompt_file(prompt_path)
        return proc

    async def cleanup_process(
        self, proc: asyncio.subprocess.Process,
    ) -> None:
        """Remove this launch's prompt file after the process exits."""
        path: str | None = None
        if isinstance(proc.pid, int):
            with self._prompt_files_lock:
                path = self._prompt_files.pop(proc.pid, None)
        if path is not None:
            _remove_prompt_file(path)

    # ---- streaming event parsing ---------------------------------------

    _FILE_TOOL_NAMES = frozenset(
        {
            "search_replace",
            "write",
            "write_file",
            "create_file",
            "edit_file",
        }
    )

    def parse_event(self, event: dict, result: AgentResult) -> None:
        etype = event.get("type", "")

        if etype == "assistant":
            self._handle_assistant_message(event, result)
            return

        if etype == "result":
            self._handle_result(event, result)
            return

        if etype == "error":
            # A late provider error must be recorded, but must not erase
            # a useful model reply that was already streamed.
            record_backend_error(result, self._error_message(event))
            return

        # ``system``, ``user`` (tool result), reasoning, and metadata events
        # are useful to Grok's own transcript but do not improve Cozter's
        # compact live status display.
        if etype not in {
            "system",
            "user",
            "usage",
            "available_commands",
            "plan",
            "auto_compact_start",
            "auto_compact_end",
            "thought",
        }:
            logger.debug(
                "Grok: unhandled event type=%r keys=%r",
                etype,
                list(event.keys()),
            )

    def extract_agent_text(self, event: dict) -> str | None:
        """Return Grok's terminal reply for internal summary calls."""
        etype = event.get("type", "")
        if etype == "result" and not event.get("is_error"):
            text = event.get("result")
            return text if isinstance(text, str) and text else None
        if etype != "assistant":
            return None
        message = event.get("message") or {}
        if not isinstance(message, dict):
            return None
        return self._message_text(message)

    def _handle_assistant_message(
        self,
        event: dict,
        result: AgentResult,
    ) -> None:
        message = event.get("message") or {}
        if not isinstance(message, dict):
            return
        content = message.get("content")
        if isinstance(content, str):
            if content:
                append_text_result(result, content)
            return
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    append_text_result(result, text)
            elif block_type == "tool_use":
                self._append_tool_event(block, result)

    def _handle_result(self, event: dict, result: AgentResult) -> None:
        usage = event.get("usage")
        if isinstance(usage, dict):
            result.usage = dict(usage)
            cost = event.get("total_cost_usd")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                result.usage["total_cost_usd"] = cost

        if event.get("is_error"):
            record_backend_error(result, self._error_message(event))
            return

        # In normal streams the last assistant message has already supplied
        # this reply. Keep the terminal field as a compatibility fallback for
        # a truncated/older stream that emits only ``result``.
        text = event.get("result")
        if (
            isinstance(text, str)
            and text
            and not any(item.kind == "text" for item in result.events)
        ):
            append_text_result(result, text)

    @staticmethod
    def _error_message(event: dict) -> str:
        """Return Grok's human-readable error across its two envelopes.

        Stream-level errors use ``message``.  Terminal headless failures,
        including model-selection failures, instead expose one or more strings
        under ``errors`` and leave both ``error`` and ``result`` absent.  Do
        not stringify provider objects: an absent usable message must retain
        the common ``Unknown error`` fallback instead of leaking a Python
        representation into a chat reply.
        """
        for key in ("message", "error", "result"):
            message = event.get(key)
            if isinstance(message, str) and message.strip():
                return message
        errors = event.get("errors")
        if isinstance(errors, list):
            for message in errors:
                if isinstance(message, str) and message.strip():
                    return message
        return "Unknown error"

    def _append_tool_event(self, block: dict, result: AgentResult) -> None:
        name = block.get("name") or "tool"
        if not isinstance(name, str):
            name = "tool"
        tool_input = block.get("input")
        if not isinstance(tool_input, dict):
            tool_input = {}
        command = tool_input.get("command") or tool_input.get("cmd")
        path = (
            tool_input.get("path")
            or tool_input.get("file_path")
            or tool_input.get("file")
        )
        if isinstance(command, str) and command:
            summary = f"$ {command}"
        elif isinstance(path, str) and path:
            summary = f"{name}: {path}"
        else:
            summary = name
        result.events.append(ChatEvent(kind="tool", content=summary))

        if name in self._FILE_TOOL_NAMES and isinstance(path, str) and path:
            result.events.append(
                ChatEvent(
                    kind="file",
                    content=f"📄 changed: {path}",
                )
            )

    @staticmethod
    def _message_text(message: dict) -> str | None:
        """Flatten a complete Messages-style assistant text response."""
        content = message.get("content")
        if isinstance(content, str):
            return content or None
        if not isinstance(content, list):
            return None
        texts = [
            block["text"]
            for block in content
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"]
            )
        ]
        return "\n".join(texts) or None
