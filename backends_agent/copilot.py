"""GitHub Copilot CLI backend.

Copilot's non-interactive mode (`copilot -p <prompt> --output-format json`)
streams JSONL events to stdout. Unlike codex, copilot:
  - takes the prompt via argv (`-p`), not stdin - prompts are capped to
    the platform's exec limit (Windows' ~32K CreateProcess command line;
    POSIX ARG_MAX, commonly ~2 MB). copilot has no prompt-via-stdin or
    --prompt-file path yet, so argv is the only delivery mechanism.
  - has no `-C <dir>` flag - we use the subprocess `cwd` parameter
  - has coarser permission semantics: --allow-all-tools, --yolo, or default
    (which would prompt, unusable in non-interactive mode)

The JSONL event schema is not formally documented. ``parse_event`` and
``extract_agent_text`` use best-effort key probing (``text``/``content``
for assistant messages, ``tool_use``/``tool_call`` for tool invocations)
and log unknown event types so the schema can be refined.

For the model picker, the CLI's help text is deliberately *not* used: it
lists model names understood by the binary, including names disabled for an
account by an enterprise policy. The ACP session configuration API exposes
the authenticated account's model selector instead.
"""

import asyncio
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from .base import (
    AgentResult, Backend, ChatEvent, append_text_result, executable_command,
    set_error_result, truncate_status_text,
)

logger = logging.getLogger(__name__)

# Floor applied on every platform: the Windows CreateProcess command line
# caps at 32767 chars for the whole argv, so keep the prompt well under it.
_WINDOWS_PROMPT_CHARS = 28_000
_ACP_PROTOCOL_VERSION = 1
_MODEL_DISCOVERY_TIMEOUT_SEC = 12
_MODEL_CATALOG_TTL_SEC = 60
_MODEL_FAILURE_RETRY_SEC = 15
_MAX_ACP_MESSAGES_PER_REQUEST = 100
_COPILOT_HOME_FILES = ("config.json", "settings.json")

# ``auto`` is accepted by Copilot even if a named-model catalog cannot be
# queried. Do not fall back to the generic models from ``copilot help``:
# those names may be disabled for this account or enterprise.
_FALLBACK_MODELS = ("auto",)


def _max_prompt_chars() -> int:
    """Largest prompt (chars) we can safely pass to copilot via ``-p``.

    copilot delivers the prompt as a single argv value - it has no
    prompt-via-stdin or ``--prompt-file`` path yet - so the OS exec limit
    applies. Windows' CreateProcess caps the whole command line at 32767
    chars; POSIX bounds argv + env combined by ARG_MAX (commonly ~2 MB).
    Use a conservative fraction of ARG_MAX to leave room for env vars, the
    executable path, and the other flags. The old fixed 28K cap truncated
    POSIX prompts far below both the OS limit and agent.py's own 50K
    context budget.
    """
    if sys.platform == "win32":
        return _WINDOWS_PROMPT_CHARS
    try:
        arg_max = os.sysconf("SC_ARG_MAX")
    except (ValueError, OSError, AttributeError):
        arg_max = 0
    if arg_max <= 0:
        return 128_000
    return max(_WINDOWS_PROMPT_CHARS, min(arg_max // 4, 1_000_000))


def _create_isolated_copilot_home() -> str:
    """Create a short-lived Copilot home with just account metadata.

    Copilot CLI persists every prompt-mode invocation under ``~/.copilot``.
    Cozter already owns the durable conversation history, and Flexible can
    make several CLI calls per user turn.  A private home keeps those runs
    out of Copilot's visible history; the local account/settings metadata is
    copied so the CLI can still use the user's OS-backed authentication.
    """
    home = tempfile.mkdtemp(prefix="cozter-copilot-")
    source_home = os.environ.get("COPILOT_HOME") or os.path.join(
        os.path.expanduser("~"), ".copilot",
    )
    try:
        for name in _COPILOT_HOME_FILES:
            source = os.path.join(source_home, name)
            if os.path.isfile(source):
                shutil.copy2(source, os.path.join(home, name))
    except OSError:
        shutil.rmtree(home, ignore_errors=True)
        raise
    return home


def _remove_isolated_copilot_home(home: str) -> None:
    """Discard temporary Copilot session state without touching user history."""
    shutil.rmtree(home, ignore_errors=True)


class CopilotBackend(Backend):
    name = "copilot"
    executable = "copilot"
    # ``auto`` is policy-aware: Copilot chooses from models allowed for the
    # signed-in account. It is also the only safe default before ACP has
    # returned an account-specific catalog.
    default_model = "auto"
    default_summary_model = "auto"
    # A static tier mapping could select a model forbidden for an enterprise
    # account before the picker is ever opened. Let all unset Copilot tiers
    # use the policy-aware ``default_model`` instead.
    tier_models: dict[str, str] = {}
    # An ACP list is authoritative for this account, so ``extra_models`` must
    # not inject arbitrary, unverified names back into a picker.
    allow_unverified_extra_models = False
    # No override is represented by Cozter's effort=0 (omit the flag).
    effort_levels = ("low", "medium", "high", "xhigh", "max")

    def __init__(self) -> None:
        # The backend is a process-wide singleton. Cache only an ACP result,
        # never a failed probe: a transient sign-in/network failure should
        # keep the picker fail-closed to ``auto`` but recover on its next
        # open. Refresh successful results periodically so changed enterprise
        # policy is reflected without restarting Cozter.
        self._cached_models: tuple[str, ...] | None = None
        self._catalog_expires_at = 0.0
        self._fallback_expires_at = 0.0
        self._models_lock = threading.Lock()
        self._process_homes: dict[int, str] = {}
        self._process_homes_lock = threading.Lock()

    def effort_levels_for_model(self, model: str | None) -> tuple[str, ...]:
        """Return the effort vocabulary supported by a selected model.

        Copilot's policy-aware ``auto`` selector rejects ``--effort``.  It
        is also the implicit selector when no model is supplied.  Explicit,
        account-approved model IDs retain the normal Copilot effort scale.
        """
        if not model or model.strip().casefold() == "auto":
            return ()
        return self.effort_levels

    # ---- model discovery -----------------------------------------------

    @property
    def available_models(self) -> tuple[str, ...]:  # type: ignore[override]
        """Named models enabled for the authenticated Copilot account.

        ACP's ``session/new`` response contains the account-aware model
        selector. The generic CLI help catalog is intentionally never used,
        because it can advertise models this account cannot select. If ACP
        cannot produce a structured selector, return only ``auto``.
        """
        now = time.monotonic()
        if (
            self._cached_models is not None
            and now < self._catalog_expires_at
        ):
            return self._cached_models
        if now < self._fallback_expires_at:
            return _FALLBACK_MODELS

        with self._models_lock:
            now = time.monotonic()
            if (
                self._cached_models is not None
                and now < self._catalog_expires_at
            ):
                return self._cached_models
            if now < self._fallback_expires_at:
                return _FALLBACK_MODELS

            # An expired catalog must not keep displaying names that a newly
            # applied policy might have removed. A failed refresh therefore
            # deliberately falls back to only ``auto`` rather than this old
            # value.
            self._cached_models = None
            models = self._discover_models()
            if models is not None:
                self._cached_models = models
                self._catalog_expires_at = (
                    time.monotonic() + _MODEL_CATALOG_TTL_SEC
                )
                self._fallback_expires_at = 0.0
                return models
            # This is a short retry throttle, not a model-list cache: it
            # prevents an absent/broken CLI from spawning a new process for
            # each input while still recovering quickly after sign-in.
            self._fallback_expires_at = (
                time.monotonic() + _MODEL_FAILURE_RETRY_SEC
            )
            return _FALLBACK_MODELS

    def resolve_configured_model(self, model: str) -> str:
        """Keep stored Copilot choices inside a fresh ACP catalog.

        This intentionally does not launch ACP from a chat turn. Until a
        picker has obtained a fresh account catalog (or after its short TTL),
        ``auto`` is safer than forwarding a possibly policy-disabled stored
        model ID to the CLI.
        """
        models = self._cached_models
        if (
            models is not None
            and time.monotonic() < self._catalog_expires_at
            and model in models
        ):
            return model
        return self.default_model

    def _discover_models(self) -> tuple[str, ...] | None:
        binary = shutil.which(self.executable)
        if binary is None:
            logger.debug("copilot not on PATH; model picker is auto-only")
            return None
        prefix = executable_command(self.executable)
        uses_windows_cmd_shim = (
            os.name == "nt"
            and bool(prefix)
            and os.path.basename(prefix[0]).casefold() == "cmd.exe"
        )

        proc: subprocess.Popen[str] | None = None
        isolated_home: str | None = None
        try:
            isolated_home = _create_isolated_copilot_home()
            env = os.environ.copy()
            env["COPILOT_HOME"] = isolated_home
            proc = subprocess.Popen(
                [
                    *prefix,
                    "--acp",
                    "--stdio",
                    "--disable-builtin-mcps",
                    "--no-auto-update",
                    "--no-custom-instructions",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # ACP's protocol output belongs exclusively on stdout. Do not
                # leave stderr piped: a verbose CLI failure could otherwise
                # block this small metadata-only probe.
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=os.getcwd(),
                env=env,
            )
            if proc.stdout is None:
                logger.debug("copilot ACP started without stdout")
                return None

            messages: queue.Queue[str | None] = queue.Queue()
            reader = threading.Thread(
                target=_read_acp_stdout,
                args=(proc.stdout, messages),
                daemon=True,
            )
            reader.start()
            deadline = time.monotonic() + _MODEL_DISCOVERY_TIMEOUT_SEC

            initialized = _acp_request(
                proc,
                messages,
                request_id=1,
                method="initialize",
                params={
                    "protocolVersion": _ACP_PROTOCOL_VERSION,
                    "clientCapabilities": {},
                    "clientInfo": {
                        "name": "cozter-model-catalog",
                        "version": "1",
                    },
                },
                deadline=deadline,
            )
            if initialized is None:
                logger.debug("copilot ACP initialization did not succeed")
                return None
            if not _acp_notification(proc, "initialized", {}):
                logger.debug("copilot ACP initialized notification failed")
                return None

            session = _acp_request(
                proc,
                messages,
                request_id=2,
                method="session/new",
                params={"cwd": os.path.abspath(os.getcwd()), "mcpServers": []},
                deadline=deadline,
            )
            if session is None:
                logger.debug("copilot ACP session setup did not succeed")
                return None

            models = _parse_acp_model_options(session)
            if not models:
                logger.debug(
                    "copilot ACP session yielded no structured model selector",
                )
                return None

            # The session has no prompt and is used only for its catalog. If
            # this ACP build supports it, explicitly free any session-side
            # resources before ending the process.
            _close_acp_session_if_supported(
                proc,
                messages,
                initialized,
                session,
                deadline=min(deadline, time.monotonic() + 2),
            )
            return models
        except OSError as exc:
            logger.debug("copilot ACP catalog probe failed (%s)", exc)
            return None
        finally:
            if proc is not None:
                _stop_acp_process(proc, kill_tree=uses_windows_cmd_shim)
            if isolated_home is not None:
                _remove_isolated_copilot_home(isolated_home)

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
        max_prompt_chars = _max_prompt_chars()
        if len(prompt) > max_prompt_chars:
            # Keep the tail: the user's current message is at the end of the
            # composed prompt, and the head (capability-hint preamble + old context)
            # is the least costly to drop.
            logger.warning(
                "Copilot prompt %d chars exceeds %d-char cap; "
                "dropping oldest %d chars of context",
                len(prompt), max_prompt_chars,
                len(prompt) - max_prompt_chars,
            )
            prompt = prompt[-max_prompt_chars:]

        prefix = executable_command(self.executable)
        cmd: list[str] = [
            *prefix, "--output-format", "json", "--no-color",
        ]
        self.append_model_effort_args(
            cmd,
            model,
            effort,
            effort_levels=self.effort_levels_for_model(model),
        )

        if compaction or approval == "full":
            cmd.append("--yolo")
        else:
            # "auto" / "confirm" / "deny" all collapse to --allow-all-tools
            # here: copilot cannot interactively prompt a Telegram user, so
            # non-interactive runs must be non-blocking. Log when the user's
            # intent is stricter than what copilot can enforce.
            if approval in ("confirm", "deny"):
                logger.info(
                    "Copilot backend does not support approval=%s; "
                    "falling back to --allow-all-tools",
                    approval,
                )
            cmd.append("--allow-all-tools")

        cmd += ["-p", prompt]
        isolated_home = _create_isolated_copilot_home()
        env = os.environ.copy()
        env["COPILOT_HOME"] = isolated_home
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace_path,
                env=env,
                # Own process group so a /stop or inject-restart kills the
                # whole tree, not just the copilot parent (see
                # utils.terminate_process_group). POSIX only; no-op on
                # Windows.
                start_new_session=os.name != "nt",
            )
        except BaseException:
            _remove_isolated_copilot_home(isolated_home)
            raise

        # The shared drain paths call ``cleanup_process`` after reaping this
        # process, including cancellation and injected-message restarts.
        if isinstance(proc.pid, int):
            with self._process_homes_lock:
                self._process_homes[proc.pid] = isolated_home
        else:  # Defensive fallback for a nonstandard Process implementation.
            _remove_isolated_copilot_home(isolated_home)
        return proc

    async def cleanup_process(
        self, proc: asyncio.subprocess.Process,
    ) -> None:
        """Remove this launch's private Copilot home after it exits."""
        home: str | None = None
        if isinstance(proc.pid, int):
            with self._process_homes_lock:
                home = self._process_homes.pop(proc.pid, None)
        if home is not None:
            await asyncio.to_thread(_remove_isolated_copilot_home, home)

    _TOOL_USE_TYPES = ("tool_use", "tool_call", "tool_start", "tool")
    _TOOL_RESULT_TYPES = ("tool_result", "tool_output", "tool_end")
    _FILE_TYPES = ("file_change", "edit", "file")
    _ERROR_TYPES = ("error", "turn.failed", "failed")
    _ASSISTANT_TYPES = (
        "assistant_message", "agent_message", "message",
        "completion", "response", "text", "assistant.message",
    )
    _NON_AGENT_TYPES = frozenset(
        _TOOL_USE_TYPES + _TOOL_RESULT_TYPES + _FILE_TYPES + _ERROR_TYPES
    )

    def parse_event(self, event: dict, result: AgentResult) -> None:
        etype = event.get("type") or event.get("event") or ""

        # Typed branches first so a tool event with an "output"/"content"
        # field doesn't get misrouted as assistant text by _extract_text.
        if etype in self._TOOL_USE_TYPES:
            tool = (
                event.get("name")
                or event.get("tool")
                or event.get("tool_name")
                or "?"
            )
            inp = event.get("input") or event.get("args") or {}
            result.events.append(ChatEvent(
                kind="tool", content=self._summarize_tool(tool, inp),
            ))
            return

        if etype in self._TOOL_RESULT_TYPES:
            tool = event.get("name") or event.get("tool") or "tool"
            output = event.get("output") or event.get("result") or ""
            if isinstance(output, (dict, list)):
                output = str(output)
            content = f"{tool} done"
            if output:
                content += f"\n{truncate_status_text(output)}"
            result.events.append(ChatEvent(kind="tool", content=content))
            return

        if etype in self._FILE_TYPES:
            path = (
                event.get("path")
                or event.get("file")
                or event.get("filename")
                or "?"
            )
            kind = event.get("action") or event.get("kind") or "change"
            result.events.append(ChatEvent(
                kind="file", content=f"📄 {kind}: {path}",
            ))
            return

        if etype in self._ERROR_TYPES:
            msg = event.get("message")
            if not msg:
                err = event.get("error")
                if isinstance(err, dict):
                    msg = err.get("message")
                elif isinstance(err, str):
                    msg = err
            if not msg:
                msg = "Unknown error"
            set_error_result(result, msg)
            return

        # Fall through: treat as assistant text if it looks like one.
        text = self._assistant_text(event, etype)
        if text:
            append_text_result(result, text)
            return

        logger.debug("Copilot: unhandled event type=%r keys=%r",
                     etype, list(event.keys()))

    def extract_agent_text(self, event: dict) -> str | None:
        etype = event.get("type") or event.get("event") or ""
        # Typed tool/file/error events never carry the agent's final reply.
        if etype in self._NON_AGENT_TYPES:
            return None
        return self._assistant_text(event, etype)

    # -- helpers ----------------------------------------------------------

    def _assistant_text(self, event: dict, etype: str) -> str | None:
        """Return the event's text content if it looks like assistant output."""
        role = event.get("role")
        if role and role not in ("assistant", "model", "agent"):
            return None
        is_assistant_type = (
            etype in self._ASSISTANT_TYPES
            or etype == ""
            or etype == "item.completed"
        )
        if not is_assistant_type:
            return None
        return self._extract_text(event)

    @staticmethod
    def _extract_text(event: dict) -> str | None:
        """Pull text content from an event using best-effort key probing."""
        # Copilot CLI 1.0.70 wraps streamed assistant output in an event
        # envelope: ``{"type": "assistant.message", "data":
        # {"content": "..."}}``.  Older builds put the content directly
        # on the event, so accept both shapes.
        payloads = [event]
        data = event.get("data")
        if isinstance(data, dict):
            payloads.append(data)
        for payload in payloads:
            for key in ("text", "content", "message"):
                val = payload.get(key)
                if isinstance(val, str) and val.strip():
                    return val
                if isinstance(val, dict):
                    inner = val.get("text") or val.get("content")
                    if isinstance(inner, str) and inner.strip():
                        return inner
        return None

    @staticmethod
    def _summarize_tool(tool: str, inp: dict) -> str:
        if not isinstance(inp, dict):
            return tool
        # Common patterns: {command, cmd} for shells; {path, file} for edits
        cmd = inp.get("command") or inp.get("cmd")
        if cmd:
            return f"$ {cmd}"
        path = inp.get("path") or inp.get("file") or inp.get("filename")
        if path:
            return f"{tool}: {path}"
        return tool


def _parse_acp_model_options(payload: object) -> tuple[str, ...]:
    """Extract the ACP session's account-aware model selector.

    Copilot's session metadata has an ``availableModels`` list with exact
    model IDs. ACP's standard ``configOptions`` model selector is retained
    as a compatibility fallback for builds that do not expose that metadata.
    """
    if not isinstance(payload, dict):
        return ()

    models_metadata = payload.get("models")
    if isinstance(models_metadata, dict):
        models = _catalog_model_ids(
            models_metadata.get("availableModels"), key="modelId",
        )
        if models:
            return models

    config_options = payload.get("configOptions")
    if not isinstance(config_options, list):
        return ()

    for option in config_options:
        if not isinstance(option, dict):
            continue
        category = option.get("category")
        option_id = option.get("id")
        is_model_selector = (
            isinstance(category, str) and category.casefold() == "model"
        ) or (
            isinstance(option_id, str) and option_id.casefold() == "model"
        )
        if not is_model_selector or option.get("type") != "select":
            continue
        values = option.get("options")
        models = _catalog_model_ids(values, key="value")
        if models:
            return models
    return ()


def _catalog_model_ids(values: object, *, key: str) -> tuple[str, ...]:
    """Normalize a structured ACP model list and always lead with auto."""
    if not isinstance(values, list):
        return ()

    models: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        model = value.get(key)
        if not isinstance(model, str):
            continue
        model = model.strip()
        if model and model not in seen:
            seen.add(model)
            if model != "auto":
                models.append(model)
    # Auto is an official Copilot model-selection sentinel. It remains
    # available even when an ACP catalog lists concrete models only.
    return ("auto", *models) if models or "auto" in seen else ()


def _read_acp_stdout(
    stdout: object, messages: queue.Queue[str | None],
) -> None:
    """Move ACP's line-delimited stdout into a timeout-capable queue."""
    # ``stdout`` is a TextIOWrapper from Popen. Keep this tiny adapter loosely
    # typed so tests can provide an in-memory stream without a subprocess.
    try:
        readline = getattr(stdout, "readline", None)
        if not callable(readline):
            return
        while True:
            try:
                line = readline()
            except (OSError, ValueError):
                return
            if not line:
                return
            if isinstance(line, str):
                messages.put(line)
    finally:
        # EOF lets the requester fail immediately instead of waiting out the
        # whole discovery timeout after a CLI startup error.
        messages.put(None)


def _acp_notification(
    proc: subprocess.Popen[str], method: str, params: dict,
) -> bool:
    """Send an ACP notification (a JSON-RPC message without an id)."""
    if proc.stdin is None:
        return False
    try:
        proc.stdin.write(json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }) + "\n")
        proc.stdin.flush()
    except (BrokenPipeError, OSError):
        return False
    return True


def _acp_request(
    proc: subprocess.Popen[str],
    messages: queue.Queue[str | None],
    *,
    request_id: int,
    method: str,
    params: dict,
    deadline: float | None = None,
) -> dict | None:
    """Send one ACP request and return its object result before timeout."""
    if proc.stdin is None:
        return None
    try:
        proc.stdin.write(json.dumps({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }) + "\n")
        proc.stdin.flush()
    except (BrokenPipeError, OSError):
        return None

    if deadline is None:
        deadline = time.monotonic() + _MODEL_DISCOVERY_TIMEOUT_SEC
    for _ in range(_MAX_ACP_MESSAGES_PER_REQUEST):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            line = messages.get(timeout=remaining)
        except queue.Empty:
            return None
        if line is None:
            return None
        try:
            message = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(message, dict):
            continue
        if message.get("id") == request_id:
            result = message.get("result")
            return result if isinstance(result, dict) else None
        # A metadata-only handshake should not need a client request. Reject
        # one explicitly rather than leave the ACP server blocked on an
        # unhandled request until this probe's timeout.
        if "method" in message and "id" in message:
            _reject_acp_request(proc, message.get("id"))
    return None


def _reject_acp_request(proc: subprocess.Popen[str], request_id: object) -> None:
    if proc.stdin is None:
        return
    try:
        proc.stdin.write(json.dumps({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32601,
                "message": "Cozter model discovery supports no ACP callbacks",
            },
        }) + "\n")
        proc.stdin.flush()
    except (BrokenPipeError, OSError):
        return


def _close_acp_session_if_supported(
    proc: subprocess.Popen[str],
    messages: queue.Queue[str | None],
    initialized: dict,
    session: dict,
    *,
    deadline: float,
) -> None:
    """Best-effort close of the empty ACP session used for discovery."""
    capabilities = initialized.get("agentCapabilities")
    session_capabilities = (
        capabilities.get("sessionCapabilities")
        if isinstance(capabilities, dict) else None
    )
    if not isinstance(session_capabilities, dict):
        return
    if "close" not in session_capabilities:
        return
    session_id = session.get("sessionId")
    if not isinstance(session_id, str) or not session_id:
        return
    _acp_request(
        proc,
        messages,
        request_id=3,
        method="session/close",
        params={"sessionId": session_id},
        deadline=deadline,
    )


def _stop_acp_process(
    proc: subprocess.Popen[str], *, kill_tree: bool = False,
) -> None:
    """Close and terminate a short-lived catalog-only ACP process."""
    if proc.stdin is not None:
        try:
            proc.stdin.close()
        except OSError:
            pass
    if kill_tree:
        # A .cmd shim runs beneath cmd.exe. Terminating only that parent can
        # leave its Copilot/Node child alive on Windows, so tear down the
        # process tree rooted at the PID we created above.
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        try:
            proc.kill()
            proc.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            pass
    for stream in (proc.stdout, proc.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
