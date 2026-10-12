"""GitHub Copilot CLI backend.

Copilot's non-interactive mode (`copilot --output-format json` with stdin)
streams JSONL events to stdout. Cozter's Copilot integration:
  - reads a redirected stdin prompt when `-p` is omitted. File-backed stdin
    preserves the full UTF-8 prompt, including newlines, without platform
    argv limits or Windows cmd.exe/.cmd quoting and forwarding changes.
  - supports `-C <dir>`, but Cozter uses the subprocess `cwd` parameter so
    the launch path stays compatible with older CLI versions
  - requires an explicit ``--allow-all-tools`` for autonomous headless work;
    restricted calls use ``--available-tools ""`` to expose no tools at all
  - can export a session to GitHub web and mobile; Cozter explicitly disables
    that for its private, workspace-owned runs

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
import tempfile
import threading
import time

from .base import (
    MODEL_CATALOG_TTL_SEC, AgentResult, Backend, ChatEvent, ProcessResourceMap,
    append_text_result, attachment_image_paths,
    create_prompt_subprocess, executable_command, fresh_model_catalog,
    record_backend_error, summarize_cli_tool, truncate_status_text,
)
from ..utils import terminate_windows_process_tree

logger = logging.getLogger(__name__)

_ACP_PROTOCOL_VERSION = 1
_MODEL_DISCOVERY_TIMEOUT_SEC = 12
_MODEL_FAILURE_RETRY_SEC = 15
_MAX_ACP_MESSAGES_PER_REQUEST = 100
# Bound provider-controlled catalog data like HTTP backends.
_MAX_ACP_MODEL_ID_CHARS = 512
_MAX_ACP_MODEL_OPTIONS = 4_096
# Also bound node count/depth.
_MAX_ACP_OPTION_NODES = _MAX_ACP_MODEL_OPTIONS * 4
_MAX_ACP_OPTION_GROUP_DEPTH = 16
# Bound one JSON-RPC line.
_MAX_ACP_LINE_CHARS = 1 * 1024 * 1024
_COPILOT_HOME_FILES = ("config.json", "settings.json")
# Short-lived discovery caches, bounded.
_MAX_WORKSPACE_MODEL_CACHE_ENTRIES = 64

# ``auto`` always works; don't fall back to help names.
_FALLBACK_MODELS = ("auto",)

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
    supports_vision = True
    vision_mode = "cli_file_flag"
    # Restricted modes expose no tools.
    permission_arg_sets = {
        "full": ("--yolo",),
        "auto": ("--allow-all-tools",),
        "restricted": ("--available-tools", ""),
    }
    # ``auto`` is the policy-aware pre-catalog default.
    default_model = "auto"
    default_summary_model = "auto"
    # No static tiers: unset tiers use the policy-aware default.
    tier_models: dict[str, str] = {}
    # An ACP list is authoritative; ignore unverified ``extra_models``.
    allow_unverified_extra_models = False
    # effort=0 omits the flag.
    effort_levels = ("minimal", "low", "medium", "high", "xhigh", "max")

    def __init__(self) -> None:
        # Cache ACP per workspace; fail closed to auto.
        self._workspace_model_catalogs: dict[
            str, tuple[tuple[str, ...], float],
        ] = {}
        self._workspace_fallback_expires_at: dict[str, float] = {}
        self._models_lock = threading.Lock()
        self._process_homes = ProcessResourceMap()

    def effort_levels_for_model(self, model: str | None) -> tuple[str, ...]:
        """Return the effort vocabulary supported by a selected model.

        Copilot's policy-aware ``auto`` selector rejects ``--effort``.  It
        is also the implicit selector when no model is supplied.  Explicit,
        account-approved model IDs retain the normal Copilot effort scale.
        """
        if not model or model.strip().casefold() == "auto":
            return ()
        return self.effort_levels

    # discovery

    @property
    def available_models(self) -> tuple[str, ...]:  # type: ignore[override]
        """Named models enabled for the current-directory project.

        Callers that know the selected workspace should use
        :meth:`available_models_for_workspace`; this compatibility property
        retains the base backend interface for non-workspace callers.
        """
        return self.available_models_for_workspace(os.getcwd())

    @staticmethod
    def _workspace_catalog_key(workspace_path: str) -> str:
        """Return a stable cache key for a repository-scoped ACP catalog."""
        return os.path.normcase(os.path.realpath(os.path.abspath(workspace_path)))

    def _prune_workspace_model_caches(
        self, now: float, *, keep_key: str,
    ) -> None:
        """Discard expired and least-fresh workspace discovery cache entries.

        Model catalogs and failed-probe throttles are both keyed by arbitrary
        workspace paths.  Without pruning, a daemon that serves many projects
        retains a path and a cache record for every project until restart.
        This runs only while the discovery lock is held; eviction can merely
        cause a later picker to rediscover a catalog and never changes the
        fail-closed model resolution behavior.
        """
        for cache in (
            self._workspace_model_catalogs,
            self._workspace_fallback_expires_at,
        ):
            def expiry(cache_key: str, _bound: dict = cache) -> float:
                value = _bound[cache_key]
                return value[1] if isinstance(value, tuple) else value

            for cache_key in tuple(cache):
                if cache_key != keep_key and expiry(cache_key) <= now:
                    cache.pop(cache_key, None)
            while len(cache) >= _MAX_WORKSPACE_MODEL_CACHE_ENTRIES:
                oldest_key = min(cache, key=expiry)
                cache.pop(oldest_key, None)

    def _cached_workspace_models(
        self, workspace_key: str, now: float,
    ) -> tuple[str, ...] | None:
        """Return a fresh catalog or a throttled fail-closed fallback."""
        catalog = self._workspace_model_catalogs.get(workspace_key)
        cached = (
            fresh_model_catalog(*catalog, now=now)
            if catalog is not None
            else None
        )
        if cached is not None:
            return cached
        if now < self._workspace_fallback_expires_at.get(workspace_key, 0.0):
            return _FALLBACK_MODELS
        return None

    def available_models_for_workspace(
        self, workspace_path: str,
    ) -> tuple[str, ...]:
        """Named models enabled for the authenticated Copilot account.

        ACP's ``session/new`` response contains the account-aware model
        selector. The generic CLI help catalog is intentionally never used,
        because it can advertise models this account cannot select. The ACP
        session uses the requested workspace as its current directory, so
        repository policy is included too. If ACP cannot produce a structured
        selector, return only ``auto``.
        """
        workspace_key = self._workspace_catalog_key(workspace_path)
        now = time.monotonic()
        cached = self._cached_workspace_models(workspace_key, now)
        if cached is not None:
            return cached

        with self._models_lock:
            now = time.monotonic()
            cached = self._cached_workspace_models(workspace_key, now)
            if cached is not None:
                return cached

            self._prune_workspace_model_caches(now, keep_key=workspace_key)
            # Failed refresh falls back to ``auto``.
            self._workspace_model_catalogs.pop(workspace_key, None)
            models = self._discover_models(workspace_path)
            if models is not None:
                self._workspace_model_catalogs[workspace_key] = (
                    models, time.monotonic() + MODEL_CATALOG_TTL_SEC,
                )
                self._workspace_fallback_expires_at.pop(workspace_key, None)
                return models
            # Short throttle: avoid per-input spawns.
            self._workspace_fallback_expires_at[workspace_key] = (
                time.monotonic() + _MODEL_FAILURE_RETRY_SEC
            )
            return _FALLBACK_MODELS

    def resolve_configured_model(self, model: str) -> str:
        """Resolve a stored model for a non-workspace caller's CWD."""
        return self.resolve_configured_model_for_workspace(model, os.getcwd())

    def resolve_configured_model_for_workspace(
        self, model: str, workspace_path: str,
    ) -> str:
        """Keep stored Copilot choices inside a fresh ACP catalog.

        This intentionally does not launch ACP from a chat turn. Until a
        picker has obtained a fresh account catalog (or after its short TTL),
        ``auto`` is safer than forwarding a possibly policy-disabled stored
        model ID to the CLI.
        """
        catalog = self._workspace_model_catalogs.get(
            self._workspace_catalog_key(workspace_path),
        )
        if catalog is None:
            return self.default_model
        models, expires_at = catalog
        if (
            isinstance(model, str)
            and time.monotonic() < expires_at
            and model in models
        ):
            return model
        return self.default_model

    def _discover_models(self, workspace_path: str) -> tuple[str, ...] | None:
        binary = shutil.which(self.executable)
        if binary is None:
            logger.debug("copilot not on PATH; model picker is auto-only")
            return None
        prefix = executable_command(self.executable)
        # Both cmd shims and direct Node loaders can own a native CLI child.
        # Catalog-only launches own all their descendants on Windows.
        owns_windows_tree = os.name == "nt"

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
                    "--no-remote-export",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # ACP speaks on stdout only; DEVNULL stderr.
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=workspace_path,
                env=env,
            )
            if proc.stdout is None:
                logger.debug("copilot ACP started without stdout")
                return None

            # The reader runs independently of the request consumer. Bound
            # queued output as well as each line so a noisy probe cannot
            # retain an unlimited number of otherwise valid messages.
            messages: queue.Queue[str | None] = queue.Queue(
                maxsize=_MAX_ACP_MESSAGES_PER_REQUEST,
            )
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
                params={"cwd": workspace_path, "mcpServers": []},
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

            # Catalog-only session: free session resources when supported.
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
                _stop_acp_process(proc, kill_tree=owns_windows_tree)
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
        # Resolve images before launch; prompt bytes stay intact on stdin.
        image_paths = (
            attachment_image_paths(prompt, workspace_path)
            if self.supports_vision and not compaction else []
        )
        prefix = executable_command(self.executable)
        cmd: list[str] = [
            *prefix,
            "--output-format",
            "json",
            "--no-color",
            # Keep this private session local; never export it remotely.
            "--no-remote-export",
        ]
        self.append_launch_options(cmd, model, effort, approval)

        # Native vision via --attachment flags.
        for _image_path in image_paths:
            cmd += ["--attachment", _image_path]

        isolated_home = _create_isolated_copilot_home()
        env = os.environ.copy()
        env["COPILOT_HOME"] = isolated_home
        try:
            proc = await create_prompt_subprocess(
                cmd, prompt,
                cwd=workspace_path,
                env=env,
            )
        except BaseException:
            _remove_isolated_copilot_home(isolated_home)
            raise

        # Key by Process, not PID.
        self._process_homes.remember(proc, isolated_home)
        return proc

    async def cleanup_process(
        self, proc: asyncio.subprocess.Process,
    ) -> None:
        """Remove this launch's private Copilot home after it exits."""
        home = self._process_homes.pop(proc)
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
        if not isinstance(event, dict):
            return
        etype = event.get("type") or event.get("event") or ""
        if not isinstance(etype, str):
            return

        # Typed branches first.
        if etype in self._TOOL_USE_TYPES:
            tool = (
                event.get("name")
                or event.get("tool")
                or event.get("tool_name")
                or "?"
            )
            inp = event.get("input") or event.get("args") or {}
            result.events.append(ChatEvent(
                kind="tool", content=summarize_cli_tool(tool, inp),
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
            record_backend_error(result, msg)
            return

        # Fall through on assistant-like text.
        text = self._assistant_text(event, etype)
        if text:
            append_text_result(result, text)
            return

        logger.debug("Copilot: unhandled event type=%r keys=%r",
                     etype, list(event.keys()))

    def extract_agent_text(self, event: dict) -> str | None:
        if not isinstance(event, dict):
            return None
        etype = event.get("type") or event.get("event") or ""
        if not isinstance(etype, str):
            return None
        # Tool/file/error events never carry the final reply.
        if etype in self._NON_AGENT_TYPES:
            return None
        return self._assistant_text(event, etype)

    # event helpers

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
        # Accept both pre/post-1.0.70 shapes.
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


def _parse_acp_model_options(payload: object) -> tuple[str, ...]:
    """Extract the ACP session's account-aware model selector.

    Copilot has exposed the legacy session model state in both
    ``models.availableModels`` and top-level ``availableModels`` forms. Both
    contain exact, account- and workspace-policy-scoped IDs, so prefer either
    one over ACP's standard ``configOptions`` model selector. The category is
    optional, so recognize the protocol's usual ``model``/``models`` selector
    IDs and labels as well.
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

    # Compat: old ACP SDKs put SessionModelState on session/new.
    models = _catalog_model_ids(payload.get("availableModels"), key="modelId")
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
        option_name = option.get("name")
        is_model_selector = (
            isinstance(category, str) and category.casefold() == "model"
        ) or (
            isinstance(option_id, str)
            and option_id.casefold() in {"model", "models"}
        ) or (
            isinstance(option_name, str)
            and option_name.casefold() in {"model", "models"}
        )
        if not is_model_selector or option.get("type") != "select":
            continue
        values = option.get("options")
        models = _catalog_model_ids(values, key="value")
        if models:
            return models
    return ()


def _catalog_model_ids(values: object, *, key: str) -> tuple[str, ...]:
    """Normalize a structured ACP model list and always lead with auto.

    ACP permits select options to be grouped (for example, by provider). A
    grouped catalog is still account-authoritative, so preserve its declared
    order while flattening its nested ``options`` lists.
    """
    if not isinstance(values, list):
        return ()

    models: list[str] = []
    seen: set[str] = set()
    inspected = 0
    # Iterative DFS: preserves order without copying lists or recursing.
    pending: list[tuple[list, int, int]] = [(values, 0, 0)]
    while pending and len(models) < _MAX_ACP_MODEL_OPTIONS:
        options, index, depth = pending[-1]
        if index >= len(options):
            pending.pop()
            continue
        pending[-1] = (options, index + 1, depth)

        inspected += 1
        if inspected > _MAX_ACP_OPTION_NODES:
            logger.debug(
                "copilot ACP model catalog exceeded %d option nodes",
                _MAX_ACP_OPTION_NODES,
            )
            break

        value = options[index]
        if not isinstance(value, dict):
            continue
        model = value.get(key)
        if isinstance(model, str):
            model = model.strip()
            if (
                model
                and len(model) <= _MAX_ACP_MODEL_ID_CHARS
                and model not in seen
            ):
                seen.add(model)
                if model != "auto":
                    models.append(model)

        # Grouped-select form nests a list under ``options``.
        nested = value.get("options")
        if isinstance(nested, list) and depth < _MAX_ACP_OPTION_GROUP_DEPTH:
            pending.append((nested, 0, depth + 1))
    # ``auto`` stays available.
    return ("auto", *models) if models or "auto" in seen else ()


def _read_acp_stdout(
    stdout: object, messages: queue.Queue[str | None],
) -> None:
    """Move ACP's line-delimited stdout into a timeout-capable queue."""
    # Loosely typed so tests can stub the stream.
    try:
        readline = getattr(stdout, "readline", None)
        if not callable(readline):
            return
        while True:
            try:
                line = readline(_MAX_ACP_LINE_CHARS + 1)
            except (OSError, ValueError):
                return
            if not line:
                return
            if isinstance(line, str):
                if len(line) > _MAX_ACP_LINE_CHARS:
                    logger.debug(
                        "copilot ACP stdout line exceeded %d characters",
                        _MAX_ACP_LINE_CHARS,
                    )
                    return
                try:
                    messages.put_nowait(line)
                except queue.Full:
                    logger.debug("copilot ACP stdout queue exceeded its limit")
                    return
    finally:
        # EOF fails fast instead of waiting out the discovery timeout.
        try:
            messages.put_nowait(None)
        except queue.Full:
            # A request consumes at most this queue's capacity, then fails.
            # Never block the reader after discovery has already stopped.
            pass


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
        # Handshake needs no request; reject explicitly.
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
        # Windows cmd shims and Node loaders may own a native CLI child.
        terminate_windows_process_tree(proc.pid)
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
