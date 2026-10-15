"""Shared in-process agent loop for OpenAI-compatible chat backends.

llama-server, Z.ai (GLM), and any other server that speaks OpenAI-style
``/chat/completions`` with streaming + tool calls share one tool-calling
loop: send messages + tools, execute independent read-only ``tool_calls``
in bounded batches while preserving state-changing tool barriers,
append the results, and re-call until the model stops calling tools. The
loop runs inside a fake ``asyncio.subprocess.Process`` (HttpAgentProcess)
so agent.py consumes its events the same way it does the CLI backends'.

:class:`OpenAIChatBackend` owns everything provider-agnostic (the loop,
event parsing, streaming client). A concrete backend supplies only the
differences via hooks: the chat endpoint URL, auth headers, how a model
name is resolved, and the per-loop safety limits.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import math
import os
import random
import time
import urllib.parse
import urllib.request
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

import aiohttp

from .. import agent_tools as tools
from ..agent_tools.builtin.bash import shell_context
from ..execution_state import current_run
from ..utils import clip_status_value, iter_bounded_lines
from ._http_process import HttpAgentProcess, http_error_translator
from .base import (
    DOC_FOLLOWING_RULE, PROGRESS_RULE, VISION_RULE, WHOLE_SCOPE_RULE,
    AgentResult, Backend, CachedModelCatalog, ChatEvent, append_text_result,
    attachment_image_paths, record_error_event, vision_image_url_parts,
)

logger = logging.getLogger(__name__)
performance_logger = logging.getLogger("Cozter.performance")

# One SSE event can exceed the readline limit.
_MAX_SSE_LINE_BYTES = 4 * 1024 * 1024
# SSE allows many data: lines; cap the aggregate.
_MAX_SSE_EVENT_BYTES = 4 * 1024 * 1024
# Bound retained completion state.
_MAX_COMPLETION_TEXT_BYTES = 4 * 1024 * 1024
_MAX_COMPLETION_REASONING_BYTES = 4 * 1024 * 1024
_MAX_TOOL_ARGUMENT_BYTES = 4 * 1024 * 1024
_MAX_COMPLETION_BUFFER_BYTES = 8 * 1024 * 1024
# Cap call ids/names too.
_MAX_TOOL_CALL_ID_CHARS = 512
_MAX_TOOL_NAME_CHARS = 512
# Cap the retained conversation.
_MAX_AGENT_MESSAGE_BYTES = 32 * 1024 * 1024
# Also bounds bookkeeping for malformed index-spam streams.
_MAX_TOOL_CALLS_PER_COMPLETION = 128
# Bound discovery responses.
_MAX_MODEL_DISCOVERY_BYTES = 1 * 1024 * 1024
# Reject absurd ID lengths.
_MAX_MODEL_ID_CHARS = 512
_MAX_MODEL_IDS = 4_096
_HTTP_CONNECT_TIMEOUT = 30.0
_COMPLETION_TAIL_DRAIN_SECONDS = 0.05
_COMPLETION_TAIL_DRAIN_BYTES = 64 * 1024
_BLOCKED_TOOL_ROUND_LIMIT = 3
_PARALLEL_READ_TOOLS = 4
# 3600s real-work cap; bound error bodies.
_MAX_HTTP_ERROR_BODY_BYTES = 8 * 1024
_EOF_SUCCESS_FINISH_REASONS = frozenset({
    "stop", "length", "tool_calls", "function_call", "content_filter",
})


def extract_model_ids(payload: object) -> tuple[str, ...]:
    """Extract unique, non-empty IDs from an OpenAI-style ``/models`` payload.

    Shared by the HTTP backends (llama-server, Meta Model API, Z.ai) so the
    parsing of ``{"data": [{"id": ...}, ...}`` lives in one place. IDs are
    stripped of surrounding whitespace; duplicates and empties are dropped
    while preserving first-seen order.
    """
    if not isinstance(payload, dict):
        return ()
    data = payload.get("data")
    if not isinstance(data, list):
        return ()

    ids: list[str] = []
    seen: set[str] = set()
    for entry in data:
        if len(ids) >= _MAX_MODEL_IDS:
            break
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if not isinstance(model_id, str):
            continue
        model_id = model_id.strip()
        if (
            model_id
            and len(model_id) <= _MAX_MODEL_ID_CHARS
            and model_id not in seen
        ):
            seen.add(model_id)
            ids.append(model_id)
    return tuple(ids)


def fetch_model_ids(
    url: str,
    *,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Fetch and parse an OpenAI-style ``/models`` catalog.

    The HTTP backends use the same small, synchronous discovery
    request. Keeping it here avoids providers drifting in response decoding
    or model-ID normalization while still letting each backend choose its URL,
    timeout, authentication headers, and fallback behavior.

    Only ``http``/``https`` catalog URLs are fetched: a hand-edited
    ``*_base_url`` reaching a ``file:``/custom-scheme URL must fail loudly
    instead of reading local files through this probe.
    """
    try:
        scheme = urllib.parse.urlsplit(url).scheme.lower()
    except ValueError:
        scheme = ""
    if scheme not in ("http", "https"):
        raise ValueError(f"refusing to fetch model catalog from {url!r}")
    request: str | urllib.request.Request = url
    if headers:
        request = urllib.request.Request(url, method="GET")
        for name, value in headers.items():
            # Auth must never follow redirects.
            request.add_unredirected_header(name, value)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read(_MAX_MODEL_DISCOVERY_BYTES + 1)
    if len(body) > _MAX_MODEL_DISCOVERY_BYTES:
        raise ValueError(
            "model catalog response exceeded "
            f"{_MAX_MODEL_DISCOVERY_BYTES} byte limit"
        )
    payload = json.loads(body.decode("utf-8"))
    return extract_model_ids(payload)


def discover_bearer_models(
    *,
    label: str,
    api_key: str,
    base_url: str,
    timeout: float,
    fallback_models: tuple[str, ...],
    keep_model_id: Callable[[object], bool] | None = None,
    fetch: Callable[..., tuple[str, ...]] = fetch_model_ids,
) -> tuple[str, ...]:
    """Discover an OpenAI-style ``/models`` catalog with a Bearer key.

    Shared by the Meta Model API and Z.ai backends: without a key the
    curated *fallback_models* are returned without a network probe; a
    failed probe or an empty-after-filter catalog also falls back.
    *keep_model_id* filters known non-chat IDs from the live catalog while
    allowing unknown account-specific IDs; ``None`` keeps every ID.
    *fetch* is the catalog probe (default :func:`fetch_model_ids`);
    backends pass their module-level alias so tests can keep patching
    the per-backend ``fetch_model_ids`` name.
    """
    if not api_key:
        logger.debug(
            "%s API key is unset; using fallback model list", label,
        )
        return fallback_models

    url = base_url.rstrip("/") + "/models"
    try:
        model_ids = fetch(
            url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )
    except Exception as exc:
        logger.debug(
            "Could not query %s models at %s (%s); using fallback",
            label, url, exc,
        )
        return fallback_models

    if keep_model_id is not None:
        chat_models = tuple(
            model_id for model_id in model_ids if keep_model_id(model_id)
        )
    else:
        chat_models = model_ids
    if not chat_models and model_ids:
        logger.debug(
            "%s model catalog contained no chat-completion models; "
            "using fallback",
            label,
        )
    return chat_models or fallback_models


def _compact_json(payload: object) -> str:
    """Serialize requests without optional JSON whitespace or changed escaping."""
    return json.dumps(payload, separators=(",", ":"))


def _message_payload_bytes(message: dict[str, Any]) -> int:
    """Return the approximate UTF-8 JSON size retained for one chat message."""
    # Match aiohttp escaping so non-ASCII can't evade.
    return len(_compact_json(message).encode("utf-8"))


def _is_blocked_tool_result(result: str) -> bool:
    """Recognize dispatcher refusals, not ordinary fixable tool failures."""
    if result.startswith(("Blocked:", "Unknown tool:")):
        return True
    path_refusals = (
        "Error: path escapes workspace:", "Glob failed: path escapes workspace:",
        "List failed: path escapes workspace:", "Grep failed: path escapes workspace:",
        "Copy failed: path escapes workspace:", "Move failed: path escapes workspace:",
        "Create directory failed: path escapes workspace:",
    )
    if result.startswith(path_refusals):
        return True
    if result.startswith("Tool ") and " failed: path escapes workspace:" in result:
        return True
    # apply_patch can report several files. Treat only an entirely refused
    # patch as stalled; one successfully applied file is useful progress.
    lines = result.splitlines()
    return bool(lines) and all(
        ": skipped (path escapes workspace:" in line and line.endswith(")")
        for line in lines
    )


def _parallel_read_tool(name: str) -> bool:
    """Only trusted builtins can cross another read in the same batch."""
    tool = tools._BY_NAME.get(name)
    return (
        tool is not None and not tool.is_plugin
        and name in tools.READ_ONLY_TOOL_NAMES
    )


def _restore_http_messages(
    saved: object, pending: object,
) -> list[dict[str, Any]]:
    """Close interrupted tool batches without replaying their side effects.

    Results are checkpointed individually and may complete out of order. The
    provider transcript always receives them in the original tool-call order.
    Missing outcomes require fresh verification, rather than assuming that a
    write or command failed when cancellation raced its completion.
    """
    if saved is None:
        return []
    if not isinstance(saved, list):
        raise RuntimeError("Saved HTTP transcript is invalid; preserve it for recovery.")
    messages: list[dict[str, Any]] = []
    byte_count = 0
    for message in saved:
        if not isinstance(message, dict) or message.get("role") not in {
            "user", "assistant", "tool",
        }:
            raise RuntimeError("Saved HTTP transcript contains an invalid message.")
        try:
            byte_count += _message_payload_bytes(message)
        except (TypeError, ValueError):
            raise RuntimeError("Saved HTTP transcript is not serializable.") from None
        if byte_count > _MAX_AGENT_MESSAGE_BYTES:
            raise RuntimeError("Saved HTTP transcript exceeds its safety limit.")
        messages.append(copy.deepcopy(message))
    if not messages or messages[-1].get("role") != "assistant":
        return messages
    calls = messages[-1].get("tool_calls")
    if not isinstance(calls, list):
        return messages
    results = pending if isinstance(pending, dict) else {}
    for position, call in enumerate(calls):
        if not isinstance(call, dict):
            raise RuntimeError("Saved HTTP transcript contains an invalid tool call.")
        name, _ = tools.parse_openai_call(call)
        result = results.get(str(position))
        if not isinstance(result, str):
            result = (
                "Interrupted: this tool batch ended before a durable result"
                " was recorded for this call. Its outcome is unknown; it may"
                " already have changed files or started a process. Inspect"
                " the actual files, process state, and existing logs before"
                " deciding what remains. Do not automatically replay this"
                " call or repeat completed side effects."
            )
        messages.append({
            "role": "tool", "tool_call_id": call.get("id", ""),
            "name": name, "content": result,
        })
    return messages


class OpenAIChatBackend(Backend):
    """Backend that drives an OpenAI-compatible chat-completions endpoint.

    Subclasses override the hooks below; the loop, event parsing, and
    streaming client are inherited.
    """

    supports_read_only_workers = True
    # OpenAI-shape tools consumed directly.
    supports_typed_plugins = True
    supports_vision = True
    vision_mode = "openai_parts"
    # Standard 4-level effort words.
    effort_levels: tuple[str, ...] = ("minimal", "low", "medium", "high")

    def _chat_endpoint(self) -> str:
        """Full URL of the chat/completions endpoint."""
        raise NotImplementedError

    def _auth_headers(self) -> dict[str, str]:
        """Extra HTTP headers (e.g. Authorization). Empty for local servers."""
        return {}

    def _request_model(self, model: str | None) -> str | None:
        """Model id to send, or None to omit the field entirely."""
        return model or None

    def _effort_fields(
        self,
        percent: int,
        _model: str | None = None,
    ) -> dict[str, Any]:
        """Provider-specific request fields for a reasoning-effort override.

        OpenAI-compatible servers conventionally accept
        ``reasoning_effort``. Providers with a different request shape can
        override this hook without duplicating the agent loop. ``model`` is
        the resolved request model so providers can account for model-specific
        parameter support.
        """
        native_effort = self.convert_effort(percent)
        return {"reasoning_effort": native_effort} if native_effort else {}

    def _tool_request_fields(self, _model: str | None) -> dict[str, Any]:
        """Return provider fields for a streaming completion with tools.

        Most OpenAI-compatible servers need no extra fields. Providers that
        expose opt-in streaming for incremental tool-call arguments can
        override this hook without sending an unsupported field on ordinary
        chat or the final no-tools completion.
        """
        return {}

    def _supports_tools_for_model(self, _model: str | None) -> bool:
        """Whether the selected model accepts this agent's function schema.

        OpenAI-compatible backends normally support function tools for every
        selected model. A provider can opt out a documented chat-only model
        without exposing a tool schema it will reject; the model still gets a
        normal text-only agent turn.
        """
        return True

    def _preserve_reasoning_content(self, _model: str | None) -> bool:
        """Whether assistant reasoning must be retained across tool turns.

        Some OpenAI-compatible providers require their opaque reasoning
        blocks to be returned unmodified with the following tool result.
        Keep that provider-specific message field opt-in: ordinary
        OpenAI-compatible servers must continue receiving the standard
        assistant/tool transcript shape.
        """
        return False

    def _preserved_reasoning_request_fields(
        self,
        _model: str | None,
        _effort_fields: dict[str, Any],
    ) -> dict[str, Any]:
        """Return provider fields needed when preserving reasoning blocks."""
        return {}

    def _max_agent_turns(self) -> int:
        return 40

    def _tool_repeat_limit(self) -> int:
        return 3

    def _auto_continue_after_tool_limit(self) -> bool:
        return False

    def _max_segments(self) -> int:
        """Hard cap on auto-continue segments.

        Only relevant when :meth:`_auto_continue_after_tool_limit` is True:
        without it the segment loop has no exit when the model keeps calling
        tools and never returns a tool-free answer, so it would re-plan
        forever (unbounded API calls, unbounded message growth). Each segment
        is up to :meth:`_max_agent_turns` tool-call turns, so the default
        bounds a run at 8 x 40 = 320 tool turns before a final answer is
        forced.
        """
        return 8

    def _socket_timeout(self) -> int | None:
        # Real-work cap hook: subclass socket timeout.
        return 3600

    def _socket_timeout_setting(self) -> str:
        """Config setting to mention in a request error (legacy label)."""
        return "the configured socket timeout"

    def _max_retries(self) -> int:
        return 2

    # No real request; that would bill.

    def _api_key(self) -> str:
        """Configured Bearer key, or ``""`` when unset. Providers override."""
        return ""

    def _api_key_setting(self) -> str:
        """Config key name shown when no API key is configured."""
        return "api_key"

    def health_check(self) -> tuple[bool, str]:
        # Readiness = "key configured?".
        if not self._api_key():
            return False, (
                f"no API key set (set {self._api_key_setting()}"
                " in config.json)"
            )
        return True, (
            f"configured (endpoint {self._chat_endpoint()},"
            f" default model {self.default_model})"
        )

    async def launch(  # type: ignore[override]
        self,
        workspace_path: str,
        prompt: str,
        model: str | None,
        approval: str,
        *,
        compaction: bool = False,
        effort: int = 0,
    ) -> HttpAgentProcess:
        proc = HttpAgentProcess(f"{self.name} agent")
        proc.start(self._run_agent(
            proc, workspace_path, prompt, model, approval, compaction, effort,
        ))
        return proc

    async def _run_agent(
        self,
        proc: HttpAgentProcess,
        workspace_path: str,
        prompt: str,
        model: str | None,
        approval: str,
        compaction: bool,
        effort: int,
    ) -> None:
        endpoint = self._chat_endpoint()
        headers = self._auth_headers()
        sock_read = self._socket_timeout()
        timeout_setting = self._socket_timeout_setting()
        max_retries = self._max_retries()
        # Per-turn caps, read fresh.
        max_agent_turns = self._max_agent_turns()
        max_segments = self._max_segments()
        tool_repeat_limit = self._tool_repeat_limit()
        request_model = self._request_model(model)

        # Approval -> tools.
        effective_approval = "deny" if compaction else approval
        tools_schema = _tools_for_approval(effective_approval, compaction)
        if tools_schema and not self._supports_tools_for_model(request_model):
            tools_schema = None
        enabled_tool_names = tuple(
            entry["function"]["name"] for entry in tools_schema
        ) if tools_schema else ()

        # Translate 0-100 effort into provider fields (empty = server default).
        effort_fields = self._effort_fields(effort, request_model)
        preserve_reasoning = self._preserve_reasoning_content(request_model)

        # Vision backends attach image bytes too.
        user_content: str | list[dict[str, Any]] = prompt
        if (
            getattr(self, "supports_vision", False)
            and not compaction
            and "attachment saved to:" in prompt.casefold()
        ):
            vision_parts = await asyncio.to_thread(
                _vision_parts_for_prompt, prompt, workspace_path,
            )
            if vision_parts is not None:
                user_content = vision_parts

        # The execution record is scoped to this queued job/worker/provider.
        # Compaction calls remain stateless and never acquire tool history.
        execution = None if compaction else current_run()
        identity = hashlib.sha256(_compact_json({
            "backend": self.name,
            "workspace": os.path.normcase(os.path.realpath(workspace_path)),
            "model": request_model,
            "approval": effective_approval,
            "effort": effort_fields,
            "endpoint": endpoint,
        }).encode("utf-8")).hexdigest()
        prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        restored: list[dict[str, Any]] = []
        previous_prompt_digest: object = None
        if execution is not None and execution.get("http_identity") == identity:
            restored = _restore_http_messages(
                execution.get("http_messages"),
                execution.get("http_pending_results"),
            )
            previous_prompt_digest = execution.get("http_prompt_digest")
        messages: list[dict] = [
            {
                "role": "system",
                "content": _system_prompt(
                    workspace_path, enabled_tool_names, effective_approval,
                ),
            },
            *restored,
        ]
        if not restored or previous_prompt_digest != prompt_digest:
            messages.append({"role": "user", "content": user_content})
        message_bytes = sum(_message_payload_bytes(message) for message in messages)
        pending_results: dict[str, str] = {}
        checkpoint_lock = asyncio.Lock()

        async def save_transcript() -> None:
            if execution is not None:
                await execution.save(
                    http_identity=identity,
                    http_messages=copy.deepcopy(messages[1:]),
                    http_prompt_digest=prompt_digest,
                    http_pending_results=dict(pending_results),
                )

        def append_message(message: dict[str, Any]) -> bool:
            """Retain *message* only while the aggregate safety cap permits."""
            nonlocal message_bytes
            message_size = _message_payload_bytes(message)
            if message_bytes + message_size > _MAX_AGENT_MESSAGE_BYTES:
                return False
            messages.append(message)
            message_bytes += message_size
            return True

        def emit_message_limit_error() -> None:
            proc.emit({
                "type": "error",
                "message": (
                    f"{self.name} conversation exceeded the "
                    f"{_MAX_AGENT_MESSAGE_BYTES // (1024 * 1024)} MiB "
                    "limit; the retained transcript is a truncated preview —"
                    " narrow the task or cut tool output, and say PARTIAL +"
                    " remainder when work is left."
                ),
            })

        if message_bytes > _MAX_AGENT_MESSAGE_BYTES:
            emit_message_limit_error()
            return
        tool_repeat_counts: dict[str, int] = {}
        blocked_rounds = 0
        stopped_on_refusals = False
        stopped_on_budget = False
        stop_reason = "Tool limit reached."
        tool_rounds = 0
        has_preserved_reasoning = preserve_reasoning and any(
            message.get("reasoning_content") for message in restored
        )
        # One session per turn.

        def completion_payload(
            tools_schema: list[dict] | None,
        ) -> dict[str, Any]:
            """Build one provider request from the retained transcript."""
            payload = _completion_payload(
                messages, request_model, effort_fields, tools_schema,
            )
            # Preserve reasoning once present.
            if preserve_reasoning and (
                tools_schema is not None or has_preserved_reasoning
            ):
                payload.update(self._preserved_reasoning_request_fields(
                    request_model, effort_fields,
                ))
            if tools_schema is not None:
                payload["tool_choice"] = "auto"
                payload.update(self._tool_request_fields(request_model))
            return payload

        await save_transcript()
        async with aiohttp.ClientSession(json_serialize=_compact_json) as http_session:
            async def stream_completion(payload: dict[str, Any]) -> tuple[
                str, str, list[dict],
            ]:
                return await _stream_completion(
                    endpoint, payload, headers, sock_read, max_retries,
                    self.name, timeout_setting, session=http_session,
                )

            segment = 1
            while True:
                for _ in range(max_agent_turns):
                    payload = completion_payload(tools_schema)

                    assistant_text, reasoning_content, tool_calls = (
                        await stream_completion(payload)
                    )

                    # OpenAI wants content null with tool_calls.
                    assistant_msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": assistant_text if assistant_text else None,
                    }
                    if tool_calls:
                        assistant_msg["tool_calls"] = tool_calls
                        if preserve_reasoning and reasoning_content:
                            # Opaque Z.ai provider state.
                            assistant_msg["reasoning_content"] = (
                                reasoning_content
                            )
                            has_preserved_reasoning = True
                    # Emit per-turn commentary.
                    if assistant_text:
                        proc.emit({
                            "type": "assistant_text",
                            "text": assistant_text,
                        })

                    if not tool_calls:
                        if not append_message(assistant_msg):
                            emit_message_limit_error()
                            return
                        await save_transcript()
                        return
                    tool_rounds += 1

                    # Refuse before running if retaining it breaks the run bound.
                    if not append_message(assistant_msg):
                        emit_message_limit_error()
                        return
                    pending_results.clear()
                    await save_transcript()

                    # Batch consecutive reads; writes and unknown plugins are
                    # ordered barriers even if the provider groups them.
                    _batch_names: list[str] = []
                    _batch_args: list[dict] = []
                    _batch_skipped: list[bool] = []
                    for call in tool_calls:
                        call_name, call_args = tools.parse_openai_call(call)
                        call_signature = tools.tool_signature(call_name, call_args)
                        tool_repeat_counts[call_signature] = (
                            tool_repeat_counts.get(call_signature, 0) + 1
                        )
                        _batch_names.append(call_name)
                        _batch_args.append(call_args)
                        _batch_skipped.append(
                            tool_repeat_counts[call_signature] > tool_repeat_limit,
                        )

                    async def run_tool_call(run_index: int) -> None:
                        # execute_tool owns permissions, status, truncation, timeout.
                        _name = _batch_names[run_index]
                        result = await tools.execute_tool(
                            _name, _batch_args[run_index], workspace_path, effective_approval,
                            proc.emit,
                        )
                        async with checkpoint_lock:
                            _results[run_index] = result
                            pending_results[str(run_index)] = result
                            await save_transcript()

                    _results: list[str | None] = [None] * len(tool_calls)
                    batch_started = time.monotonic()
                    position = 0
                    while position < len(tool_calls):
                        if _batch_skipped[position]:
                            position += 1
                            continue
                        if not _parallel_read_tool(_batch_names[position]):
                            await run_tool_call(position)
                            position += 1
                            continue
                        read_indices: list[int] = []
                        while (
                            position < len(tool_calls)
                            and _parallel_read_tool(_batch_names[position])
                            and len(read_indices) < _PARALLEL_READ_TOOLS
                        ):
                            if not _batch_skipped[position]:
                                read_indices.append(position)
                            position += 1
                        # Bounded task creation also bounds filesystem threads.
                        read_tasks = [
                            asyncio.create_task(run_tool_call(index))
                            for index in read_indices
                        ]
                        try:
                            await asyncio.gather(*read_tasks)
                        except BaseException:
                            # gather does not cancel siblings after an I/O or
                            # checkpoint error. Settle this batch before its
                            # execution record can leave the owning worker.
                            for task in read_tasks:
                                if not task.done():
                                    task.cancel()
                            cleanup = asyncio.gather(*read_tasks, return_exceptions=True)
                            while not cleanup.done():
                                try:
                                    await asyncio.shield(cleanup)
                                except asyncio.CancelledError:
                                    continue
                            raise
                    performance_logger.info(
                        "stage=tool_batch backend=%s seconds=%.3f calls=%d skipped=%d",
                        self.name, time.monotonic() - batch_started,
                        len(tool_calls), sum(_batch_skipped),
                    )
                    batch_results: list[str] = []
                    for _i, call in enumerate(tool_calls):
                        name = _batch_names[_i]
                        if _batch_skipped[_i]:
                            result = (
                                f"Skipped: {name} repeated"
                                f" >{tool_repeat_limit}x. Answer"
                                " from info so far."
                            )
                            proc.emit({
                                "type": "tool_result",
                                "name": name,
                                "output": result,
                            })
                        else:
                            call_result: str | None = _results[_i]
                            result_text: str = (
                                call_result
                                if isinstance(call_result, str)
                                else f"Tool {name} failed: no result"
                            )
                            result = result_text
                        batch_results.append(result)

                        # Include ``name`` alongside tool_call_id.
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": call.get("id", ""),
                            "name": name,
                            "content": result,
                        }
                        if not append_message(tool_message):
                            emit_message_limit_error()
                            return
                    pending_results.clear()
                    await save_transcript()

                    # A skipped-only round cannot add evidence. Do not spend
                    # hundreds of completions asking for the same blocked calls.
                    if all(_batch_skipped):
                        stop_reason = "Repeated tool calls made no further progress."
                        break
                    if all(_is_blocked_tool_result(item) for item in batch_results):
                        blocked_rounds += 1
                    else:
                        blocked_rounds = 0
                    if blocked_rounds >= _BLOCKED_TOOL_ROUND_LIMIT:
                        stop_reason = "Successive tool rounds were blocked."
                        stopped_on_refusals = True
                        break

                if stop_reason != "Tool limit reached.":
                    performance_logger.info(
                        "stage=tool_loop backend=%s tool_rounds=%d outcome=stalled",
                        self.name, tool_rounds,
                    )
                    break

                if not self._auto_continue_after_tool_limit():
                    stopped_on_budget = True
                    break
                if segment >= max_segments:
                    stopped_on_budget = True
                    # Ceiling hit: fall through to a no-tools answer.
                    logger.warning(
                        "%s hit the %d-segment cap (%d tool turns each);"
                        " forcing a final answer",
                        self.name, max_segments, max_agent_turns,
                    )
                    break

                segment += 1
                logger.info(
                    "%s reached %d tool-call turns; continuing segment %d",
                    self.name, max_agent_turns, segment,
                )
                continuation_message = {
                    "role": "user",
                    "content": (
                        "Segment limit reached. Continue with more tools if"
                        " needed; don't repeat completed calls."
                    ),
                }
                if not append_message(continuation_message):
                    emit_message_limit_error()
                    return
                await save_transcript()

            # Fall out of the loop: force one final no-tools response.
            final_request_message = {
                "role": "user",
                "content": (
                    stop_reason + " No more tools: answer from"
                    " collected info; say what's missing if incomplete."
                ),
            }
            if not append_message(final_request_message):
                emit_message_limit_error()
                return
            await save_transcript()

            payload = completion_payload(None)
            assistant_text, _, _ = await stream_completion(payload)

            if assistant_text:
                if not append_message({"role": "assistant", "content": assistant_text}):
                    emit_message_limit_error()
                    return
                await save_transcript()
                proc.emit({"type": "assistant_text", "text": assistant_text})
                if stopped_on_budget:
                    # The final report survives, but another judge must not
                    # restart the same work with a fresh tool allowance.
                    proc.emit({
                        "type": "error",
                        "message": f"Tool budget exhausted after {tool_rounds} tool rounds.",
                    })
                if stopped_on_refusals:
                    # Preserve the report but prevent judge/auto-chain from
                    # retrying a scope that permissions/tool availability block.
                    proc.emit({
                        "type": "error",
                        "message": (
                            f"{self.name} stopped after {_BLOCKED_TOOL_ROUND_LIMIT}"
                            " consecutive rounds where every tool call was refused."
                            " Review effective permission or available tools before retrying."
                        ),
                    })
            else:
                proc.emit({
                    "type": "error",
                    "message": (
                        f"{self.name} agent exceeded {max_agent_turns} "
                        "tool-call turns and failed to produce a final answer."
                    ),
                })

    def parse_event(self, event: dict, result: AgentResult) -> None:
        if not isinstance(event, dict):
            return
        event_type = event.get("type", "")

        if event_type == "assistant_text":
            text = event.get("text")
            if isinstance(text, str) and text:
                # Last text wins.
                append_text_result(result, text)
            return

        if event_type == "tool_use":
            tool = event.get("name", "?")
            if not isinstance(tool, str):
                tool = "?"
            tool_input = event.get("input") or {}
            if not isinstance(tool_input, dict):
                tool_input = {}
            content = tools.summarize_tool_use(tool, tool_input)
            result.events.append(ChatEvent(kind="tool", content=content))
            if event.get("file_action"):
                # write/edit/delete surface as kind="file".
                path = tool_input.get("path", "?")
                action = event["file_action"]
                result.events.append(ChatEvent(
                    kind="file", content=f"📄 {action}: {path}",
                ))
            return

        if event_type == "tool_result":
            # Covered by the tool_use event; skip.
            return

        if record_error_event(event, result):
            return

        logger.debug("%s: unhandled event %r", self.name, event)

    def extract_agent_text(self, event: dict) -> str | None:
        if not isinstance(event, dict):
            return None
        if event.get("type") == "assistant_text":
            text = event.get("text")
            if isinstance(text, str) and text:
                return text
        return None


class CachedOpenAIChatBackend(CachedModelCatalog, OpenAIChatBackend):
    """OpenAI-compatible backend with a short-lived, thread-safe model cache.

    Providers can implement :meth:`_fetch_models` to obtain their live model
    catalog. The common cache avoids a network request each time users open a
    model picker while retaining the existing one-minute refresh cadence.
    """

    @property
    def available_models(self) -> tuple[str, ...]:  # type: ignore[override]
        return self._live_model_catalog()


# Streaming client


def _completion_payload(
    messages: list[dict],
    request_model: str | None,
    effort_fields: dict[str, Any],
    tools_schema: list[dict] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": messages,
        "stream": True,
    }
    if request_model is not None:
        payload["model"] = request_model
    payload.update(effort_fields)
    if tools_schema is not None:
        payload["tools"] = tools_schema
        # Providers can group independent reads; execution preserves writes.
        payload["parallel_tool_calls"] = True
    return payload


class _RetryableError(RuntimeError):
    """A transient failure worth retrying (network / 429 / 5xx)."""

    def __init__(
        self, message: str, *, retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        if (
            retry_after is None or isinstance(retry_after, bool)
            or not isinstance(retry_after, (int, float))
        ):
            retry_after = None
        self.retry_after: float | None = (
            None if retry_after is None else float(retry_after)
        )


class _SSEEventTooLargeError(RuntimeError):
    """A framed SSE event exceeded the aggregate memory limit."""


class _CompletionTooLargeError(RuntimeError):
    """A completion exceeded its retained text, tool, or call-count limit."""


def _check_completion_buffer_limit(
    text_bytes: int,
    reasoning_bytes: int,
    tool_argument_bytes: int,
) -> None:
    """Raise when retained state for one completion exceeds its total cap."""
    if (
        text_bytes + reasoning_bytes + tool_argument_bytes
        > _MAX_COMPLETION_BUFFER_BYTES
    ):
        raise _CompletionTooLargeError(
            "completion buffers exceeded "
            f"{_MAX_COMPLETION_BUFFER_BYTES} byte limit",
        )


def _backoff_delay(
    attempt: int, retry_after: float | None = None,
    *, base: float = 0.5, cap: float = 10.0,
) -> float:
    """Seconds to wait before retry *attempt* (1-based); honors Retry-After."""
    if retry_after is not None:
        if isinstance(retry_after, bool) or not isinstance(
            retry_after, (int, float),
        ):
            retry_after = None
        elif not math.isfinite(retry_after):
            # Never sleep nan/inf.
            retry_after = cap if retry_after == float("inf") else None
        elif retry_after < 0:
            retry_after = None
        if retry_after is not None:
            return min(max(float(retry_after), 0.0), cap)
    if isinstance(attempt, bool) or not isinstance(attempt, int):
        attempt = 1
    attempt = max(1, attempt)
    delay = min(base * (2 ** (attempt - 1)), cap)
    return delay + random.uniform(0.0, delay * 0.25)


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header (delta-seconds form); ignore HTTP dates."""
    if not value or isinstance(value, bool):
        # A bool is not a duration.
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    # Reject nan/-inf; callers cap +inf.
    if not math.isfinite(parsed) and parsed != float("inf"):
        return None
    return max(parsed, 0.0)


async def _stream_completion(
    endpoint: str,
    payload: dict,
    headers: dict[str, str],
    sock_read: int | None,
    max_retries: int,
    label: str,
    timeout_setting: str = "the configured socket timeout",
    session: aiohttp.ClientSession | None = None,
) -> tuple[str, str, list[dict]]:
    """POST the chat/completions endpoint (streaming); retry transient fails.

    Returns ``(text, reasoning_content, tool_calls)``. Connection drops
    and HTTP 429/5xx are retried with exponential backoff up to
    *max_retries* times - retrying a completion is safe because tool side
    effects only run *after* this returns. A bad status or malformed
    response is not retried. Real-work cap: the stream runs up to the
    ``sock_read`` bound (default 3600s), shared across retries and backoff,
    or until the turn is cancelled. Heartbeats cannot extend this budget.
    """
    started = time.monotonic()
    deadline = started + sock_read if sock_read is not None and sock_read > 0 else None
    retries = 0
    outcome = "error"

    async def with_retries() -> tuple[str, str, list[dict]]:
        nonlocal retries
        while True:
            try:
                return await _stream_once(
                    endpoint, payload, headers, sock_read, label,
                    session=session, deadline=deadline,
                )
            except _RetryableError as exc:
                if retries >= max_retries:
                    raise
                retries += 1
                delay = _backoff_delay(retries, exc.retry_after)
                logger.warning(
                    "%s transient failure (attempt %d/%d): %s;"
                    " retrying in %.1fs",
                    label, retries, max_retries, exc, delay,
                )
                await asyncio.sleep(delay)

    try:
        async with http_error_translator(label, sock_read, timeout_setting):
            if sock_read is not None and sock_read > 0:
                result = await asyncio.wait_for(with_retries(), timeout=sock_read)
            else:
                result = await with_retries()
        outcome = "ok"
        return result
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    finally:
        performance_logger.info(
            "stage=http_completion backend=%s model=%s seconds=%.3f retries=%d outcome=%s",
            label, clip_status_value(str(payload.get("model", "default")), 120),
            time.monotonic() - started, retries, outcome,
        )


async def _stream_once(
    endpoint: str,
    payload: dict,
    headers: dict[str, str],
    sock_read: int | None,
    label: str,
    session: aiohttp.ClientSession | None = None,
    deadline: float | None = None,
) -> tuple[str, str, list[dict]]:
    """One streaming attempt; raise _RetryableError for transient failures.

    Parses Server-Sent Events. ``data:`` lines carry JSON deltas;
    ``data: [DONE]`` terminates the stream. Real-work cap: the stream runs
    up to the subclass-provided ``sock_read`` bound (default 3600s) or
    until the turn is cancelled; cancel still stops instantly.
    """
    if sock_read is not None and sock_read > 0:
        if deadline is None:
            deadline = time.monotonic() + sock_read
        timeout = aiohttp.ClientTimeout(
            total=sock_read, sock_read=sock_read,
            connect=min(_HTTP_CONNECT_TIMEOUT, sock_read),
            sock_connect=min(_HTTP_CONNECT_TIMEOUT, sock_read),
        )
    else:
        timeout = aiohttp.ClientTimeout(
            total=None, connect=_HTTP_CONNECT_TIMEOUT,
            sock_connect=_HTTP_CONNECT_TIMEOUT,
        )
    if session is None:
        async with aiohttp.ClientSession(json_serialize=_compact_json) as owned_session:
            return await _stream_once(
                endpoint, payload, headers, sock_read, label,
                session=owned_session, deadline=deadline,
            )
    return await _post_completion_stream(
        session, endpoint, payload, headers, timeout, label, deadline,
    )


async def _post_completion_stream(
    session: aiohttp.ClientSession,
    endpoint: str,
    payload: dict,
    headers: dict[str, str],
    timeout: aiohttp.ClientTimeout,
    label: str,
    deadline: float | None = None,
) -> tuple[str, str, list[dict]]:
    """Read one streaming completion from an existing HTTP session."""
    text_parts: list[str] = []
    text_bytes = 0
    reasoning_parts: list[str] = []
    reasoning_bytes = 0
    tool_argument_bytes = 0
    # Tool calls fragment: accumulate by index.
    tool_buffers: dict[int, dict[str, Any]] = {}
    # EOF alone isn't completion.
    saw_terminal_marker = False
    terminal_finish_reason: str | None = None
    started = time.monotonic()
    first_delta: float | None = None
    outcome = "error"

    try:
        async with session.post(
            endpoint, json=payload, headers=headers or None,
            timeout=timeout,
        ) as resp:
            if resp.status == 429 or resp.status >= 500:
                body = await _read_error_body(resp)
                clipped = clip_status_value(body, 200)
                raise _RetryableError(
                    f"{label} returned HTTP {resp.status}: {clipped}",
                    retry_after=_parse_retry_after(
                        resp.headers.get("Retry-After"),
                    ),
                )
            if resp.status != 200:
                body = await _read_error_body(resp)
                clipped = clip_status_value(body, 500)
                raise RuntimeError(
                    f"{label} returned HTTP {resp.status}: {clipped}"
                )
            async for raw_data in _iter_sse_events(resp.content):
                data = raw_data.strip()
                if data == "[DONE]":
                    saw_terminal_marker = True
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    logger.debug("Non-JSON SSE line: %r", data)
                    continue
                # Ignore malformed events; still surface explicit errors.
                if not isinstance(obj, dict):
                    continue
                if "error" in obj:
                    error = obj["error"]
                    message = (
                        error.get("message")
                        if isinstance(error, dict) else error
                    )
                    if not isinstance(message, str) or not message.strip():
                        message = str(error)
                    clipped_message = clip_status_value(message.strip(), 500)
                    raise RuntimeError(
                        f"{label} stream error: {clipped_message}"
                    )
                choices = obj.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                finish_reason = choice.get("finish_reason")
                terminal_event = False
                if finish_reason == "network_error":
                    raise _RetryableError(
                        f"{label} stream ended with network_error",
                    )
                if isinstance(finish_reason, str) and finish_reason:
                    if finish_reason in _EOF_SUCCESS_FINISH_REASONS:
                        saw_terminal_marker = True
                        terminal_finish_reason = finish_reason
                        terminal_event = True
                    else:
                        # Never run a partial call closed by error + [DONE].
                        raise RuntimeError(
                            f"{label} stream ended with {finish_reason}; "
                            "response was incomplete",
                        )
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    if terminal_event:
                        await _drain_completion_tail(resp, deadline)
                        break
                    continue
                if first_delta is None and any(
                    delta.get(key) for key in ("content", "reasoning_content", "tool_calls")
                ):
                    first_delta = time.monotonic() - started
                content = delta.get("content")
                if isinstance(content, str) and content:
                    text_bytes += len(content.encode("utf-8", errors="replace"))
                    if text_bytes > _MAX_COMPLETION_TEXT_BYTES:
                        raise _CompletionTooLargeError(
                            "completion text exceeded "
                            f"{_MAX_COMPLETION_TEXT_BYTES} byte limit",
                        )
                    _check_completion_buffer_limit(
                        text_bytes, reasoning_bytes, tool_argument_bytes,
                    )
                    text_parts.append(content)
                reasoning_content = delta.get("reasoning_content")
                if isinstance(reasoning_content, str) and reasoning_content:
                    reasoning_bytes += len(
                        reasoning_content.encode("utf-8", errors="replace"),
                    )
                    if reasoning_bytes > _MAX_COMPLETION_REASONING_BYTES:
                        raise _CompletionTooLargeError(
                            "completion reasoning exceeded "
                            f"{_MAX_COMPLETION_REASONING_BYTES} byte limit",
                        )
                    _check_completion_buffer_limit(
                        text_bytes, reasoning_bytes, tool_argument_bytes,
                    )
                    reasoning_parts.append(reasoning_content)
                tool_call_deltas = delta.get("tool_calls")
                if isinstance(tool_call_deltas, list):
                    for tool_call_delta in tool_call_deltas:
                        tool_argument_bytes += _merge_tool_call(
                            tool_buffers, tool_call_delta,
                        )
                        _check_completion_buffer_limit(
                            text_bytes, reasoning_bytes, tool_argument_bytes,
                        )
                # Choice finished: keep its final delta; never retry a stalled tail.
                if terminal_event:
                    await _drain_completion_tail(resp, deadline)
                    break
            if saw_terminal_marker and not (
                terminal_finish_reason in {"length", "content_filter"} and tool_buffers
            ):
                outcome = "ok"
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except (
        aiohttp.ClientConnectorError,
        aiohttp.ServerDisconnectedError,
        aiohttp.ClientPayloadError,
        TimeoutError,
    ) as exc:
        raise _RetryableError(f"{label}: {exc}") from exc
    except (_SSEEventTooLargeError, _CompletionTooLargeError) as exc:
        if terminal_finish_reason == "content_filter":
            # The filtered terminal delta may itself exceed a buffer cap.
            # Its explicit refusal remains deterministic even in that case.
            raise RuntimeError(
                f"{label} stream ended with content_filter; response was incomplete",
            ) from exc
        # Never continue after dropped deltas/caps (retry is safe).
        raise _RetryableError(f"{label}: {exc}") from exc
    finally:
        performance_logger.info(
            "stage=http_attempt backend=%s seconds=%.3f first_delta_seconds=%s outcome=%s",
            label, time.monotonic() - started,
            f"{first_delta:.3f}" if first_delta is not None else "none", outcome,
        )

    if not saw_terminal_marker:
        raise _RetryableError(
            f"{label} stream ended before a completion marker",
        )

    # ``length`` = output cap.
    if terminal_finish_reason == "length" and tool_buffers:
        raise _RetryableError(
            f"{label} stream ended with length while a tool call was pending",
        )

    # A filtered completion can truncate argument JSON mid-call. Never
    # execute it, and do not retry a deterministic provider refusal.
    if terminal_finish_reason == "content_filter" and tool_buffers:
        raise RuntimeError(
            f"{label} stream ended with content_filter while a tool call was pending; "
            "response was incomplete",
        )

    # Normalize buffers into tool_calls.
    tool_calls: list[dict] = []
    for idx in sorted(tool_buffers.keys()):
        buf = tool_buffers[idx]
        function = buf.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            continue
        argument_parts = buf.pop("_argument_parts", [])
        buf.pop("_argument_bytes", None)
        if isinstance(argument_parts, list):
            function["arguments"] = "".join(argument_parts)
        tool_calls.append(buf)
    return "".join(text_parts), "".join(reasoning_parts), tool_calls


async def _drain_completion_tail(
    resp: aiohttp.ClientResponse, deadline: float | None,
) -> None:
    """Keep normal completed responses reusable without waiting on stalled tails.

    Finishing the small [DONE]/usage trailer permits connection reuse. The
    completion is already retained, so a slow or broken tail must not spend
    the real-work timeout or trigger another model request.
    """
    if deadline is not None and (
        deadline - time.monotonic() <= _COMPLETION_TAIL_DRAIN_SECONDS + 0.01
    ):
        return

    async def drain() -> None:
        remaining = _COMPLETION_TAIL_DRAIN_BYTES
        while remaining > 0:
            chunk = await resp.content.read(remaining)
            if not chunk:
                return
            remaining -= len(chunk)

    try:
        await asyncio.wait_for(drain(), timeout=_COMPLETION_TAIL_DRAIN_SECONDS)
    except (TimeoutError, aiohttp.ClientError):
        # Closing the response discards this connection when EOF did not arrive.
        pass


async def _read_error_body(resp: aiohttp.ClientResponse) -> str:
    """Read a bounded, best-effort error-body preview from *resp*.

    The user-facing error format only includes at most 500 characters, but
    ``ClientResponse.text()`` first reads the entire payload.  A reverse
    proxy can attach an unexpectedly large diagnostic page to a 4xx/5xx
    response, so cap the raw read before decoding it.  A malformed declared
    charset should not hide the useful HTTP status either.
    """
    body = await resp.content.read(_MAX_HTTP_ERROR_BODY_BYTES)
    encoding = getattr(resp, "charset", None) or "utf-8"
    try:
        return body.decode(encoding, errors="replace")
    except (LookupError, TypeError):
        return body.decode("utf-8", errors="replace")


async def _iter_sse_lines(
    content: aiohttp.StreamReader,
) -> AsyncIterator[str]:
    """Yield decoded SSE lines while bounding each retained line.

    Raw chunk reads avoid aiohttp's comparatively small ``readline()`` cap,
    which some providers exceed with one complete tool-call argument. A
    malformed stream that exceeds the cap aborts this attempt: discarding an
    argument delta and accepting a later [DONE] could execute a partial call.
    """
    def line_too_long() -> None:
        raise _SSEEventTooLargeError(
            f"SSE line exceeded {_MAX_SSE_LINE_BYTES} byte limit",
        )

    async for line in iter_bounded_lines(
        content.iter_any(),
        max_line_bytes=_MAX_SSE_LINE_BYTES,
        source="SSE",
        log=logger,
        on_line_too_long=line_too_long,
    ):
        yield line


async def _iter_sse_events(
    content: aiohttp.StreamReader,
) -> AsyncIterator[str]:
    """Yield complete SSE ``data`` events, including multiline payloads.

    Server-Sent Events join multiple ``data:`` fields in one blank-line
    delimited event with newline characters.  Parsing each physical line
    separately loses valid pretty-printed JSON, so frame events here before
    the completion parser decodes them.
    """
    data_lines: list[str] = []
    event_bytes = 0
    async for line in _iter_sse_lines(content):
        if not line.strip():
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
                event_bytes = 0
            continue
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):]
        # SSE permits one space after the colon.
        if data.startswith(" "):
            data = data[1:]
        # Newline counts too.
        event_bytes += len(data.encode("utf-8")) + (1 if data_lines else 0)
        if event_bytes > _MAX_SSE_EVENT_BYTES:
            raise _SSEEventTooLargeError(
                "SSE event exceeded "
                f"{_MAX_SSE_EVENT_BYTES} byte limit",
            )
        data_lines.append(data)
    if data_lines:
        # Tolerate close right after a complete event.
        yield "\n".join(data_lines)


def _merge_tool_call(
    buffers: dict[int, dict[str, Any]], delta: object,
) -> int:
    """Fold a streaming tool-call delta into its buffer and return byte delta.

    Argument fragments remain as a list until the stream completes, avoiding
    quadratic copying from repeatedly appending to one growing string. The
    return value tracks a replacement or append in the caller's aggregate
    completion budget.
    """
    if not isinstance(delta, dict):
        return 0

    idx = delta.get("index", 0)
    # Reject non-int/negative indices.
    if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
        return 0

    function_delta = delta.get("function")
    # A non-object function field is an invalid delta.
    if function_delta is not None and not isinstance(function_delta, dict):
        return 0

    # Pre-seed an id; the real id overrides.
    buf = buffers.get(idx)
    if buf is None:
        if len(buffers) >= _MAX_TOOL_CALLS_PER_COMPLETION:
            raise _CompletionTooLargeError(
                "completion exceeded "
                f"{_MAX_TOOL_CALLS_PER_COMPLETION} tool calls",
            )
        buf = {
            "id": f"call_{idx}_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": "", "arguments": ""},
            "_argument_parts": [],
            "_argument_bytes": 0,
        }
        buffers[idx] = buf
    call_id = delta.get("id")
    if isinstance(call_id, str) and call_id:
        if len(call_id) > _MAX_TOOL_CALL_ID_CHARS:
            raise _CompletionTooLargeError(
                "tool-call id exceeded "
                f"{_MAX_TOOL_CALL_ID_CHARS} character limit",
            )
        buf["id"] = call_id
    if function_delta is None:
        return 0

    name = function_delta.get("name")
    if isinstance(name, str) and name:
        if len(name) > _MAX_TOOL_NAME_CHARS:
            raise _CompletionTooLargeError(
                "tool-call name exceeded "
                f"{_MAX_TOOL_NAME_CHARS} character limit",
            )
        buf["function"]["name"] = name
    raw_arguments = function_delta.get("arguments")
    argument_parts = buf.get("_argument_parts")
    if not isinstance(argument_parts, list):
        argument_parts = []
        buf["_argument_parts"] = argument_parts
    previous_bytes = buf.get("_argument_bytes", 0)
    if not isinstance(previous_bytes, int):
        previous_bytes = 0
    if isinstance(raw_arguments, str):
        # Arguments stream as fragments.
        fragment_bytes = len(raw_arguments.encode("utf-8", errors="replace"))
        next_bytes = previous_bytes + fragment_bytes
        if next_bytes > _MAX_TOOL_ARGUMENT_BYTES:
            raise _CompletionTooLargeError(
                "tool-call arguments exceeded "
                f"{_MAX_TOOL_ARGUMENT_BYTES} byte limit",
            )
        argument_parts.append(raw_arguments)
        buf["_argument_bytes"] = next_bytes
        return fragment_bytes
    if isinstance(raw_arguments, dict):
        # Some servers send the whole arguments object in one delta.
        try:
            serialized = json.dumps(raw_arguments)
        except (TypeError, ValueError):
            # Keeps this boundary safe for direct custom-provider calls.
            logger.debug("Ignoring non-serializable tool-call arguments")
            return 0
        serialized_bytes = len(serialized.encode("utf-8", errors="replace"))
        if serialized_bytes > _MAX_TOOL_ARGUMENT_BYTES:
            raise _CompletionTooLargeError(
                "tool-call arguments exceeded "
                f"{_MAX_TOOL_ARGUMENT_BYTES} byte limit",
            )
        buf["_argument_parts"] = [serialized]
        buf["_argument_bytes"] = serialized_bytes
        return serialized_bytes - previous_bytes
    return 0


def _vision_parts_for_prompt(
    prompt: str, workspace_path: str,
) -> list[dict[str, Any]] | None:
    """Build multimodal user content for *prompt*, or None to stay text-only.

    Delegates marker scanning + workspace checks to the shared
    ``backends_agent.base`` helpers so every vision backend resolves the
    same image set. Returns None when no in-workspace image is referenced
    (normal text turn) so the text-only message shape is preserved.
    """
    images = vision_image_url_parts(
        attachment_image_paths(prompt, workspace_path),
    )
    if not images:
        return None
    return [{"type": "text", "text": prompt}, *images]


# Tools + system prompt


def _tools_for_approval(
    approval: str, compaction: bool,
) -> list[dict] | None:
    """Tool schema exposed to the model for this turn, by permission.

    - deny / compaction: no tools (chat only).
    - confirm: read-only tools only. A chat bot can't prompt per tool call,
      so confirm becomes a look-but-don't-touch surface; ``execute_tool``
      also blocks state-changing tools as a backstop. Use /style
      collaborative for ask-before-acting on state changes.
    - auto: workspace-bounded tools; direct host-access tools require full.
    - full: the full tool set.
    """
    # Malformed approval never widens tools.
    if compaction or approval not in {"auto", "full", "confirm"}:
        return None
    if approval == "confirm":
        return tools.READ_ONLY_TOOL_SCHEMA or None
    if approval == "auto":
        return tools.AUTO_TOOL_SCHEMA or None
    return tools.TOOL_SCHEMA


def _system_prompt(
    workspace_path: str, tool_names: tuple[str, ...], approval: str = "auto",
) -> str:
    """Build the per-turn system message.

    Embeds the current workspace path so the model is aware of where it is
    operating; lists the exact tools available this turn (which may be a
    read-only subset under the confirm permission) so the model doesn't try
    to call tools that aren't exposed. States the effective permission too,
    so full-access turns do not inherit a workspace sandbox assumption.
    Rebuilt on every turn, so permission changes and workspace switches are
    automatically reflected.
    """
    effective_approval = (
        approval if approval in {"auto", "full", "confirm"} else "deny"
    )
    parts = [
        "Cozter assistant.",
        f"Workspace: {workspace_path}",
        f"Effective permission: {effective_approval}.",
        WHOLE_SCOPE_RULE,
        DOC_FOLLOWING_RULE,
        PROGRESS_RULE,
        VISION_RULE,
    ]
    if effective_approval == "full":
        parts.append(
            "Built-in filesystem tools may access paths outside the workspace,"
            " including absolute paths and relative paths with '..', subject"
            " to the host account's operating system permissions. Relative"
            " paths are based on the workspace. Full permission removes"
            " Cozter's workspace boundary for these tools; use them for"
            " authorized work instead of assuming this turn is sandboxed."
            " Other tool-specific restrictions still apply."
        )
    elif effective_approval in {"auto", "confirm"}:
        parts.append(
            "Filesystem tools are confined to the workspace, including"
            " resolved symlink targets. Relative paths are based on the"
            " workspace; absolute paths must remain inside it. The direct"
            " host shell is unavailable."
        )
        if effective_approval == "confirm":
            parts.append("Only read-only tools are available; state changes are blocked.")
    else:
        parts.append("Tool execution is disabled by the effective permission.")
    if tool_names:
        parts.append(
            f"Tools: {', '.join(tool_names)}."
            " You may call multiple independent tools in one turn;"
            " they run concurrently (results return in order)."
            " One pass; don't repeat calls."
        )
        if "bash" in tool_names:
            parts.append(shell_context())
    else:
        parts.append("No tools this turn.")
    return "\n".join(parts)
