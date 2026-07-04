"""Shared in-process agent loop for OpenAI-compatible chat backends.

llama-server, Z.ai (GLM), and any other server that speaks OpenAI-style
``/chat/completions`` with streaming + tool calls share one tool-calling
loop: send messages + tools, execute the ``tool_calls`` the model returns,
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
import json
import logging
import random
import uuid
from typing import Any

import aiohttp

from .. import agent_tools as tools
from ._http_proc import HttpAgentProcess, http_error_translator
from .base import (
    AgentResult, Backend, ChatEvent, append_text_result, set_error_result,
)

logger = logging.getLogger(__name__)


class OpenAIChatBackend(Backend):
    """Backend that drives an OpenAI-compatible chat-completions endpoint.

    Subclasses override the hooks below; the loop, event parsing, and
    streaming client are inherited.
    """

    # OpenAI-shape tools list is consumed directly, so plugins become typed
    # tool entries in TOOL_SCHEMA the same way the built-ins do.
    supports_typed_plugins = True
    # OpenAI Chat Completions supports the standard 4-level effort words.
    effort_levels: tuple[str, ...] = ("minimal", "low", "medium", "high")

    # ---- hooks a concrete backend overrides ----------------------------

    def _chat_endpoint(self) -> str:
        """Full URL of the chat/completions endpoint."""
        raise NotImplementedError

    def _auth_headers(self) -> dict[str, str]:
        """Extra HTTP headers (e.g. Authorization). Empty for local servers."""
        return {}

    def _request_model(self, model: str | None) -> str | None:
        """Model id to send, or None to omit the field entirely."""
        return model or None

    def _max_agent_turns(self) -> int:
        return 40

    def _tool_repeat_limit(self) -> int:
        return 3

    def _socket_timeout(self) -> int:
        return 300

    def _max_retries(self) -> int:
        return 2

    # ---- launch ---------------------------------------------------------

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
        max_retries = self._max_retries()
        # Per-AI-turn safety caps. Read fresh each turn so a config edit
        # takes effect on the next message without a restart.
        max_agent_turns = self._max_agent_turns()
        tool_repeat_limit = self._tool_repeat_limit()
        request_model = self._request_model(model)

        # Approval -> tool exposure (see _tools_for_approval):
        #   * deny / compaction -> no tools (chat only)
        #   * confirm -> read-only tools only (look-but-don't-touch)
        #   * auto / full -> all tools
        tools_schema = _tools_for_approval(approval, compaction)
        enabled_tool_names = tuple(
            entry["function"]["name"] for entry in tools_schema
        ) if tools_schema else ()

        # Translate the 0-100 effort into the native vocabulary; an empty
        # result means "do not send the reasoning_effort field".
        reasoning_effort = self.convert_effort(effort) or ""

        # The endpoint is stateless, so the model has no idea what cwd it is
        # operating against unless we tell it. CLI backends learn the
        # workspace via their --add-dir / -C / cwd flag; here it goes in the
        # prompt, rebuilt every turn so a workspace switch propagates.
        messages: list[dict] = [
            {
                "role": "system",
                "content": _system_prompt(workspace_path, enabled_tool_names),
            },
            {"role": "user", "content": prompt},
        ]
        tool_repeat_counts: dict[str, int] = {}

        for _ in range(max_agent_turns):
            payload = _completion_payload(
                messages, request_model, reasoning_effort, tools_schema,
            )
            if tools_schema is not None:
                payload["tool_choice"] = "auto"

            assistant_text, tool_calls = await _stream_completion(
                endpoint, payload, headers, sock_read, max_retries, self.name,
            )

            # OpenAI spec: when ``tool_calls`` is present, ``content`` should
            # be null (not ""). Some strict servers reject empty-string
            # content alongside tool_calls.
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_text if assistant_text else None,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            # Surface this turn's commentary even if more tool calls follow;
            # otherwise the user would only see whatever the model says in
            # the FINAL turn, losing "Let me check that file" narration.
            if assistant_text:
                proc.emit({"type": "assistant_text", "text": assistant_text})

            if not tool_calls:
                return

            # Execute each requested tool and append the result. ``approval``
            # is passed through to execute_tool, which enforces the
            # confirm-mode read-only gate as a backstop even if the model
            # asks for a state-changing tool it wasn't offered.
            for call in tool_calls:
                name, args = tools.parse_openai_call(call)
                sig = tools.tool_signature(name, args)
                tool_repeat_counts[sig] = tool_repeat_counts.get(sig, 0) + 1

                if tool_repeat_counts[sig] > tool_repeat_limit:
                    result = (
                        f"Skipped repeated tool call: {name}. "
                        f"The same tool call was requested more than "
                        f"{tool_repeat_limit} times. Stop repeating this "
                        "call and produce the final answer using the "
                        "information already available."
                    )
                    proc.emit({
                        "type": "tool_result", "name": name, "output": result,
                    })
                else:
                    result = await tools.execute_tool(
                        name, args, workspace_path, approval, proc.emit,
                    )

                # Include ``name`` alongside tool_call_id; strict servers
                # reject tool messages without it.
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "name": name,
                    "content": result,
                })

        # If we fall out of the loop, force one final no-tools response
        # instead of returning only an error.
        messages.append({
            "role": "user",
            "content": (
                "You have reached the tool-call limit. Do not call any more"
                " tools. Based only on the information already collected,"
                " provide the final answer now. If something is incomplete,"
                " clearly say what is missing."
            ),
        })

        payload = _completion_payload(messages, request_model, reasoning_effort)
        assistant_text, _ = await _stream_completion(
            endpoint, payload, headers, sock_read, max_retries, self.name,
        )

        if assistant_text:
            proc.emit({"type": "assistant_text", "text": assistant_text})
        else:
            proc.emit({
                "type": "error",
                "message": (
                    f"{self.name} agent exceeded {max_agent_turns} tool-call"
                    " turns and failed to produce a final answer."
                ),
            })

    # ---- event parsing --------------------------------------------------

    def parse_event(self, event: dict, result: AgentResult) -> None:
        etype = event.get("type", "")

        if etype == "assistant_text":
            text = event.get("text") or ""
            if text:
                # Last text wins for result.text (matches codex/copilot).
                append_text_result(result, text)
            return

        if etype == "tool_use":
            tool = event.get("name", "?")
            inp = event.get("input") or {}
            content = tools.summarize_tool_use(tool, inp)
            result.events.append(ChatEvent(kind="tool", content=content))
            if event.get("file_action"):
                # write/edit/delete - also surface as kind="file" so the
                # status display routes it through the file UX.
                path = inp.get("path", "?")
                action = event["file_action"]
                result.events.append(ChatEvent(
                    kind="file", content=f"📄 {action}: {path}",
                ))
            return

        if etype == "tool_result":
            # Already covered by the preceding tool_use event; suppress
            # duplicate noise.
            return

        if etype == "error":
            msg = event.get("message") or "Unknown error"
            set_error_result(result, msg)
            return

        logger.debug("%s: unhandled event %r", self.name, event)

    def extract_agent_text(self, event: dict) -> str | None:
        if event.get("type") == "assistant_text":
            text = event.get("text")
            if isinstance(text, str) and text:
                return text
        return None


# ---------------------------------------------------------------------------
# Streaming chat-completions client
# ---------------------------------------------------------------------------


def _completion_payload(
    messages: list[dict],
    request_model: str | None,
    reasoning_effort: str,
    tools_schema: list[dict] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": messages,
        "stream": True,
    }
    if request_model is not None:
        payload["model"] = request_model
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if tools_schema is not None:
        payload["tools"] = tools_schema
    return payload


class _RetryableError(RuntimeError):
    """A transient failure worth retrying (network / 429 / 5xx)."""

    def __init__(
        self, message: str, *, retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _backoff_delay(
    attempt: int, retry_after: float | None = None,
    *, base: float = 0.5, cap: float = 10.0,
) -> float:
    """Seconds to wait before retry *attempt* (1-based); honors Retry-After."""
    if retry_after is not None:
        return min(max(retry_after, 0.0), cap)
    delay = min(base * (2 ** (attempt - 1)), cap)
    return delay + random.uniform(0.0, delay * 0.25)


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header (delta-seconds form); ignore HTTP dates."""
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return None


async def _stream_completion(
    endpoint: str,
    payload: dict,
    headers: dict[str, str],
    sock_read: int,
    max_retries: int,
    label: str,
) -> tuple[str, list[dict]]:
    """POST the chat/completions endpoint (streaming); retry transient fails.

    Returns ``(text, tool_calls)``. Connection drops, read timeouts, and
    HTTP 429/5xx are retried with exponential backoff up to *max_retries*
    times - retrying a completion is safe because tool side effects only
    run *after* this returns. A bad status or malformed response is not
    retried.
    """
    async with http_error_translator(label, sock_read):
        attempt = 0
        while True:
            try:
                return await _stream_once(
                    endpoint, payload, headers, sock_read, label,
                )
            except _RetryableError as exc:
                attempt += 1
                if attempt > max_retries:
                    raise
                delay = _backoff_delay(attempt, exc.retry_after)
                logger.warning(
                    "%s transient failure (attempt %d/%d): %s;"
                    " retrying in %.1fs",
                    label, attempt, max_retries, exc, delay,
                )
                await asyncio.sleep(delay)


async def _stream_once(
    endpoint: str,
    payload: dict,
    headers: dict[str, str],
    sock_read: int,
    label: str,
) -> tuple[str, list[dict]]:
    """One streaming attempt; raise _RetryableError for transient failures.

    Parses Server-Sent Events. ``data:`` lines carry JSON deltas;
    ``data: [DONE]`` terminates the stream.
    """
    text_parts: list[str] = []
    # Tool calls arrive in pieces: we accumulate by index because the
    # OpenAI streaming protocol fragments name/arguments across deltas.
    tool_buffers: dict[int, dict[str, Any]] = {}

    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                endpoint, json=payload, headers=headers or None,
                timeout=aiohttp.ClientTimeout(total=None, sock_read=sock_read),
            ) as resp,
        ):
            if resp.status == 429 or resp.status >= 500:
                body = await resp.text()
                raise _RetryableError(
                    f"{label} returned HTTP {resp.status}: {body[:200]}",
                    retry_after=_parse_retry_after(
                        resp.headers.get("Retry-After"),
                    ),
                )
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"{label} returned HTTP {resp.status}: {body[:500]}"
                )
            async for raw in resp.content:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    logger.debug("Non-JSON SSE line: %r", data)
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    text_parts.append(content)
                for tc in delta.get("tool_calls") or []:
                    _merge_tool_call(tool_buffers, tc)
    except (
        aiohttp.ClientConnectorError,
        aiohttp.ServerDisconnectedError,
        aiohttp.ClientPayloadError,
        TimeoutError,
    ) as exc:
        raise _RetryableError(f"{label}: {exc}") from exc

    # Normalize tool_buffers into the OpenAI tool_calls list shape.
    tool_calls = [
        tool_buffers[idx] for idx in sorted(tool_buffers.keys())
        if tool_buffers[idx].get("function", {}).get("name")
    ]
    return "".join(text_parts), tool_calls


def _merge_tool_call(buffers: dict[int, dict], delta: dict) -> None:
    """Fold a single streaming tool_call delta into the index'd buffer."""
    idx = delta.get("index", 0)
    # Pre-seed with a synthetic id so we always have something to put in the
    # tool-result message's tool_call_id field; some servers reject empty
    # tool_call_ids. The server-provided id below overrides this if present.
    buf = buffers.setdefault(idx, {
        "id": f"call_{idx}_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {"name": "", "arguments": ""},
    })
    if "id" in delta and delta["id"]:
        buf["id"] = delta["id"]
    fn = delta.get("function") or {}
    if "name" in fn and fn["name"]:
        buf["function"]["name"] = fn["name"]
    args_frag = fn.get("arguments")
    if isinstance(args_frag, str):
        # Arguments stream in as JSON-string fragments; concatenate.
        buf["function"]["arguments"] += args_frag
    elif isinstance(args_frag, dict):
        # Some servers (GLM / Z.ai, some local runtimes) send the whole
        # arguments object in one delta instead of string fragments.
        buf["function"]["arguments"] = json.dumps(args_frag)


# ---------------------------------------------------------------------------
# Tool exposure + system prompt
# ---------------------------------------------------------------------------


def _tools_for_approval(
    approval: str, compaction: bool,
) -> list[dict] | None:
    """Tool schema exposed to the model for this turn, by permission.

    - deny / compaction: no tools (chat only).
    - confirm: read-only tools only. A chat bot can't prompt per tool call,
      so confirm becomes a look-but-don't-touch surface; ``execute_tool``
      also blocks state-changing tools as a backstop. Use /style
      collaborative for ask-before-acting on state changes.
    - auto / full: the full tool set.
    """
    if compaction or approval == "deny":
        return None
    if approval == "confirm":
        return tools.READ_ONLY_TOOL_SCHEMA or None
    return tools.TOOL_SCHEMA


def _system_prompt(
    workspace_path: str, tool_names: tuple[str, ...],
) -> str:
    """Build the per-turn system message.

    Embeds the current workspace path so the model is aware of where it is
    operating; lists the exact tools available this turn (which may be a
    read-only subset under the confirm permission) so the model doesn't try
    to call tools that aren't exposed. Rebuilt on every turn, so any
    workspace switch is automatically reflected.
    """
    parts = [
        "You are a coding assistant running inside Cozter.",
        f"Current workspace: {workspace_path}",
    ]
    if tool_names:
        parts.append(
            f"Available tools: {', '.join(tool_names)}."
            " File, shell, and discovery tools run inside this"
            " workspace. Paths may be relative to the workspace root"
            " or absolute inside it. Use list_dir/glob/grep to explore"
            " the workspace before reading or editing files; prefer"
            " specific patterns like '**/*.py' over '**/*' to avoid"
            " noise from .git, node_modules, etc. For large files,"
            " pass *offset* and *limit* to read_file. Web tools use"
            " this client's internet connection - web_search to find"
            " pages, then web_fetch to read specific URLs. Do not"
            " repeat the same tool call. Once you have enough"
            " information, stop using tools and provide the final"
            " answer."
        )
    else:
        parts.append(
            "No tools are available this turn - respond in plain text."
        )
    return "\n".join(parts)
