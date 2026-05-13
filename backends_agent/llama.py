"""llama-server backend: in-process agent loop against an OpenAI-compatible HTTP API.

Unlike codex/copilot/claude_code which delegate to a CLI subprocess,
llama-server is just a chat-completion HTTP endpoint. To behave like an
agent (file edits, shell, etc.) we have to run the tool-calling loop
ourselves: send messages + tools, execute any tool_calls the model
returns, append the results, and re-call until the model stops calling
tools. The whole loop runs inside a fake asyncio.subprocess.Process so
the existing orchestrator code in ``agent.py`` consumes events the same
way it does for the CLI-backed agents.

Tool definitions live in the top-level ``agent_tools/`` package (one
file per tool, plus an :class:`AgentTool` base in
``agent_tools/base.py``); that package is backend-agnostic and any
chat-completion agent could drive it. This module owns the
llama-server agent loop and HTTP plumbing only.

Config: ``config.json``'s ``llama_server_url`` (default
``http://127.0.0.1:8080``). The server must speak OpenAI-compatible
``/v1/chat/completions`` with streaming and tool calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
import uuid
from typing import Any

import aiohttp

from .. import agent_tools as tools
from .. import config as cfg
from .base import AgentResult, Backend, ChatEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fake process wrapper - mimics asyncio.subprocess.Process for agent.run
# ---------------------------------------------------------------------------


class _LlamaSession:
    """Process-like adapter for the in-process llama agent loop.

    Satisfies the duck-type that ``agent.run`` expects from
    ``backend.launch``: ``stdout`` and ``stderr`` are async streams,
    ``kill`` cancels the underlying task, ``wait`` blocks until it
    settles, and ``returncode`` reports success/cancel/error.
    """

    pid: int = -1

    def __init__(self) -> None:
        self.stdout: asyncio.StreamReader = asyncio.StreamReader()
        self.stderr: asyncio.StreamReader = asyncio.StreamReader()
        # No separate stderr channel from the HTTP path; close it now
        # so any reader sees EOF immediately.
        self.stderr.feed_eof()
        self.returncode: int | None = None
        self._task: asyncio.Task | None = None

    def emit(self, event: dict) -> None:
        """Push an event line into stdout for the orchestrator to read."""
        line = (json.dumps(event) + "\n").encode("utf-8")
        self.stdout.feed_data(line)

    def kill(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def wait(self) -> int:
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        return self.returncode if self.returncode is not None else 0

    def start(self, coro) -> None:
        async def _driver() -> None:
            try:
                await coro
                self.returncode = 0
            except asyncio.CancelledError:
                self.returncode = 130
                raise
            except Exception as exc:
                logger.exception("llama agent loop crashed")
                self.emit({"type": "error", "message": str(exc)})
                self.returncode = 1
            finally:
                self.stdout.feed_eof()

        self._task = asyncio.create_task(_driver())


# ---------------------------------------------------------------------------
# LlamaBackend
# ---------------------------------------------------------------------------


class LlamaBackend(Backend):
    name = "llama"
    executable = "llama-server"  # only used in "not found" error text

    # llama-server consumes the OpenAI-shape tools list directly, so
    # plugins discovered in agent_tools/plugins/ become typed tool
    # entries in TOOL_SCHEMA the same way the built-ins do. CLI backends
    # leave this False and instead see plugins via cli_plugin_prelude().
    supports_typed_plugins = True

    # The model list is populated dynamically from /v1/models on first
    # access. Stored as a tuple per the Backend contract.
    default_model = "auto"
    default_summary_model = "auto"

    def __init__(self) -> None:
        self._cached_models: tuple[str, ...] | None = None

    # ---- model discovery ------------------------------------------------

    @property
    def available_models(self) -> tuple[str, ...]:
        if self._cached_models is None:
            self._cached_models = self._fetch_models()
        return self._cached_models

    def refresh_models(self) -> tuple[str, ...]:
        """Re-query the server (e.g. after a model swap)."""
        self._cached_models = self._fetch_models()
        return self._cached_models

    def _fetch_models(self) -> tuple[str, ...]:
        url = cfg.get_llama_server_url().rstrip("/") + "/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            ids = tuple(
                m["id"] for m in payload.get("data", [])
                if isinstance(m, dict) and isinstance(m.get("id"), str)
            )
            return ids or ("auto",)
        except Exception as exc:
            logger.debug(
                "Could not query %s for models (%s); using 'auto'",
                url, exc,
            )
            return ("auto",)

    # ---- launch ---------------------------------------------------------

    def convert_effort(self, percent: int) -> str | None:
        # llama-server / OpenAI Chat Completions support the standard
        # 4-level effort vocabulary. Split 1-100 into roughly equal bands.
        if percent <= 0:
            return None
        if percent < 25:
            return "minimal"
        if percent < 50:
            return "low"
        if percent < 75:
            return "medium"
        return "high"

    async def launch(
        self,
        workspace_path: str,
        prompt: str,
        model: str | None,
        approval: str,
        *,
        compaction: bool = False,
        effort: int = 0,
    ) -> _LlamaSession:  # type: ignore[override]
        proc = _LlamaSession()
        proc.start(self._run_agent(
            proc, workspace_path, prompt, model, approval, compaction,
            effort,
        ))
        return proc

    async def _run_agent(
        self,
        proc: _LlamaSession,
        workspace_path: str,
        prompt: str,
        model: str | None,
        approval: str,
        compaction: bool,
        effort: int,
    ) -> None:
        url = cfg.get_llama_server_url().rstrip("/")
        # Per-AI-turn safety caps. Read fresh each turn so a config edit
        # takes effect on the next message without a restart.
        max_agent_turns = cfg.get_llama_max_agent_turns()
        tool_repeat_limit = cfg.get_llama_tool_repeat_limit()
        # ``model="auto"`` (our sentinel default) means "let the server
        # decide" - drop the field from the request and llama-server uses
        # whichever model it has loaded.
        request_model: str | None = (
            None if not model or model == "auto" else model
        )
        # Approval -> tool exposure:
        #   * deny -> no tools at all (chat-only)
        #   * full/auto/confirm -> tools enabled. Compaction also gets
        #     tools off since the summarizer doesn't need them.
        tools_enabled = approval != "deny" and not compaction

        # Translate the 0-100 effort into llama's native vocabulary;
        # an empty result means "do not send the reasoning_effort field".
        reasoning_effort = self.convert_effort(effort) or ""

        # Llama-server is stateless, so the model has no idea what cwd it
        # is operating against unless we tell it. Codex/copilot/claude_code
        # learn the workspace via their CLI's --add-dir / -C / cwd flag;
        # llama needs it in the prompt. Refreshed every turn so /new,
        # /open, and a bot restart automatically propagate.
        messages: list[dict] = [
            {
                "role": "system",
                "content": _system_prompt(workspace_path, tools_enabled),
            },
            {"role": "user", "content": prompt},
        ]
        tools_schema = tools.TOOL_SCHEMA if tools_enabled else None
        tool_repeat_counts: dict[str, int] = {}

        for _ in range(max_agent_turns):
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
                payload["tool_choice"] = "auto"

            assistant_text, tool_calls = await _stream_completion(
                url, payload,
            )

            # OpenAI spec: when ``tool_calls`` is present, ``content``
            # should be null (not ""). Some strict servers (notably
            # llama-server in some builds) reject empty-string content
            # alongside tool_calls.
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_text if assistant_text else None,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            # Surface this turn's commentary even if more tool calls
            # follow; otherwise the user would only see whatever the
            # model says in the FINAL turn, losing explanations like
            # "Let me check that file" that come before tool calls.
            if assistant_text:
                proc.emit({
                    "type": "assistant_text",
                    "text": assistant_text,
                })

            if not tool_calls:
                return

            # Execute each requested tool and append the result. ``approval``
            # is honored when tools_enabled is True, so by the time we get
            # here every tool call is permitted.
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
                        "type": "tool_result",
                        "name": name,
                        "output": result,
                    })
                else:
                    result = await tools.execute_tool(
                        name, args, workspace_path, approval, proc.emit,
                    )

                # Include ``name`` alongside tool_call_id; strict
                # llama-server builds reject tool messages without it.
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

        payload: dict[str, Any] = {
            "messages": messages,
            "stream": True,
        }
        if request_model is not None:
            payload["model"] = request_model
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        assistant_text, _ = await _stream_completion(url, payload)

        if assistant_text:
            proc.emit({
                "type": "assistant_text",
                "text": assistant_text,
            })
        else:
            proc.emit({
                "type": "error",
                "message": (
                    f"llama agent exceeded {max_agent_turns} tool-call"
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
                result.text = text
                result.events.append(ChatEvent(kind="text", content=text))
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
                    kind="file",
                    content=f"📄 {action}: {path}",
                ))
            return

        if etype == "tool_result":
            # Already covered by the preceding tool_use event in the
            # status display; suppress duplicate noise.
            return

        if etype == "error":
            msg = event.get("message") or "Unknown error"
            result.text = f"Error: {msg}"
            result.events.append(ChatEvent(kind="text", content=result.text))
            return

        logger.debug("Llama: unhandled event %r", event)

    def extract_agent_text(self, event: dict) -> str | None:
        if event.get("type") == "assistant_text":
            text = event.get("text")
            if isinstance(text, str) and text:
                return text
        return None


# ---------------------------------------------------------------------------
# Streaming chat-completions client
# ---------------------------------------------------------------------------


async def _stream_completion(
    base_url: str, payload: dict,
) -> tuple[str, list[dict]]:
    """POST /v1/chat/completions with stream=True; return (text, tool_calls).

    Parses Server-Sent Events. ``data:`` lines carry JSON deltas;
    ``data: [DONE]`` terminates the stream.
    """
    endpoint = base_url + "/v1/chat/completions"
    text_parts: list[str] = []
    # Tool calls arrive in pieces: we accumulate by index because the
    # OpenAI streaming protocol fragments name/arguments across deltas.
    tool_buffers: dict[int, dict[str, Any]] = {}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint, json=payload,
                timeout=aiohttp.ClientTimeout(total=None, sock_read=300),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"llama-server at {base_url} returned HTTP"
                        f" {resp.status}: {body[:500]}"
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
    except aiohttp.ClientConnectorError as exc:
        # Couldn't even establish the TCP connection - server is down,
        # the host is unreachable, or the URL is wrong.
        raise RuntimeError(
            f"llama-server at {base_url} is unreachable - is it running?"
        ) from exc
    except (
        aiohttp.ServerDisconnectedError,
        aiohttp.ClientPayloadError,
    ) as exc:
        # Connected once, then the server closed or dropped the connection
        # mid-response. The model output we got back (if any) is partial
        # and likely useless.
        raise RuntimeError(
            f"llama-server at {base_url} dropped the connection"
            " mid-response"
        ) from exc
    except TimeoutError as exc:
        # ``aiohttp.ClientTimeout(sock_read=300)`` fires this if a
        # 5-minute gap passes between successive reads of the streamed
        # body - i.e. the server got stuck or hung up silently.
        raise RuntimeError(
            f"llama-server at {base_url} did not respond in time"
        ) from exc
    except aiohttp.ClientError as exc:
        # Catch-all for the rest of aiohttp's client-side exception
        # hierarchy (TLS handshake failures, bad URL, etc.).
        raise RuntimeError(
            f"llama-server at {base_url} request failed: {exc}"
        ) from exc

    # Normalize tool_buffers into the OpenAI tool_calls list shape.
    tool_calls = [
        tool_buffers[idx] for idx in sorted(tool_buffers.keys())
        if tool_buffers[idx].get("function", {}).get("name")
    ]
    return "".join(text_parts), tool_calls


def _merge_tool_call(buffers: dict[int, dict], delta: dict) -> None:
    """Fold a single streaming tool_call delta into the index'd buffer."""
    idx = delta.get("index", 0)
    # Pre-seed with a synthetic id so we always have something to put in
    # the tool-result message's tool_call_id field; some OpenAI-compatible
    # servers reject empty tool_call_ids. The server-provided id below
    # overrides this if present.
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
    if "arguments" in fn and fn["arguments"] is not None:
        # Arguments stream in as JSON-string fragments; concatenate.
        buf["function"]["arguments"] += fn["arguments"]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def _system_prompt(workspace_path: str, tools_enabled: bool) -> str:
    """Build the per-turn system message.

    Embeds the current workspace path so the model is aware of where it
    is operating; mentions the tool surface (or lack of it) so the
    model doesn't try to call tools that aren't exposed. Rebuilt on
    every turn, so any workspace switch is automatically reflected.
    """
    parts = [
        "You are a coding assistant running inside Cozter.",
        f"Current workspace: {workspace_path}",
    ]
    if tools_enabled:
        parts.append(
            f"Available tools: {', '.join(tools.TOOL_NAMES)}."
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
