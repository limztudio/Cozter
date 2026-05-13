"""llama-server backend: in-process agent loop against an OpenAI-compatible HTTP API.

Unlike codex/copilot/claude_code which delegate to a CLI subprocess,
llama-server is just a chat-completion HTTP endpoint. To behave like an
agent (file edits, shell, etc.) we have to run the tool-calling loop
ourselves: send messages + tools, execute any tool_calls the model
returns, append the results, and re-call until the model stops calling
tools. The whole loop runs inside a fake asyncio.subprocess.Process so
the existing orchestrator code in ``agent.py`` consumes events the same
way it does for the CLI-backed agents.

Config: ``config.json``'s ``llama_server_url`` (default
``http://127.0.0.1:8080``). The server must speak OpenAI-compatible
``/v1/chat/completions`` with streaming and tool calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import urllib.request
import uuid
from typing import Any

import aiohttp

from .. import config as cfg
from .base import AgentResult, Backend, ChatEvent

logger = logging.getLogger(__name__)

# Per-AI-turn safety cap on tool-call iterations to keep a runaway model
# from hammering the server forever.
_MAX_AGENT_TURNS = 30

# Cap each tool result we feed back to the model: huge bash outputs
# blow up the prompt and rarely help the model.
_TOOL_RESULT_MAX = 4_000

# Bash tool default timeout (model can override via the ``timeout``
# argument up to this hard cap).
_BASH_DEFAULT_TIMEOUT = 30
_BASH_MAX_TIMEOUT = 120


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

    async def launch(
        self,
        workspace_path: str,
        prompt: str,
        model: str | None,
        approval: str,
        *,
        compaction: bool = False,
    ) -> _LlamaSession:  # type: ignore[override]
        proc = _LlamaSession()
        proc.start(self._run_agent(
            proc, workspace_path, prompt, model, approval, compaction,
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
    ) -> None:
        url = cfg.get_llama_server_url().rstrip("/")
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
        tools_schema = _TOOL_SCHEMA if tools_enabled else None

        for turn in range(_MAX_AGENT_TURNS):
            payload: dict[str, Any] = {
                "messages": messages,
                "stream": True,
            }
            if request_model is not None:
                payload["model"] = request_model
            if tools_schema is not None:
                payload["tools"] = tools_schema
                payload["tool_choice"] = "auto"

            assistant_text, tool_calls = await _stream_completion(
                url, payload, proc,
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
                result = await _execute_tool(
                    call, workspace_path, approval, proc,
                )
                # Include ``name`` alongside tool_call_id; strict
                # llama-server builds reject tool messages without it.
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "name": call.get("function", {}).get("name", ""),
                    "content": result,
                })

        # If we fall out of the loop, the model couldn't terminate within
        # the iteration cap - surface that explicitly so the user knows.
        proc.emit({
            "type": "error",
            "message": (
                f"llama agent exceeded {_MAX_AGENT_TURNS} tool-call"
                " turns without producing a final answer."
            ),
        })

    # ---- event parsing --------------------------------------------------

    def parse_event(self, event: dict, result: AgentResult) -> None:
        etype = event.get("type", "")

        if etype == "text_delta":
            # Streaming text token from the assistant. The codex/copilot
            # backends only surface final text; doing the same here keeps
            # the "Thinking..." rendering uncluttered.
            return

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
            content = _summarize_tool_use(tool, inp)
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
    base_url: str, payload: dict, proc: _LlamaSession,
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
                        f"llama-server returned HTTP {resp.status}: "
                        f"{body[:500]}"
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
                        proc.emit({"type": "text_delta", "text": content})
                    for tc in delta.get("tool_calls") or []:
                        _merge_tool_call(tool_buffers, tc)
    except aiohttp.ClientError as exc:
        raise RuntimeError(
            f"llama-server connection failed: {exc}"
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
# Tools
# ---------------------------------------------------------------------------


_TOOL_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file from the workspace. Returns the"
                " full file contents (or an error message)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path relative to the workspace root, or an"
                            " absolute path inside the workspace."
                        ),
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write *content* to *path*, creating parent dirs as"
                " needed. Overwrites any existing file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "In-place string replacement: replace *old_string* with"
                " *new_string* in *path*. Fails if *old_string* is not"
                " found exactly once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command in the workspace. Use sparingly;"
                " prefer read_file/write_file/edit_file for file ops."
            ),
            "parameters": {
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
            },
        },
    },
]


async def _execute_tool(
    call: dict, workspace_path: str, approval: str, proc: _LlamaSession,
) -> str:
    """Run a single tool call; return a string the model can read back."""
    fn = call.get("function") or {}
    name = fn.get("name", "")
    raw_args = fn.get("arguments") or "{}"
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except json.JSONDecodeError:
        args = {}

    # Surface the tool call to the status display.
    proc.emit({
        "type": "tool_use",
        "name": name,
        "input": args,
        "file_action": _file_action(name),
    })

    if approval == "confirm":
        # We can't interactively confirm in non-interactive backends, but
        # the user asked to be told - logging is the best we can do.
        logger.info("llama tool call (confirm mode): %s %r", name, args)

    try:
        if name == "read_file":
            result = _tool_read_file(workspace_path, args)
        elif name == "write_file":
            result = _tool_write_file(workspace_path, args)
        elif name == "edit_file":
            result = _tool_edit_file(workspace_path, args)
        elif name == "bash":
            result = await _tool_bash(workspace_path, args)
        else:
            result = f"Unknown tool: {name}"
    except Exception as exc:
        result = f"Tool {name} failed: {exc}"

    if len(result) > _TOOL_RESULT_MAX:
        result = (
            result[:_TOOL_RESULT_MAX]
            + f"\n... [truncated, {len(result)} chars total]"
        )

    proc.emit({"type": "tool_result", "name": name, "output": result})
    return result


def _file_action(tool: str) -> str | None:
    return {"write_file": "write", "edit_file": "edit"}.get(tool)


def _resolve_inside_workspace(workspace: str, path: str) -> str:
    """Return absolute path; raise if it escapes the workspace."""
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    abs_ws = os.path.realpath(workspace)
    candidate = (
        path if os.path.isabs(path) else os.path.join(workspace, path)
    )
    abs_path = os.path.realpath(candidate)
    if not (abs_path == abs_ws or abs_path.startswith(abs_ws + os.sep)):
        raise ValueError(f"path escapes workspace: {path}")
    return abs_path


def _tool_read_file(workspace: str, args: dict) -> str:
    target = _resolve_inside_workspace(workspace, args.get("path", ""))
    if not os.path.isfile(target):
        return f"File not found: {args.get('path')}"
    with open(target, encoding="utf-8", errors="replace") as f:
        return f.read()


def _tool_write_file(workspace: str, args: dict) -> str:
    target = _resolve_inside_workspace(workspace, args.get("path", ""))
    content = args.get("content")
    if not isinstance(content, str):
        return "Error: 'content' must be a string"
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to {args.get('path')}"


def _tool_edit_file(workspace: str, args: dict) -> str:
    target = _resolve_inside_workspace(workspace, args.get("path", ""))
    old = args.get("old_string")
    new = args.get("new_string")
    if not isinstance(old, str) or not isinstance(new, str):
        return "Error: old_string and new_string must be strings"
    if not os.path.isfile(target):
        return f"File not found: {args.get('path')}"
    with open(target, encoding="utf-8", errors="replace") as f:
        original = f.read()
    count = original.count(old)
    if count == 0:
        return f"old_string not found in {args.get('path')}"
    if count > 1:
        return (
            f"old_string appears {count} times in {args.get('path')};"
            " include more context to make it unique."
        )
    updated = original.replace(old, new, 1)
    with open(target, "w", encoding="utf-8") as f:
        f.write(updated)
    return f"Replaced 1 occurrence in {args.get('path')}"


async def _tool_bash(workspace: str, args: dict) -> str:
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        return "Error: 'command' must be a non-empty string"
    timeout = args.get("timeout") or _BASH_DEFAULT_TIMEOUT
    try:
        timeout = int(timeout)
    except (TypeError, ValueError):
        timeout = _BASH_DEFAULT_TIMEOUT
    timeout = max(1, min(timeout, _BASH_MAX_TIMEOUT))

    # Use the shell so the model can use pipes, redirection, etc.
    shell = _find_shell()
    if shell is None:
        return "Error: no shell available to run bash commands"

    try:
        proc = await asyncio.create_subprocess_exec(
            *shell, command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=workspace,
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
            "Tool calls (read_file, write_file, edit_file, bash) run"
            " inside this workspace. Paths may be relative to the"
            " workspace root or absolute inside it."
        )
    else:
        parts.append(
            "No tools are available this turn - respond in plain text."
        )
    return "\n".join(parts)


def _summarize_tool_use(tool: str, args: dict) -> str:
    if tool == "bash":
        cmd = args.get("command", "")
        return f"$ {cmd[:200]}" + ("..." if len(cmd) > 200 else "")
    if tool in ("read_file", "write_file", "edit_file"):
        return f"{tool}: {args.get('path', '?')}"
    return tool
