"""Reusable tool surface for chat-completion agent backends.

This package is backend-agnostic: any agent loop that does
chat-completion + function-calling (llama-server, OpenAI, Mistral,
Gemini, Claude API, LM Studio, etc.) can drive it. The package never
sees backend protocol details - callers extract ``(name, args)`` from
their native tool-call format and hand them in.

Plug-and-play: every concrete subclass of :class:`AgentTool` (defined
in any sibling module) auto-registers itself via
``__init_subclass__``. The block below imports every sibling module so
those side-effect registrations fire at import time. To add a new tool,
drop a ``my_tool.py`` file in this directory that defines an
``AgentTool`` subclass - no edits to this file or to any backend are
needed.

Backends consume:

  - :data:`TOOL_SCHEMA` - OpenAI-shape ``tools`` list. Works as-is for
    OpenAI-compatible APIs. Anthropic-shape callers can translate by
    walking each entry's ``function`` dict.
  - :data:`TOOL_NAMES` - ordered tuple of tool names (for system
    prompts).
  - :func:`execute_tool` - run a tool by ``name`` + parsed ``args``;
    returns the result string.
  - :func:`tool_signature` - stable JSON fingerprint of a call for
    repeat detection.
  - :func:`summarize_tool_use` - one-line status-display formatter.
  - :func:`parse_openai_call` - convenience: pull ``(name, args)`` out
    of an OpenAI tool_call dict. Skip if your backend speaks a
    different native format.
"""

from __future__ import annotations

import importlib
import json
import logging
import pkgutil
from collections.abc import Callable
from typing import Any

from .base import AgentTool

logger = logging.getLogger(__name__)

# Cap each tool result fed back to the model: huge outputs blow up the
# prompt and rarely help the model.
_TOOL_RESULT_MAX = 4_000


# ---------------------------------------------------------------------------
# Tool discovery: import every sibling module to trigger self-registration
# ---------------------------------------------------------------------------

# Skip the base module (no tool class) and any underscore-prefixed
# module (treated as private by convention).
for _mod_info in pkgutil.iter_modules(__path__):
    if _mod_info.name == "base" or _mod_info.name.startswith("_"):
        continue
    importlib.import_module(f"{__name__}.{_mod_info.name}")

# Sort registered tools deterministically: explicit ``order`` then name.
_TOOLS: tuple[AgentTool, ...] = tuple(
    sorted(AgentTool.registry, key=lambda t: (t.order, t.name))
)
_BY_NAME: dict[str, AgentTool] = {t.name: t for t in _TOOLS}

TOOL_SCHEMA: list[dict[str, Any]] = [
    {"type": "function", "function": t.schema} for t in _TOOLS
]

TOOL_NAMES: tuple[str, ...] = tuple(t.name for t in _TOOLS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


EmitFn = Callable[[dict], None]


def parse_openai_call(call: dict) -> tuple[str, dict]:
    """Pull ``(name, args)`` out of an OpenAI-shape tool_call dict.

    Per the OpenAI Chat Completions spec, ``function.arguments`` is a
    JSON-encoded string; this helper parses it into a dict. Backends
    using other tool-call formats should construct ``(name, args)``
    themselves and call :func:`execute_tool` directly.
    """
    fn = call.get("function") or {}
    name = fn.get("name", "")
    raw = fn.get("arguments") or "{}"
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args


async def execute_tool(
    name: str,
    args: dict,
    workspace_path: str,
    approval: str,
    emit: EmitFn,
) -> str:
    """Run a tool by name; emit status events; return the result string."""
    tool = _BY_NAME.get(name)

    emit({
        "type": "tool_use",
        "name": name,
        "input": args,
        "file_action": tool.file_action if tool else None,
    })

    if approval == "confirm":
        # We can't interactively confirm in non-interactive backends, but
        # the user asked to be told - logging is the best we can do.
        logger.info("agent tool call (confirm mode): %s %r", name, args)

    if tool is None:
        result = f"Unknown tool: {name}"
    else:
        try:
            result = await tool.run(workspace_path, args)
        except Exception as exc:
            result = f"Tool {name} failed: {exc}"

    if len(result) > _TOOL_RESULT_MAX:
        result = (
            result[:_TOOL_RESULT_MAX]
            + f"\n... [truncated, {len(result)} chars total]"
        )

    emit({"type": "tool_result", "name": name, "output": result})
    return result


def tool_signature(name: str, args: dict) -> str:
    """Stable JSON fingerprint for repeat detection."""
    return json.dumps(
        {"name": name, "args": args},
        sort_keys=True,
        ensure_ascii=False,
    )


def summarize_tool_use(name: str, args: dict) -> str:
    """One-line status-display summary of a tool invocation."""
    tool = _BY_NAME.get(name)
    return tool.summarize(args) if tool else name


__all__ = [
    "AgentTool",
    "TOOL_SCHEMA",
    "TOOL_NAMES",
    "execute_tool",
    "tool_signature",
    "summarize_tool_use",
    "parse_openai_call",
]
