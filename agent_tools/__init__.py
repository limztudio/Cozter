"""Reusable tool surface for chat-completion agent backends.

This package is backend-agnostic: any agent loop that does
chat-completion + function-calling (llama-server, OpenAI, Mistral,
Gemini, Claude API, LM Studio, etc.) can drive it. The package never
sees backend protocol details - callers extract ``(name, args)`` from
their native tool-call format and hand them in.

Layout (builtin vs plugins):

  - ``agent_tools/builtin/*.py`` - the baseline toolkit shipped
    with the bot. Always loaded. ``is_plugin`` stays False.
  - ``agent_tools/plugins/*.py`` - user drop-in zone. Loaded the same
    way; instances are marked ``is_plugin = True`` after registration.
    See ``plugins/README.md`` for the template.

HTTP backends (llama, future Mistral/Gemini/...) see builtin and
plugins identically as typed tools in :data:`TOOL_SCHEMA`. CLI
backends (codex, claude_code, copilot) cannot accept external tool
injections; for them the orchestrator prepends :func:`cli_plugin_prelude`
to the prompt so the model knows to invoke plugins through its own
``bash`` tool via ``python -m Cozter.agent_tools.plugins.<name>``.

Backends consume:

  - :data:`TOOL_SCHEMA` - OpenAI-shape ``tools`` list (builtin + plugins).
  - :data:`TOOL_NAMES` - ordered tuple of tool names.
  - :func:`execute_tool` - run a tool by ``name`` + parsed ``args``.
  - :func:`tool_signature` - stable JSON fingerprint for repeat detection.
  - :func:`summarize_tool_use` - one-line status-display formatter.
  - :func:`parse_openai_call` - convenience for OpenAI-shape callers.
  - :func:`cli_plugin_prelude` - prompt addendum for CLI backends.
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


def _load_subpackage(subpkg: str, *, mark_as_plugin: bool) -> None:
    """Import every module of ``agent_tools/<subpkg>/`` so tool classes
    inside auto-register via ``AgentTool.__init_subclass__``. New
    registrations are tagged with ``is_plugin`` per the flag.

    Files starting with ``_`` are skipped, so an example plugin can
    ship in-tree without being live until renamed.
    """
    pkg_name = f"{__name__}.{subpkg}"
    try:
        pkg = importlib.import_module(pkg_name)
    except ImportError as exc:
        logger.warning("Could not import %s: %s", pkg_name, exc)
        return
    for _mod_info in pkgutil.iter_modules(pkg.__path__):
        if _mod_info.name.startswith("_"):
            continue
        before = {id(t) for t in AgentTool.registry}
        try:
            importlib.import_module(f"{pkg_name}.{_mod_info.name}")
        except Exception:
            logger.exception(
                "Failed to load %s.%s", pkg_name, _mod_info.name,
            )
            continue
        if mark_as_plugin:
            for t in AgentTool.registry:
                if id(t) not in before:
                    t.is_plugin = True


_load_subpackage("builtin", mark_as_plugin=False)
_load_subpackage("plugins", mark_as_plugin=True)

# Sort registered tools deterministically: explicit ``order`` then name.
_TOOLS: tuple[AgentTool, ...] = tuple(
    sorted(AgentTool.registry, key=lambda t: (t.order, t.name))
)
_BY_NAME: dict[str, AgentTool] = {t.name: t for t in _TOOLS}

TOOL_SCHEMA: list[dict[str, Any]] = [
    {"type": "function", "function": t.schema} for t in _TOOLS
]

TOOL_NAMES: tuple[str, ...] = tuple(t.name for t in _TOOLS)

PLUGIN_NAMES: tuple[str, ...] = tuple(
    t.name for t in _TOOLS if t.is_plugin
)


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


def cli_plugin_prelude() -> str:
    """Prompt addendum enumerating plugins for CLI backends.

    Returns ``""`` if no plugins are loaded. Otherwise returns a
    paragraph describing each plugin (name, description, args, and
    a bash-mode invocation template) so CLI-backed agents that can't
    receive typed tool definitions can still call plugins through
    their built-in ``bash`` / shell tool.
    """
    plugins = [t for t in _TOOLS if t.is_plugin]
    if not plugins:
        return ""

    lines = [
        "PLUGINS (extra tools available in this workspace, invoked via"
        " the bash/shell tool):",
        "",
    ]
    for tool in plugins:
        props = tool.parameters.get("properties", {})
        required = set(tool.parameters.get("required", []))
        args_summary = (
            ", ".join(
                f"{k}{'' if k in required else '?'}: {v.get('type', '?')}"
                for k, v in props.items()
            )
            or "no args"
        )
        # Use the class's actual __module__ so the python -m line works
        # even when the plugin file's name differs from the tool's name
        # attribute (e.g. weather_lookup.py defining GetWeatherTool).
        module_path = tool.__class__.__module__
        lines.append(f"- {tool.name}: {tool.description}")
        lines.append(f"  Args: {{{args_summary}}}")
        lines.append(f"  Invoke: python -m {module_path} '<JSON args>'")
        lines.append("")
    return "\n".join(lines).rstrip()


__all__ = [
    "AgentTool",
    "TOOL_SCHEMA",
    "TOOL_NAMES",
    "PLUGIN_NAMES",
    "execute_tool",
    "tool_signature",
    "summarize_tool_use",
    "parse_openai_call",
    "cli_plugin_prelude",
]
