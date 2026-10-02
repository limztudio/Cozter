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

HTTP backends (llama, zai, future Mistral/Gemini/...) see builtin and
plugins identically as typed tools in :data:`TOOL_SCHEMA`. CLI
backends (codex, claude_code, copilot, grok) cannot accept external tool
injections; for them the orchestrator prepends :func:`cli_plugin_prelude`
to the prompt so the model knows to invoke plugins through its own
``bash`` tool via ``python -m Cozter.agent_tools.plugins.<name>``.

Backends consume:

  - :data:`TOOL_SCHEMA` - OpenAI-shape ``tools`` list (builtin + plugins).
  - :func:`execute_tool` - run a tool by ``name`` + parsed ``args``.
  - :func:`tool_signature` - stable JSON fingerprint for repeat detection.
  - :func:`summarize_tool_use` - one-line status-display formatter.
  - :func:`parse_openai_call` - convenience for OpenAI-shape callers.
  - :func:`cli_plugin_prelude` - prompt addendum for CLI backends.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import pkgutil
import sys
from collections.abc import Callable
from typing import Any

from .base import AgentTool
from .base import truncate_with_marker

logger = logging.getLogger(__name__)

# Bound tool results: huge outputs blow up the prompt, rarely help.
_TOOL_RESULT_MAX = 4_000


def _tool_timeout_seconds() -> float | None:
    """Return the real-work cap for one tool call (default 3600s).

    Read lazily so a config edit takes effect on the next tool call
    without a restart. Malformed/absent config falls back to 3600s.
    """
    try:
        from .. import config as _cfg

        value = _cfg.get_tool_timeout()
    except Exception:
        return 3600.0
    try:
        seconds = float(value) if value is not None else 3600.0
    except (TypeError, ValueError):
        return 3600.0
    return seconds if seconds > 0 else 3600.0


# Tool discovery: import every sibling module to trigger self-registration


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
        before = list(AgentTool.registry)
        try:
            importlib.import_module(f"{pkg_name}.{_mod_info.name}")
        except Exception:
            # Class definition may already have self-registered before the
            # failure: roll back the whole registry to the pre-import state.
            AgentTool.registry[:] = before
            logger.exception(
                "Failed to load %s.%s", pkg_name, _mod_info.name,
            )
            continue
        if mark_as_plugin:
            before_ids = {id(t) for t in before}
            for t in AgentTool.registry:
                if id(t) not in before_ids:
                    t.is_plugin = True


_load_subpackage("builtin", mark_as_plugin=False)

# Defer plugin imports under ``python -m``: preloading there makes runpy
# warn about unpredictable execution. Normal startup still loads them.
if not sys.argv or sys.argv[0] != "-m":
    _load_subpackage("plugins", mark_as_plugin=True)

# Deterministic order: explicit ``order`` then name.
_TOOLS: tuple[AgentTool, ...] = tuple(
    sorted(AgentTool.registry, key=lambda t: (t.order, t.name))
)
_BY_NAME: dict[str, AgentTool] = {t.name: t for t in _TOOLS}

# Read-only surface for "confirm" mode (no prompts per call on chat bots):
# anything unlisted (mutating builtins, bash, all plugins) is withheld.
READ_ONLY_TOOL_NAMES: frozenset[str] = frozenset({
    "read_file",
    "list_dir",
    "tree",
    "glob",
    "grep",
    "web_search",
    "web_fetch",
})


def _is_confirm_read_only(tool: AgentTool | None) -> bool:
    """Return whether *tool* may be exposed and run in confirm mode.

    Registry is keyed by name, so a colliding plugin could shadow a
    read-only builtin: withhold every plugin regardless of name.
    """
    return (
        tool is not None
        and not tool.is_plugin
        and tool.name in READ_ONLY_TOOL_NAMES
    )


def _is_auto_allowed(tool: AgentTool | None) -> bool:
    """Return whether *tool* is safe for HTTP agents in auto mode.

    New tools default to auto; mark escape-capable ones with
    ``requires_full_permission`` to opt out.
    """
    return tool is not None and not bool(
        getattr(tool, "requires_full_permission", False),
    )


def _filtered_tool_schema(
    registered_tools: tuple[AgentTool, ...],
    predicate: Callable[[AgentTool], bool],
) -> list[dict[str, Any]]:
    """Build a tool schema from the registered tools matching *predicate*."""
    return [
        {"type": "function", "function": tool.schema}
        for tool in registered_tools
        if predicate(tool)
    ]


TOOL_SCHEMA: list[dict[str, Any]] = [
    {"type": "function", "function": t.schema} for t in _TOOLS
]

AUTO_TOOL_SCHEMA: list[dict[str, Any]] = _filtered_tool_schema(
    _TOOLS, _is_auto_allowed,
)

READ_ONLY_TOOL_SCHEMA: list[dict[str, Any]] = _filtered_tool_schema(
    _TOOLS, _is_confirm_read_only,
)


# Public API


# Internal: signature alias for the per-event emit callback that
# every backend gives us so tools can stream status updates back.
_EmitFn = Callable[[dict], None]


def parse_openai_call(call: dict) -> tuple[str, dict]:
    """Pull ``(name, args)`` out of an OpenAI-shape tool_call dict.

    Per the OpenAI spec, ``function.arguments`` is a JSON string; some
    servers return an already-parsed object. Accept both.
    """
    if not isinstance(call, dict):
        return "", {}
    fn = call.get("function")
    if not isinstance(fn, dict):
        return "", {}
    name = fn.get("name")
    if not isinstance(name, str):
        name = ""
    raw = fn.get("arguments")
    # Also accept an already-parsed object (GLM/Z.ai, local runtimes).
    if isinstance(raw, dict):
        args = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            args = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            args = {}
    else:
        args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args


def _emit_tool_result(emit: _EmitFn, name: str, result: str) -> str:
    """Emit and return one model-facing tool result consistently."""
    emit({"type": "tool_result", "name": name, "output": result})
    return result


async def execute_tool(
    name: str,
    args: dict,
    workspace_path: str,
    approval: str,
    emit: _EmitFn,
) -> str:
    """Run a tool by name; emit status events; return the result string."""
    # Provider output may be malformed: normalize before emitting events.
    if not isinstance(name, str):
        name = ""
    if not isinstance(args, dict):
        args = {}
    tool = _BY_NAME.get(name)

    emit({
        "type": "tool_use",
        "name": name,
        "input": args,
        "file_action": tool.file_action if tool else None,
    })

    if approval not in {"auto", "full", "confirm"}:
        # Schemas omit tools here, but a stray provider call could still
        # arrive: stay fail-closed against workspace mutation.
        logger.info("%s mode blocked tool: %s", approval, name)
        result = (
            f"Blocked: '{name}' cannot run because permission mode "
            f"'{approval}' permits no tools."
        )
        return _emit_tool_result(emit, name, result)

    if approval == "confirm" and not _is_confirm_read_only(tool):
        # "confirm" is a read-only gate (ask-before-write lives at turn level).
        logger.info("confirm mode blocked state-changing tool: %s", name)
        result = (
            f"Blocked: '{name}' can change state, and confirm mode only "
            "permits read-only tools. Ask the user to switch permission to "
            "auto or full to allow changes, or continue using read-only "
            "tools (read_file, list_dir, glob, grep, web_search, web_fetch)."
        )
        return _emit_tool_result(emit, name, result)

    if (
        approval == "auto"
        and tool is not None
        and not _is_auto_allowed(tool)
    ):
        # Re-check at execution: a stray/hallucinated call must not turn
        # ``auto`` into a back door to the host shell.
        logger.info("auto mode blocked full-only tool: %s", name)
        result = (
            f"Blocked: '{name}' requires full permission because it can "
            "access the host outside Cozter's workspace-bounded tool "
            "surface. Switch to full only if the operator accepts that "
            "risk."
        )
        return _emit_tool_result(emit, name, result)

    if tool is None:
        result = f"Unknown tool: {name}"
    else:
        # Real-work cap (up to tool_timeout); cancel still stops instantly.
        try:
            cap = _tool_timeout_seconds()
            if cap is not None and cap > 0:
                raw_result: object = await asyncio.wait_for(
                    tool.run(workspace_path, args), timeout=cap,
                )
            else:
                raw_result = await tool.run(workspace_path, args)
            if isinstance(raw_result, str):
                result = raw_result
            else:
                result_type = type(raw_result).__name__
                logger.warning(
                    "Tool %s returned %s instead of text",
                    name,
                    result_type,
                )
                result = (
                    f"Tool {name} returned an invalid non-text result "
                    f"({result_type})."
                )
        except asyncio.CancelledError:
            raise  # cancel is the only stop.
        except Exception as exc:
            result = f"Tool {name} failed: {exc}"

    if len(result) > _TOOL_RESULT_MAX:
        result = truncate_with_marker(
            result, _TOOL_RESULT_MAX,
            "truncated — use read_file offset/limit or grep to fetch"
            " remainder (page until no truncation marker remains);"
            " say PARTIAL + remainder when work is left",
        )

    return _emit_tool_result(emit, name, result)


def tool_signature(name: str, args: dict) -> str:
    """Stable JSON fingerprint for repeat detection."""
    return json.dumps(
        {"name": name, "args": args},
        sort_keys=True,
        ensure_ascii=False,
    )


def summarize_tool_use(name: str, args: dict) -> str:
    """One-line status-display summary of a tool invocation."""
    if not isinstance(name, str):
        return ""
    if not isinstance(args, dict):
        args = {}
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

    lines = ["Plugins (bash):", ""]
    for tool in plugins:
        props = tool.parameters.get("properties", {})
        required = set(tool.parameters.get("required", []))
        args_summary = (
            ", ".join(
                f"{k}{'' if k in required else '?'}"
                for k in props
            )
            or "no args"
        )
        # Use the class's actual __module__ so the python -m line works
        # even when the plugin file's name differs from the tool's name
        # attribute (e.g. weather_lookup.py defining GetWeatherTool).
        module_path = tool.__class__.__module__
        lines.append(f"- {tool.name}: {tool.description}")
        lines.append(f"  {args_summary} | python -m {module_path} '<JSON>'")
        lines.append("")
    return "\n".join(lines).rstrip()


__all__ = [
    "AUTO_TOOL_SCHEMA",
    "READ_ONLY_TOOL_NAMES",
    "READ_ONLY_TOOL_SCHEMA",
    "TOOL_SCHEMA",
    "AgentTool",
    "cli_plugin_prelude",
    "execute_tool",
    "parse_openai_call",
    "summarize_tool_use",
    "tool_signature",
]
