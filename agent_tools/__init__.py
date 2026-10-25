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
backends (codex, claude_code, copilot, grok, antigravity_cli) cannot accept external tool
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
import os
import pkgutil
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from glob import has_magic
from typing import Any

from .. import supervision
from .base import AgentTool
from .base import resolve_inside_workspace, tool_permission_scope, truncate_with_marker
from .discovery import DISCOVERY_RESULT_MAX

logger = logging.getLogger(__name__)
performance_logger = logging.getLogger("Cozter.performance")

_TOOL_RESULT_MAX = 4_000
_DIAGNOSTIC_GREP_BUDGET_SECONDS = 15.0
SKILL_TOOL_NAMES: frozenset[str] = frozenset({"list_skills", "read_skill"})
_SKILL_PLANNING: ContextVar[bool] = ContextVar("cozter_skill_planning", default=False)
DIAGNOSTIC_READ_TOOL_NAMES: frozenset[str] = frozenset({
    "read_file", "read_files", "list_dir", "glob", "grep",
})
_DIAGNOSTIC_READING: ContextVar[bool] = ContextVar("cozter_diagnostic_reading", default=False)


def diagnostic_reading_active() -> bool:
    """Whether this internal diagnostic phase can inspect only local files."""
    return _DIAGNOSTIC_READING.get()


@contextmanager
def diagnostic_reading_scope() -> Iterator[None]:
    """Restrict a diagnostic and its child tasks to local builtin reads."""
    token = _DIAGNOSTIC_READING.set(True)
    try:
        yield
    finally:
        _DIAGNOSTIC_READING.reset(token)


def skill_planning_active() -> bool:
    """Whether this trusted internal phase may read only discovered skills."""
    return _SKILL_PLANNING.get()


@contextmanager
def skill_planning_scope() -> Iterator[None]:
    """Give the Flexible planner skill discovery without ordinary tool access."""
    token = _SKILL_PLANNING.set(True)
    try:
        yield
    finally:
        _SKILL_PLANNING.reset(token)


def _load_subpackage(subpackage: str, *, mark_as_plugin: bool) -> None:
    """Import every module of ``agent_tools/<subpackage>/`` so tool classes
    inside auto-register via ``AgentTool.__init_subclass__``. New
    registrations are tagged with ``is_plugin`` per the flag.

    Files starting with ``_`` are skipped, so an example plugin can
    ship in-tree without being live until renamed.
    """
    package_name = f"{__name__}.{subpackage}"
    try:
        package = importlib.import_module(package_name)
    except ImportError as exc:
        logger.warning("Could not import %s: %s", package_name, exc)
        return
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_"):
            continue
        before = list(AgentTool.registry)
        try:
            importlib.import_module(f"{package_name}.{module_info.name}")
        except Exception:
            AgentTool.registry[:] = before
            logger.exception(
                "Failed to load %s.%s", package_name, module_info.name,
            )
            continue
        if mark_as_plugin:
            before_ids = {id(tool) for tool in before}
            for tool in AgentTool.registry:
                if id(tool) not in before_ids:
                    tool.is_plugin = True


_load_subpackage("builtin", mark_as_plugin=False)

# Defer plugin imports under ``python -m``.
if not sys.argv or sys.argv[0] != "-m":
    _load_subpackage("plugins", mark_as_plugin=True)

_TOOLS: tuple[AgentTool, ...] = tuple(
    sorted(AgentTool.registry, key=lambda tool: (tool.order, tool.name))
)
TOOLS_BY_NAME: dict[str, AgentTool] = {tool.name: tool for tool in _TOOLS}

READ_ONLY_TOOL_NAMES: frozenset[str] = frozenset({
    "read_file",
    "read_files",
    "list_skills",
    "read_skill",
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
    {"type": "function", "function": tool.schema} for tool in _TOOLS
]

AUTO_TOOL_SCHEMA: list[dict[str, Any]] = _filtered_tool_schema(
    _TOOLS, _is_auto_allowed,
)

READ_ONLY_TOOL_SCHEMA: list[dict[str, Any]] = _filtered_tool_schema(
    _TOOLS, _is_confirm_read_only,
)

SKILL_TOOL_SCHEMA: list[dict[str, Any]] = _filtered_tool_schema(
    _TOOLS, lambda tool: tool.name in SKILL_TOOL_NAMES and _is_confirm_read_only(tool),
)

DIAGNOSTIC_READ_TOOL_SCHEMA: list[dict[str, Any]] = _filtered_tool_schema(
    _TOOLS, lambda tool: tool.name in DIAGNOSTIC_READ_TOOL_NAMES and _is_confirm_read_only(tool),
)


_EmitFunc = Callable[[dict], None]


def parse_openai_call(call: dict) -> tuple[str, dict]:
    """Pull ``(name, args)`` out of an OpenAI-shape tool_call dict.

    Per the OpenAI spec, ``function.arguments`` is a JSON string; some
    servers return an already-parsed object. Accept both.
    """
    if not isinstance(call, dict):
        return "", {}
    function_payload = call.get("function")
    if not isinstance(function_payload, dict):
        return "", {}
    name = function_payload.get("name")
    if not isinstance(name, str):
        name = ""
    raw = function_payload.get("arguments")
    if isinstance(raw, dict):
        args = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            args = json.loads(raw)
        except (ValueError, TypeError, RecursionError):
            args = {}
    else:
        args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args


def _emit_tool_result(emit: _EmitFunc, name: str, result: str) -> str:
    """Emit and return one model-facing tool result consistently."""
    emit({"type": "tool_result", "name": name, "output": result})
    return result


def _diagnostic_search_refusal(name: str, args: dict, workspace_path: str) -> str | None:
    """Avoid recursive root scans without abandoning a threaded glob owner."""
    if name not in {"glob", "grep"}:
        return None
    pattern = args.get("glob") or "**/*" if name == "grep" else args.get("pattern")
    if not isinstance(pattern, str):
        return None  # The normal tool validates malformed arguments.
    drive, tail = os.path.splitdrive(pattern.replace("\\", "/"))
    parts = [part for part in tail.split("/") if part and part != "."]
    if "**" not in parts:
        return None
    prefix = []
    for part in parts:
        if has_magic(part):
            break
        prefix.append(part)
    try:
        with tool_permission_scope("confirm"):
            root = resolve_inside_workspace(workspace_path, args.get("path") or ".")
            literal_prefix = os.path.join(*prefix) if prefix else "."
            if tail.startswith("/"):
                literal_prefix = "/" + literal_prefix
            search_root = resolve_inside_workspace(
                workspace_path, os.path.join(root, drive + literal_prefix),
            )
    except (OSError, TypeError, ValueError):
        return None  # Preserve the tool's ordinary path validation feedback.
    if os.path.normcase(search_root) != os.path.normcase(os.path.realpath(workspace_path)):
        return None
    return (
        "Diagnostic search is too broad: recursive workspace-root searches are unavailable"
        " during hang diagnosis. Use list_dir at the root, then choose a narrow source/test"
        " subtree with path, or a glob with a literal directory prefix such as tests/**/*.cpp."
        " Nonrecursive root globs remain available."
    )


async def _settle_diagnostic_read(task: asyncio.Task[Any]) -> None:
    """Cancel a killable read once; preserve operator stops during cleanup."""
    if not task.done() and not task.cancelling():
        task.cancel("diagnostic search budget exceeded")
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.wait((task,))
        except asyncio.CancelledError as exc:
            if cancelled is None:
                cancelled = exc
    try:
        task.result()
    except BaseException:
        pass
    if cancelled is not None:
        raise cancelled


async def execute_tool(
    name: str,
    args: dict,
    workspace_path: str,
    approval: str,
    emit: _EmitFunc,
) -> str:
    """Run a tool by name; emit status events; return the result string."""
    if not isinstance(name, str):
        name = ""
    if not isinstance(args, dict):
        args = {}
    tool = TOOLS_BY_NAME.get(name)

    emit({
        "type": "tool_use",
        "name": name,
        "input": args,
        "file_action": tool.file_action if tool else None,
    })

    if diagnostic_reading_active():
        if approval not in {"auto", "full", "confirm"}:
            return _emit_tool_result(
                emit, name,
                f"Blocked: '{name}' cannot run during hang diagnosis because"
                f" permission mode '{approval}' permits no tools.",
            )
        if name not in DIAGNOSTIC_READ_TOOL_NAMES or not _is_confirm_read_only(tool):
            return _emit_tool_result(
                emit, name,
                f"Blocked: '{name}' is unavailable during hang diagnosis;"
                " only local builtin file-reading tools may run.",
            )
        approval = "confirm"
        refusal = _diagnostic_search_refusal(name, args, workspace_path)
        if refusal is not None:
            return _emit_tool_result(emit, name, refusal)

    if skill_planning_active() and not diagnostic_reading_active():
        if name not in SKILL_TOOL_NAMES or not _is_confirm_read_only(tool):
            return _emit_tool_result(
                emit, name,
                f"Blocked: '{name}' is unavailable in skill-only planning;"
                " only list_skills and read_skill may run.",
            )
        approval = "confirm"

    if approval not in {"auto", "full", "confirm"}:
        logger.info("%s mode blocked tool: %s", approval, name)
        result = (
            f"Blocked: '{name}' cannot run because permission mode "
            f"'{approval}' permits no tools."
        )
        return _emit_tool_result(emit, name, result)

    if approval == "confirm" and not _is_confirm_read_only(tool):
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
        # Re-check at execution: block hallucinated full-only calls.
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
        # Supervision can return a diagnosed hang to the model after the
        # owned operation has settled. Elapsed time alone is not a deadline.
        started = time.monotonic()
        outcome = "ok"
        tool_task: asyncio.Task[Any] | None = None
        performance_logger.info(
            "stage=tool_start tool=%s", name,
        )
        try:
            details = ""
            summarizer = getattr(tool, "summarize", None)
            if callable(summarizer):
                try:
                    summary = summarizer(args)
                    if isinstance(summary, str):
                        details = summary[:4_000]
                except Exception:
                    pass  # A status formatter must not prevent tool execution.
            ticket = supervision.new_operation(
                workspace_path, kind=f"tool:{name}", **({"details": details} if details else {}),
            )
            with tool_permission_scope(approval), supervision.operation_scope(ticket):
                # Keep the real tool owner reachable throughout cancellation.
                # wait() never forwards caller cancellation to that owner, so
                # this layer can cancel it once and settle it explicitly.
                tool_task = asyncio.create_task(tool.run(workspace_path, args))
                try:
                    # Grep owns a killable subprocess. Other readers use threads,
                    # so do not pretend their cancellation settles the real read.
                    budget = _DIAGNOSTIC_GREP_BUDGET_SECONDS if diagnostic_reading_active() and name == "grep" else None
                    async with asyncio.timeout(budget):
                        diagnosis = await supervision.wait_owned(tool_task, ticket)
                except TimeoutError:
                    if budget is None:
                        raise
                    await _settle_diagnostic_read(tool_task)
                    outcome = "diagnostic_budget"
                    diagnosis = (
                        "Search exceeded the diagnostic budget; its owned scan has stopped."
                        " Use a narrower source/test path and glob to inspect the suspected cause"
                        " within the remaining diagnosis time. The original task is still running."
                    )
                raw_result: object
                if diagnosis is not None:
                    if outcome != "diagnostic_budget":
                        outcome = "recovered_hang"
                    raw_result = diagnosis
                else:
                    raw_result = tool_task.result()
            if isinstance(raw_result, str):
                result = raw_result
            else:
                outcome = "error"
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
            outcome = "cancelled"
            if tool_task is not None:
                if not tool_task.done() and not tool_task.cancelling():
                    tool_task.cancel()
                while not tool_task.done():
                    try:
                        await asyncio.wait((tool_task,))
                    except asyncio.CancelledError:
                        continue
                try:
                    tool_task.result()
                except BaseException:
                    # A late failure must be observed without replacing the
                    # caller's original cancellation or reporting success.
                    pass
            raise
        except TimeoutError:
            outcome = "timeout"
            result = (
                f"Tool {name} failed: the requested operation timed out."
                " Inspect the command or service that stalled before retrying;"
                " preserve completed work and report any unfinished part."
            )
        except Exception as exc:
            outcome = "error"
            result = f"Tool {name} failed: {exc}"
        finally:
            performance_logger.info(
                "stage=tool tool=%s seconds=%.3f outcome=%s",
                name, time.monotonic() - started, outcome,
            )

    result_limit = _TOOL_RESULT_MAX
    if tool is not None:
        declared_limit = getattr(tool, "max_result_chars", _TOOL_RESULT_MAX)
        if (
            isinstance(declared_limit, int) and not isinstance(declared_limit, bool)
            and declared_limit > 0
        ):
            # Plugin code explicitly opts in; arguments never choose this cap.
            # Keep undeclared/replacement plugins at the default and bound even
            # an incorrectly configured plugin to the largest builtin budget.
            result_limit = min(declared_limit, 64 * 1024)
    if tool is not None and not getattr(tool, "is_plugin", True):
        result_limit = max(result_limit, {
            "read_file": 32 * 1024, "read_files": 64 * 1024,
            "glob": DISCOVERY_RESULT_MAX, "grep": DISCOVERY_RESULT_MAX,
        }.get(name, 0))
    if len(result) > result_limit:
        result = truncate_with_marker(
            result, result_limit,
            "request a narrower result or use this tool's paging options"
            " if supported; say PARTIAL + remainder when work is left",
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
    tool = TOOLS_BY_NAME.get(name)
    return tool.summarize(args) if tool else name


def cli_plugin_prelude() -> str:
    """Prompt addendum enumerating plugins for CLI backends.

    Returns ``""`` if no plugins are loaded. Otherwise returns a
    paragraph describing each plugin (name, description, args, and
    a bash-mode invocation template) so CLI-backed agents that can't
    receive typed tool definitions can still call plugins through
    their built-in ``bash`` / shell tool.
    """
    plugins = [tool for tool in _TOOLS if tool.is_plugin]
    if not plugins:
        return ""

    lines = [
        "Plugins (bash):",
        "Use a module's --help to inspect its argument schema. Calls marked"
        " independent read-only may run together; keep other calls ordered.",
        "",
    ]
    for tool in plugins:
        props = tool.parameters.get("properties", {})
        required = set(tool.parameters.get("required", []))
        args_summary = (
            ", ".join(
                f"{prop_key}{'' if prop_key in required else '?'}"
                for prop_key in props
            )
            or "no args"
        )
        # Real __module__ so python -m works.
        module_path = tool.__class__.__module__
        parallel_hint = (
            " [independent read-only]"
            if getattr(tool, "parallel_safe", False) is True else ""
        )
        lines.append(f"- {tool.name}{parallel_hint}: {tool.description}")
        lines.append(f"  {args_summary} | python -m {module_path} '<JSON>'")
        lines.append("")
    return "\n".join(lines).rstrip()


__all__ = [
    "AUTO_TOOL_SCHEMA",
    "DIAGNOSTIC_READ_TOOL_NAMES",
    "DIAGNOSTIC_READ_TOOL_SCHEMA",
    "READ_ONLY_TOOL_NAMES",
    "READ_ONLY_TOOL_SCHEMA",
    "SKILL_TOOL_NAMES",
    "SKILL_TOOL_SCHEMA",
    "TOOLS_BY_NAME",
    "TOOL_SCHEMA",
    "AgentTool",
    "cli_plugin_prelude",
    "execute_tool",
    "diagnostic_reading_active",
    "diagnostic_reading_scope",
    "parse_openai_call",
    "summarize_tool_use",
    "skill_planning_active",
    "skill_planning_scope",
    "tool_signature",
]
