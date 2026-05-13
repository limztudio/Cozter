"""Base interface for agent tools and helpers shared across tools."""

from __future__ import annotations

import html
import os
import re
from abc import ABC, abstractmethod
from typing import Any

import aiohttp


# Hard cap on raw HTTP body bytes per web tool call so a pathological
# URL can't OOM the bot.
_MAX_FETCH_BYTES = 5 * 1024 * 1024  # 5 MB


class AgentTool(ABC):
    """One tool an agent backend can invoke.

    Backend-agnostic: any agent loop (llama-server, OpenAI, Mistral,
    Gemini, Claude API, etc.) can drive these tools. The tools only
    need a workspace path and an args dict; nothing about how the
    model emitted the call leaks into them.

    Subclasses must define:
      - ``name``: identifier the model uses to call the tool.
      - ``description``: model-facing tool description.
      - ``parameters``: JSON-Schema dict for tool arguments.
      - ``run(workspace_path, args)``: async, returns the result string
        the model will read back.

    Subclasses may optionally set:
      - ``file_action``: one of ``"write"``, ``"edit"``, ``"delete"`` if
        the call should surface as a file-status event in the UI.
      - ``order``: integer for the tool-list ordering sent to the model.
        Lower comes first; ties broken alphabetically by ``name``.
        Defaults to 100.
      - ``summarize(args)``: one-line status-display formatter. The
        default returns just the tool name.

    Every concrete subclass auto-registers itself in
    ``AgentTool.registry`` at class-definition time, so adding a new
    tool only requires dropping a new file in this package - no edits
    to ``__init__.py`` or any backend module are needed.
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}
    file_action: str | None = None
    order: int = 100

    # Populated by __init_subclass__. Read by the package's __init__.
    registry: list["AgentTool"] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Skip intermediate abstract classes that don't implement run().
        if getattr(cls, "__abstractmethods__", None):
            return
        instance = cls()
        # Idempotent: replace any prior registration with the same name
        # so a hot-reload doesn't accumulate duplicates.
        AgentTool.registry[:] = [
            t for t in AgentTool.registry if t.name != instance.name
        ]
        AgentTool.registry.append(instance)

    @property
    def schema(self) -> dict[str, Any]:
        """Return the inner ``function`` dict for OpenAI tool definitions."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @abstractmethod
    async def run(self, workspace_path: str, args: dict) -> str:
        """Execute the tool and return a string the model can read."""

    def summarize(self, args: dict) -> str:
        """One-line summary for the agent's status display."""
        return self.name


# ---------------------------------------------------------------------------
# Helpers shared across tools
# ---------------------------------------------------------------------------


def resolve_inside_workspace(workspace: str, path: str) -> str:
    """Return absolute path; raise if it escapes the workspace.

    ``path`` may be relative to the workspace root or an absolute path
    inside it. Symlinks are followed via ``os.path.realpath`` and the
    resolved target must stay under the workspace root.
    """
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


async def read_bounded_text(resp: aiohttp.ClientResponse) -> str:
    """Read up to MAX_FETCH_BYTES from *resp* and decode with its charset."""
    body_bytes = await resp.content.read(_MAX_FETCH_BYTES + 1)
    encoding = resp.charset or "utf-8"
    return body_bytes.decode(encoding, errors="replace")


def html_to_text(value: str) -> str:
    """Strip script/style blocks and remaining tags; collapse whitespace."""
    value = re.sub(
        r"<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"<style\b[^<]*(?:(?!</style>)<[^<]*)*</style>",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()
