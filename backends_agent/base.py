"""Backend abstract base class and shared data types."""

import asyncio
import shutil
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


def resolve_executable_prefix(name: str) -> list[str] | None:
    """Resolve *name* to a launchable subprocess argv prefix.

    Windows ships many CLIs (notably npm-installed ones like ``codex.cmd``
    and ``copilot.cmd``) as ``.cmd`` shims. ``CreateProcessW`` - used by
    ``asyncio.create_subprocess_exec`` - auto-appends ``.exe`` but does
    *not* search PATHEXT, so the shim is invisible to subprocess even
    though ``cmd.exe`` finds it via ``where``. Detect that case and
    wrap with ``cmd.exe /c`` so the shim runs.

    Returns None if the binary cannot be found anywhere on PATH; callers
    typically fall back to ``[name]`` and let subprocess raise its own
    FileNotFoundError so the existing error path stays consistent.
    """
    found = shutil.which(name)
    if found is None:
        return None
    if sys.platform == "win32" and found.lower().endswith((".cmd", ".bat")):
        return ["cmd.exe", "/c", found]
    return [found]


@dataclass
class ChatEvent:
    """An event produced during an agent turn."""
    kind: str  # "tool", "file", "text"
    content: str


@dataclass
class AgentResult:
    """Collected result from a single agent run."""
    events: list[ChatEvent] = field(default_factory=list)
    text: str = "(no response)"


class Backend(ABC):
    """Adapter for a specific agent CLI (codex, copilot, ...).

    Each concrete backend knows:
      - how to build the argv + spawn the subprocess (including how the
        prompt is delivered: stdin, argv, temp file, etc.)
      - how to translate the CLI's JSONL event schema into ChatEvents
      - how to pull the agent's final text reply out of the stream
        (used only by the compaction code path)
    """

    # Class-level metadata -------------------------------------------------
    name: str = ""
    executable: str = ""  # binary name used in "CLI not found" messages
    available_models: tuple[str, ...] = ()
    default_model: str = ""
    default_summary_model: str = ""

    # Behavior -------------------------------------------------------------

    @abstractmethod
    async def launch(
        self,
        workspace_path: str,
        prompt: str,
        model: str | None,
        approval: str,
        *,
        compaction: bool = False,
    ) -> asyncio.subprocess.Process:
        """Spawn the CLI subprocess with *prompt* delivered appropriately.

        Returns the running subprocess with stdout/stderr piped. The caller
        reads stdout lines and feeds them through ``parse_event``.

        compaction=True indicates this is an internal summarization call
        (no user-facing tool use). Backends typically translate this to
        a broader approval scope since compaction is trusted.
        """

    @abstractmethod
    def parse_event(self, event: dict, result: AgentResult) -> None:
        """Mutate *result* based on a single parsed JSON event line."""

    @abstractmethod
    def extract_agent_text(self, event: dict) -> str | None:
        """Return the agent's final text from *event*, if it carries one.

        Used by the compaction code path to pick out the last assistant
        message in the stream. Returns None for tool/file/unknown events.
        """
