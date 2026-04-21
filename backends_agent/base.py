"""Backend abstract base class and shared data types."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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
