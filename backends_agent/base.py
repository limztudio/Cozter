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


def executable_command(name: str) -> list[str]:
    """Return a launch argv prefix, falling back to subprocess lookup."""
    return resolve_executable_prefix(name) or [name]


def effort_band(percent: int, levels: tuple[str, ...]) -> str | None:
    """Map 1-100 onto evenly sized effort *levels*; 0 disables override."""
    if percent <= 0 or not levels:
        return None
    idx = min(percent * len(levels) // 100, len(levels) - 1)
    return levels[idx]


@dataclass
class ChatEvent:
    """An event produced during an agent turn."""
    kind: str  # "tool", "file", "text", "attachment"
    content: str


@dataclass
class AgentResult:
    """Collected result from a single agent run.

    *error* is populated by :meth:`Backend.parse_event` when the
    backend emits an error event mid-stream (server-side tool
    failure, stream truncation, etc.). Individual backends should
    set it consistently rather than only writing to ``text``.
    """
    events: list[ChatEvent] = field(default_factory=list)
    text: str = "(no response)"
    error: str | None = None
    # Token/cost usage for the turn, when the backend reports it (codex's
    # turn.completed, claude_code's result). Backend-shaped dict; None
    # otherwise. See agent.format_usage for the display formatter.
    usage: dict | None = None


def append_text_result(result: AgentResult, text: str) -> None:
    """Record text as the latest agent reply and emit a text event."""
    result.text = text
    result.events.append(ChatEvent(kind="text", content=text))


def set_error_result(
    result: AgentResult,
    message: str,
    *,
    display_text: str | None = None,
) -> None:
    """Record an error and emit its user-facing text event."""
    result.error = message
    append_text_result(result, display_text or f"Error: {message}")


def truncate_status_text(text: object, *, limit: int = 200) -> str:
    """Return a clipped preview for status events."""
    value = text if isinstance(text, str) else str(text)
    return value if len(value) <= limit else value[:limit] + "..."


async def create_prompt_subprocess(
    cmd: list[str],
    prompt: str,
    *,
    cwd: str | None = None,
) -> asyncio.subprocess.Process:
    """Spawn a JSONL CLI backend and write the prompt to stdin."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    if proc.stdin is None:
        raise RuntimeError("subprocess stdin pipe was not created")
    proc.stdin.write(prompt.encode("utf-8"))
    await proc.stdin.drain()
    proc.stdin.close()
    await proc.stdin.wait_closed()
    return proc


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
    effort_levels: tuple[str, ...] = ()

    # Default model per difficulty tier of the ``flexible`` meta-agent,
    # keyed by "low"/"mid"/"high" (see :mod:`Cozter.flexible`). Backends
    # whose catalog has no meaningful cheap/strong spread leave this empty
    # and fall back to :attr:`default_model` for every tier.
    tier_models: dict[str, str] = {}

    # True for backends that consume :data:`agent_tools.TOOL_SCHEMA`
    # directly as typed tool definitions (llama and any future
    # OpenAI-shape HTTP backend). False for CLI subprocess backends
    # whose tool ecosystem is baked into the binary - those see user
    # plugins via the bash prelude generated by
    # :func:`agent_tools.cli_plugin_prelude`.
    supports_typed_plugins: bool = False

    # Whether the bash/shell prelude in agent.py should be added when
    # supports_typed_plugins is False. CLI backends (codex/copilot/
    # claude_code) keep this True because their model can shell-invoke
    # ``python -m Cozter.agent_tools.plugins.<name>`` via their bundled
    # bash tool. Pure HTTP-chat backends with no shell tool of their own
    # set this False - the prelude would describe plugins the model
    # has no way to actually call.
    supports_plugin_prelude: bool = True

    # Behavior -------------------------------------------------------------

    def health_check(self) -> tuple[bool, str]:
        """Report whether this backend is ready to run a turn.

        Default (CLI backends): the executable must resolve on PATH. HTTP
        backends override this to probe their endpoint instead. Returns
        ``(ok, detail)`` where *detail* is a short human-readable status.
        Blocking (PATH / network lookups) - call it off the event loop.
        """
        prefix = resolve_executable_prefix(self.executable)
        if prefix is None:
            return False, f"{self.executable!r} not found on PATH"
        return True, f"{self.executable} ({prefix[-1]})"

    @abstractmethod
    async def launch(
        self,
        workspace_path: str,
        prompt: str,
        model: str | None,
        approval: str,
        *,
        compaction: bool = False,
        effort: int = 0,
    ) -> asyncio.subprocess.Process:
        """Spawn the CLI subprocess with *prompt* delivered appropriately.

        Returns the running subprocess with stdout/stderr piped. The caller
        reads stdout lines and feeds them through ``parse_event``.

        compaction=True indicates this is an internal summarization call
        (no user-facing tool use). Backends typically translate this to
        a broader approval scope since compaction is trusted.

        effort is a 0-100 percentage of "how hard the model should
        think". 0 means "do not send a reasoning-effort signal at all"
        (server defaults apply). 1-100 are translated to each backend's
        native vocabulary via :meth:`convert_effort`.
        """

    def tier_model(self, tier: str) -> str:
        """Default model for one of flexible's difficulty tiers."""
        return self.tier_models.get(tier) or self.default_model

    def convert_effort(self, percent: int) -> str | None:
        """Translate a 0-100 percentage to the backend's native effort form.

        Return ``None`` to skip sending an effort signal - either because
        the percentage is 0 (no override) or because the backend leaves
        :attr:`effort_levels` empty.
        """
        return effort_band(percent, self.effort_levels)

    def append_model_effort_args(
        self,
        cmd: list[str],
        model: str | None,
        effort: int,
        *,
        model_flag: str = "--model",
        effort_flag: str = "--effort",
        effort_template: str = "{effort}",
        effort_levels: tuple[str, ...] | None = None,
    ) -> None:
        """Append optional model and reasoning-effort CLI arguments."""
        if model:
            cmd += [model_flag, model]
        native_effort = (
            effort_band(effort, effort_levels)
            if effort_levels is not None
            else self.convert_effort(effort)
        )
        if native_effort:
            cmd += [
                effort_flag,
                effort_template.format(effort=native_effort),
            ]

    @abstractmethod
    def parse_event(self, event: dict, result: AgentResult) -> None:
        """Mutate *result* based on a single parsed JSON event line."""

    @abstractmethod
    def extract_agent_text(self, event: dict) -> str | None:
        """Return the agent's final text from *event*, if it carries one.

        Used by the compaction code path to pick out the last assistant
        message in the stream. Returns None for tool/file/unknown events.
        """
