"""Agent backends - per-CLI adapters for running the AI agent.

A backend encapsulates the three things that differ between CLIs:
  1. How to launch the subprocess (args, cwd, prompt delivery mechanism)
  2. How to parse its streamed JSON events into ChatEvent objects
  3. How to extract the agent's final text reply (for compaction)

The orchestrator in agent.py is backend-neutral: session management,
context building, compaction policy, and session logging stay there.

One entry is not a CLI at all: ``flexible`` is a meta-agent that routes
each sub-task of a request to one of the *direct* backends below. Hence
the two lists - :data:`DIRECT_BACKENDS` are the ones that can actually
run a turn (and so are the only valid summary agents and flexible tiers),
while :data:`AVAILABLE_BACKENDS` is what a user may pick to chat with.
"""

from .base import Backend
from .claude_code import ClaudeCodeBackend
from .codex import CodexBackend
from .copilot import CopilotBackend
from .flexible import FlexibleBackend
from .llama import LlamaBackend
from .zai import ZaiBackend

_DIRECT: dict[str, Backend] = {
    "codex": CodexBackend(),
    "copilot": CopilotBackend(),
    "claude_code": ClaudeCodeBackend(),
    "llama": LlamaBackend(),
    "zai": ZaiBackend(),
}

FLEXIBLE_BACKEND = FlexibleBackend.name

_BACKENDS: dict[str, Backend] = {
    **_DIRECT,
    FLEXIBLE_BACKEND: FlexibleBackend(),
}

# Backends that own a real CLI/HTTP turn. Summary agents and flexible's
# difficulty tiers must come from this list - pointing either at flexible
# itself would recurse.
DIRECT_BACKENDS = list(_DIRECT.keys())

# Everything a user can select as their chat agent.
AVAILABLE_BACKENDS = list(_BACKENDS.keys())

DEFAULT_BACKEND = FLEXIBLE_BACKEND
# Fallback for every role flexible cannot fill: summary agent, and the
# agent behind each of flexible's own difficulty tiers.
DEFAULT_DIRECT_BACKEND = "codex"


def get_backend(name: str | None) -> Backend:
    """Return the Backend for *name*, falling back to the default."""
    if not name:
        return _BACKENDS[DEFAULT_BACKEND]
    backend = _BACKENDS.get(name)
    if backend is None:
        raise ValueError(
            f"Unknown backend: {name}. "
            f"Available: {', '.join(AVAILABLE_BACKENDS)}"
        )
    return backend


__all__ = [
    "AVAILABLE_BACKENDS",
    "DEFAULT_BACKEND",
    "DEFAULT_DIRECT_BACKEND",
    "DIRECT_BACKENDS",
    "FLEXIBLE_BACKEND",
    "Backend",
    "ClaudeCodeBackend",
    "CodexBackend",
    "CopilotBackend",
    "FlexibleBackend",
    "LlamaBackend",
    "ZaiBackend",
    "get_backend",
]
