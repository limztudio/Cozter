"""Agent backends - per-CLI adapters for running the AI agent.

A backend encapsulates the three things that differ between CLIs:
  1. How to launch the subprocess (args, cwd, prompt delivery mechanism)
  2. How to parse its streamed JSON events into ChatEvent objects
  3. How to extract the agent's final text reply (for compaction)

The orchestrator in agent.py is backend-neutral: session management,
context building, compaction policy, and session logging stay there.
"""

from .base import Backend
from .claude_code import ClaudeCodeBackend
from .codex import CodexBackend
from .copilot import CopilotBackend
from .llama import LlamaBackend
from .zai import ZaiBackend

_BACKENDS: dict[str, Backend] = {
    "codex": CodexBackend(),
    "copilot": CopilotBackend(),
    "claude_code": ClaudeCodeBackend(),
    "llama": LlamaBackend(),
    "zai": ZaiBackend(),
}

AVAILABLE_BACKENDS = list(_BACKENDS.keys())
DEFAULT_BACKEND = "codex"


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
    "Backend",
    "ClaudeCodeBackend",
    "CodexBackend",
    "CopilotBackend",
    "LlamaBackend",
    "ZaiBackend",
    "get_backend",
]
