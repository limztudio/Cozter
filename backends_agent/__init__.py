"""Agent backends - per-CLI adapters for running the AI agent.

A backend encapsulates the three things that differ between CLIs:
  1. How to launch the subprocess (args, cwd, prompt delivery mechanism)
  2. How to parse its streamed JSON events into ChatEvent objects
  3. How to extract the agent's final text reply (for compaction)

The orchestrator in agent.py is backend-neutral: session management,
context building, compaction policy, and session logging stay there.

One entry is not a CLI at all: ``flexible`` is a meta-agent that routes
each sub-task of a request to one of the *direct* backends below. Hence
the lists - :data:`DIRECT_BACKENDS` can run turns and Flexible tiers,
:data:`SUMMARY_BACKENDS` also enforce the restrictions required for internal
calls, and :data:`AVAILABLE_BACKENDS` is what a user may pick to chat with.
"""

from .base import Backend
from .antigravity_cli import AntigravityCliBackend
from .claude_code import ClaudeCodeBackend
from .codex import CodexBackend
from .copilot import CopilotBackend
from .flexible import FlexibleBackend
from .gemini_cli import GeminiCliBackend
from .grok import GrokBackend
from .llama import LlamaBackend
from .meta import MetaModelApiBackend
from .zai import ZaiBackend

_DIRECT: dict[str, Backend] = {
    "codex": CodexBackend(),
    "copilot": CopilotBackend(),
    "claude_code": ClaudeCodeBackend(),
    "grok": GrokBackend(),
    "gemini_cli": GeminiCliBackend(),
    "antigravity_cli": AntigravityCliBackend(),
    "llama": LlamaBackend(),
    "meta": MetaModelApiBackend(),
    "zai": ZaiBackend(),
}

FLEXIBLE_BACKEND = FlexibleBackend.name

_BACKENDS: dict[str, Backend] = {
    **_DIRECT,
    FLEXIBLE_BACKEND: FlexibleBackend(),
}

# Turn-capable only: summary agents/tiers must come from here
# (flexible would recurse).
DIRECT_BACKENDS = list(_DIRECT.keys())
SUMMARY_BACKENDS = [name for name, backend in _DIRECT.items() if backend.supports_internal_calls]

# Meta-agent first: the /agent picker gives it the dedicated ``0`` slot.
AVAILABLE_BACKENDS = [FLEXIBLE_BACKEND, *_DIRECT]

DEFAULT_BACKEND = FLEXIBLE_BACKEND
# Fallback when flexible can't fill a role (summary agent, own tiers).
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
    "SUMMARY_BACKENDS",
    "FLEXIBLE_BACKEND",
    "Backend",
    "AntigravityCliBackend",
    "ClaudeCodeBackend",
    "CodexBackend",
    "CopilotBackend",
    "FlexibleBackend",
    "GeminiCliBackend",
    "GrokBackend",
    "LlamaBackend",
    "MetaModelApiBackend",
    "ZaiBackend",
    "get_backend",
]
