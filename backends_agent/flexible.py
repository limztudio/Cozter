"""The ``flexible`` meta-agent's backend entry.

Flexible has no CLI of its own: it plans a request, routes each sub-task
to the agent+model bound to that sub-task's difficulty tier, and merges
the results. :mod:`Cozter.agent` intercepts it before any launch, so this
class exists only to give the meta-agent a seat in the backend registry —
which is what lets it be picked by /agent, validated by the workspace
settings layer, and resolved by ``get_backend`` like any other agent.
"""

import asyncio

from ..flexible import BACKEND_NAME, TIERS
from .base import AgentResult, Backend


class FlexibleBackend(Backend):
    name = BACKEND_NAME
    executable = "flexible"  # never spawned; kept for error-message parity

    # Flexible has no model of its own — each tier carries one. The /model
    # picker special-cases the empty list and points at the per-tier
    # commands instead.
    available_models = ()
    default_model = ""
    default_summary_model = ""
    effort_levels = ()

    def health_check(self) -> tuple[bool, str]:
        """Always ready: the tiers' own health is what actually matters."""
        return True, f"meta-agent (routes to its {'/'.join(TIERS)} agents)"

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
        raise RuntimeError(
            "the flexible agent is orchestrated by agent.py and never"
            " launched as a subprocess"
        )

    def parse_event(self, event: dict, result: AgentResult) -> None:
        return None

    def extract_agent_text(self, event: dict) -> str | None:
        return None
