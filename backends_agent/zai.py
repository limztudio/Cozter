"""Z.ai (Zhipu GLM) backend: OpenAI-compatible cloud API.

Z.ai serves the GLM models (glm-5.2, glm-5.1, glm-4.7, ...) through an
OpenAI-compatible endpoint at ``https://api.z.ai/api/paas/v4`` with Bearer
auth. It reuses the shared :class:`OpenAIChatBackend` loop; this module
supplies only Z.ai's specifics - the endpoint, the Authorization header
built from the configured API key, the model, and the GLM model list.

Config: ``config.json``'s ``zai_api_key`` (required to use it),
``zai_base_url`` (default ``https://api.z.ai/api/paas/v4``, already
includes the version so only ``/chat/completions`` is appended),
``zai_socket_timeout``, and ``zai_max_retries``. Pick the model with
``/model`` (or set the workspace default); add private or regional GLM ids via
``extra_models`` in config without editing source.
"""

from __future__ import annotations

from .. import config as cfg
from ._openai_agent import OpenAIChatBackend


class ZaiBackend(OpenAIChatBackend):
    name = "zai"
    executable = "z.ai"  # HTTP backend; never spawns a subprocess

    # Snapshot of Z.ai's documented text-model catalog plus GLM-5.2, the
    # current flagship. The exact set evolves; add private/regional ids via
    # config `extra_models` - the /model picker merges them.
    available_models = (
        "glm-5.2",
        "glm-5.1",
        "glm-5",
        "glm-5-turbo",
        "glm-4.7",
        "glm-4.7-flash",
        "glm-4.7-flashx",
        "glm-4.6",
        "glm-4.5",
        "glm-4.5-x",
        "glm-4.5-air",
        "glm-4.5-airx",
        "glm-4.5-flash",
        "glm-4-32b-0414-128k",
    )
    default_model = "glm-5.2"
    default_summary_model = "glm-4.5-air"
    # GLM-5.2 accepts OpenAI-compatible aliases: low/medium map to high
    # server-side, and xhigh maps to max. 0 still skips the field.
    effort_levels = ("low", "medium", "high", "xhigh", "max")

    # ---- OpenAIChatBackend hooks ---------------------------------------

    def _chat_endpoint(self) -> str:
        # base_url already carries the /api/paas/v4 version segment, so we
        # append /chat/completions directly (NOT /v1/chat/completions).
        return cfg.get_zai_base_url().rstrip("/") + "/chat/completions"

    def _auth_headers(self) -> dict[str, str]:
        key = cfg.get_zai_api_key()
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _request_model(self, model: str | None) -> str:
        # Z.ai requires a model field; fall back to the configured default.
        return model or self.default_model

    def _auto_continue_after_tool_limit(self) -> bool:
        # Long z.ai coding runs can legitimately need more tool turns than
        # Cozter's per-segment guard. Keep going in a fresh segment instead
        # of forcing a no-tools final answer.
        return True

    def _socket_timeout(self) -> int:
        return cfg.get_zai_socket_timeout()

    def _max_retries(self) -> int:
        return cfg.get_zai_max_retries()

    def health_check(self) -> tuple[bool, str]:
        # HTTP backend: readiness is "is an API key configured?". We don't
        # spend a real request here (that would bill the account).
        if not cfg.get_zai_api_key():
            return False, "no API key set (set zai_api_key in config.json)"
        return True, (
            f"configured (endpoint {self._chat_endpoint()},"
            f" default model {self.default_model})"
        )
