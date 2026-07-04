"""Z.ai (Zhipu GLM) backend: OpenAI-compatible cloud API.

Z.ai serves the GLM models (glm-5.2, glm-4.7, ...) through an
OpenAI-compatible endpoint at ``https://api.z.ai/api/paas/v4`` with Bearer
auth. It reuses the shared :class:`OpenAIChatBackend` loop; this module
supplies only Z.ai's specifics - the endpoint, the Authorization header
built from the configured API key, the model, and the GLM model list.

Config: ``config.json``'s ``zai_api_key`` (required to use it),
``zai_base_url`` (default ``https://api.z.ai/api/paas/v4``, already
includes the version so only ``/chat/completions`` is appended),
``zai_socket_timeout``, and ``zai_max_retries``. Pick the model with
``/model`` (or set the workspace default); add newer GLM ids via
``extra_models`` in config without editing source.
"""

from __future__ import annotations

from .. import config as cfg
from ._openai_agent import OpenAIChatBackend


class ZaiBackend(OpenAIChatBackend):
    name = "zai"
    executable = "z.ai"  # HTTP backend; never spawns a subprocess

    # A snapshot of Z.ai's GLM lineup (glm-5.2 is the current flagship).
    # The exact set evolves; add newer ids via config `extra_models` - the
    # /model picker merges them - rather than editing this list.
    available_models = (
        "glm-5.2",
        "glm-4.7",
        "glm-4.6",
        "glm-4.5",
        "glm-4.5-air",
        "glm-4.5-flash",
        "glm-4-air",
    )
    default_model = "glm-5.2"
    default_summary_model = "glm-4.5-flash"

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
