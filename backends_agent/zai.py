"""Z.ai (Zhipu GLM) backend: OpenAI-compatible cloud API.

Z.ai serves the GLM models (glm-5.2, glm-5.1, glm-5, ...) through an
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

import json
import logging
import threading
import urllib.request

from .. import config as cfg
from ._openai_agent import OpenAIChatBackend, extract_model_ids

logger = logging.getLogger(__name__)


# Safety net for unavailable/unauthorized model discovery.  The installed
# account's ``/models`` catalog is preferred whenever it can be queried.
_FALLBACK_MODELS = (
    "glm-5.2",
    "glm-5.1",
    "glm-5-turbo",
    "glm-5",
    "glm-4.7",
    "glm-4.7-flash",
    "glm-4.7-flashx",
    "glm-4.6",
    "glm-4.5",
    "glm-4.5-air",
    "glm-4.5-x",
    "glm-4.5-airx",
    "glm-4.5-flash",
    "glm-4-32b-0414-128k",
)
_MODEL_DISCOVERY_TIMEOUT_SEC = 10


class ZaiBackend(OpenAIChatBackend):
    name = "zai"
    executable = "z.ai"  # HTTP backend; never spawns a subprocess

    default_model = "glm-5.2"
    default_summary_model = "glm-4.5-air"
    tier_models = {"low": "glm-4.5-air", "mid": "glm-4.7", "high": "glm-5.2"}
    # GLM-5.2 accepts seven reasoning-effort values. Other current text models
    # expose only the thinking switch, handled separately in _effort_fields.
    effort_levels = (
        "none", "minimal", "low", "medium", "high", "xhigh", "max",
    )

    def __init__(self) -> None:
        # The backend is process-wide, so one account-specific lookup is
        # sufficient and avoids an HTTP round trip for every picker open.
        self._cached_models: tuple[str, ...] | None = None
        self._models_lock = threading.Lock()

    # ---- model discovery -----------------------------------------------

    @property
    def available_models(self) -> tuple[str, ...]:  # type: ignore[override]
        """Models exposed by the configured Z.ai account.

        Z.ai's OpenAI-compatible ``/models`` endpoint can include private or
        plan-specific IDs that a public static list cannot know.  Any missing
        key, endpoint failure, malformed response, or empty catalog falls
        back to the curated list above.
        """
        if self._cached_models is None:
            with self._models_lock:
                if self._cached_models is None:
                    self._cached_models = self._fetch_models()
        return self._cached_models

    def _models_endpoint(self) -> str:
        return cfg.get_zai_base_url().rstrip("/") + "/models"

    def _fetch_models(self) -> tuple[str, ...]:
        key = cfg.get_zai_api_key()
        if not key:
            logger.debug("Z.ai API key is unset; using fallback model list")
            return _FALLBACK_MODELS

        url = self._models_endpoint()
        request = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {key}"}, method="GET",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=_MODEL_DISCOVERY_TIMEOUT_SEC,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            logger.debug(
                "Could not query Z.ai models at %s (%s); using fallback",
                url, exc,
            )
            return _FALLBACK_MODELS

        return extract_model_ids(payload) or _FALLBACK_MODELS

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

    def _effort_fields(
        self,
        percent: int,
        model: str | None = None,
    ) -> dict:
        if percent <= 0:
            return {}
        if model == "glm-5.2":
            return {
                "thinking": {"type": "enabled"},
                "reasoning_effort": self.convert_effort(percent),
            }
        return {
            "thinking": {
                "type": "enabled" if percent >= 50 else "disabled",
            },
        }

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
