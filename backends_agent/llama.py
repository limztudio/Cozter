"""llama-server backend: OpenAI-compatible in-process agent loop.

llama-server is a local chat-completions endpoint with no auth. The whole
tool-calling loop lives in :class:`OpenAIChatBackend`; this module only
supplies llama's specifics - the local endpoint URL, dynamic model
discovery from ``/v1/models``, and the per-loop limits from the
``llama_*`` config knobs.

Config: ``config.json``'s ``llama_server_url`` (default
``http://127.0.0.1:8080``). The server must speak OpenAI-compatible
``/v1/chat/completions`` with streaming and tool calls.
"""

from __future__ import annotations

import json
import logging
import urllib.request

from .. import config as cfg
from ._openai_agent import OpenAIChatBackend

logger = logging.getLogger(__name__)


class LlamaBackend(OpenAIChatBackend):
    name = "llama"
    executable = "llama-server"  # only used in "not found" error text

    # The model list is populated dynamically from /v1/models on first
    # access. Stored as a tuple per the Backend contract.
    default_model = "auto"
    default_summary_model = "auto"

    def __init__(self) -> None:
        self._cached_models: tuple[str, ...] | None = None

    # ---- model discovery ------------------------------------------------

    @property
    def available_models(self) -> tuple[str, ...]:  # type: ignore[override]
        if self._cached_models is None:
            self._cached_models = self._fetch_models()
        return self._cached_models

    def _fetch_models(self) -> tuple[str, ...]:
        url = cfg.get_llama_server_url().rstrip("/") + "/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            ids = tuple(
                m["id"] for m in payload.get("data", [])
                if isinstance(m, dict) and isinstance(m.get("id"), str)
            )
            return ids or ("auto",)
        except Exception as exc:
            logger.debug(
                "Could not query %s for models (%s); using 'auto'",
                url, exc,
            )
            return ("auto",)

    def health_check(self) -> tuple[bool, str]:
        # llama is an HTTP endpoint, not a CLI: probe /v1/models instead of
        # looking for a binary on PATH.
        url = cfg.get_llama_server_url().rstrip("/") + "/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return False, f"server unreachable at {url}: {exc}"
        ids = [
            m.get("id") for m in payload.get("data", [])
            if isinstance(m, dict) and isinstance(m.get("id"), str)
        ]
        if ids:
            return True, f"server up at {url} ({len(ids)} model(s))"
        return True, f"server up at {url} (no models listed)"

    # ---- OpenAIChatBackend hooks ---------------------------------------

    def _chat_endpoint(self) -> str:
        return cfg.get_llama_server_url().rstrip("/") + "/v1/chat/completions"

    def _request_model(self, model: str | None) -> str | None:
        # ``model="auto"`` (our sentinel default) means "let the server
        # decide": drop the field and llama-server uses its loaded model.
        return None if not model or model == "auto" else model

    def _max_agent_turns(self) -> int:
        return cfg.get_llama_max_agent_turns()

    def _tool_repeat_limit(self) -> int:
        return cfg.get_llama_tool_repeat_limit()

    def _socket_timeout(self) -> int:
        return cfg.get_llama_socket_timeout()

    def _max_retries(self) -> int:
        return cfg.get_llama_max_retries()
