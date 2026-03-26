"""Client for the ChatGPT backend API (not the public API)."""

import logging

import httpx

from . import auth

logger = logging.getLogger(__name__)

CHATGPT_BASE = "https://chatgpt.com/backend-api"


class ChatGPTClient:
    def __init__(self):
        self._client = httpx.AsyncClient(timeout=60.0)

    async def close(self):
        await self._client.aclose()

    def _headers(self, tokens: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {tokens['access_token']}",
            "Content-Type": "application/json",
        }
        if tokens.get("account_id"):
            headers["chatgpt-account-id"] = tokens["account_id"]
        return headers

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        tokens = auth.refresh_if_needed()
        if not tokens:
            raise RuntimeError("Not logged in. Use /login first.")

        url = f"{CHATGPT_BASE}/{path.lstrip('/')}"
        resp = await self._client.request(method, url, headers=self._headers(tokens), **kwargs)

        # Retry once on 401 after forcing a refresh
        if resp.status_code == 401:
            tokens = auth._refresh_tokens(tokens)
            if not tokens:
                raise RuntimeError("Session expired. Use /login to re-authenticate.")
            resp = await self._client.request(method, url, headers=self._headers(tokens), **kwargs)

        resp.raise_for_status()
        return resp

    async def get_models(self) -> list[str]:
        resp = await self._request("GET", "/models")
        data = resp.json()
        return [m.get("slug", m.get("id", "?")) for m in data.get("models", [])]

    async def send_message(
        self,
        message: str,
        model: str = "gpt-4o",
        conversation_id: str | None = None,
    ) -> dict:
        """Send a message and return the full response dict."""
        payload = {
            "action": "next",
            "messages": [
                {
                    "role": "user",
                    "content": {"content_type": "text", "parts": [message]},
                }
            ],
            "model": model,
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id

        resp = await self._request("POST", "/conversation", json=payload)
        return resp.json()
