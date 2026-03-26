"""OpenAI ChatGPT OAuth authentication via device code flow."""

import json
import hashlib
import base64
import logging
import os
import secrets
import time
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

AUTH_BASE = "https://auth.openai.com"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DEVICE_CODE_URL = f"{AUTH_BASE}/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = f"{AUTH_BASE}/api/accounts/deviceauth/token"
TOKEN_URL = f"{AUTH_BASE}/oauth/token"
DEVICE_VERIFY_URL = f"{AUTH_BASE}/codex/device"

CONFIG_DIR = os.path.join(os.path.dirname(__file__), ".config")
AUTH_PATH = os.path.join(CONFIG_DIR, "auth.json")

TOKEN_REFRESH_DAYS = 8


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------

def _generate_pkce() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) for S256 PKCE."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


# ---------------------------------------------------------------------------
# Token storage
# ---------------------------------------------------------------------------

def _load_auth() -> dict | None:
    if os.path.exists(AUTH_PATH):
        with open(AUTH_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_auth(data: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(AUTH_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def is_logged_in() -> bool:
    auth = _load_auth()
    return auth is not None and bool(auth.get("access_token"))


def get_tokens() -> dict | None:
    """Return stored auth data, or None if not logged in."""
    return _load_auth()


def clear_auth() -> None:
    if os.path.exists(AUTH_PATH):
        os.remove(AUTH_PATH)


# ---------------------------------------------------------------------------
# JWT helpers (decode without verification for expiry check)
# ---------------------------------------------------------------------------

def _decode_jwt_payload(token: str) -> dict:
    payload = token.split(".")[1]
    padding = 4 - len(payload) % 4
    if padding != 4:
        payload += "=" * padding
    return json.loads(base64.urlsafe_b64decode(payload))


def _is_token_expired(auth: dict) -> bool:
    try:
        claims = _decode_jwt_payload(auth["access_token"])
        return claims.get("exp", 0) < time.time()
    except Exception:
        return True


def _is_token_stale(auth: dict) -> bool:
    """Token is stale if expired or last refresh was > 8 days ago."""
    if _is_token_expired(auth):
        return True
    last = auth.get("last_refresh")
    if not last:
        return True
    try:
        dt = datetime.fromisoformat(last)
        age = (datetime.now(timezone.utc) - dt).days
        return age >= TOKEN_REFRESH_DAYS
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Device code flow
# ---------------------------------------------------------------------------

def request_device_code() -> dict:
    """
    Start device code flow.
    Returns dict with device_auth_id, user_code, interval.
    """
    resp = httpx.post(
        DEVICE_CODE_URL,
        json={"client_id": CLIENT_ID},
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()


def poll_device_code(device_auth_id: str, user_code: str, interval: int, timeout: int = 900) -> dict:
    """
    Poll until the user completes auth or timeout.
    Returns the token response dict.
    Raises TimeoutError or RuntimeError on failure.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(interval)
        resp = httpx.post(
            DEVICE_TOKEN_URL,
            json={"device_auth_id": device_auth_id, "user_code": user_code},
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code == 200:
            data = resp.json()
            if "authorization_code" in data:
                return _exchange_auth_code(
                    data["authorization_code"],
                    data.get("code_verifier", ""),
                )
            return data
        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        error = body.get("error", "")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 1
            continue
        if error == "expired_token":
            raise TimeoutError("Device code expired. Please try /login again.")
        raise RuntimeError(f"Device auth failed: {resp.text}")
    raise TimeoutError("Login timed out. Please try /login again.")


def _exchange_auth_code(code: str, code_verifier: str) -> dict:
    """Exchange authorization_code for access + refresh tokens."""
    resp = httpx.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": f"http://localhost:1455/auth/callback",
            "client_id": CLIENT_ID,
            "code_verifier": code_verifier,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    resp.raise_for_status()
    return resp.json()


def save_tokens(token_resp: dict) -> dict:
    """Persist tokens from an OAuth response and return the auth dict."""
    id_token = token_resp.get("id_token", "")
    claims = {}
    if id_token:
        try:
            claims = _decode_jwt_payload(id_token)
        except Exception:
            pass

    auth = {
        "access_token": token_resp.get("access_token", ""),
        "refresh_token": token_resp.get("refresh_token", ""),
        "id_token": id_token,
        "account_id": claims.get("https://api.openai.com/auth", {}).get("account_id", ""),
        "email": claims.get("email", ""),
        "plan": claims.get("https://api.openai.com/auth", {}).get("chatgpt_plan_type", ""),
        "last_refresh": datetime.now(timezone.utc).isoformat(),
    }
    _save_auth(auth)
    return auth


# ---------------------------------------------------------------------------
# Token refresh
# ---------------------------------------------------------------------------

def refresh_if_needed() -> dict | None:
    """Refresh tokens if stale. Returns current auth dict or None."""
    auth = _load_auth()
    if not auth:
        return None
    if not _is_token_stale(auth):
        return auth
    return _refresh_tokens(auth)


def _refresh_tokens(auth: dict) -> dict | None:
    refresh_token = auth.get("refresh_token")
    if not refresh_token:
        logger.warning("No refresh token available.")
        return None
    try:
        resp = httpx.post(
            TOKEN_URL,
            json={
                "client_id": CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        token_resp = resp.json()
        # Merge — only update fields that are present in the response
        if token_resp.get("access_token"):
            auth["access_token"] = token_resp["access_token"]
        if token_resp.get("refresh_token"):
            auth["refresh_token"] = token_resp["refresh_token"]
        if token_resp.get("id_token"):
            auth["id_token"] = token_resp["id_token"]
        auth["last_refresh"] = datetime.now(timezone.utc).isoformat()
        _save_auth(auth)
        logger.info("Tokens refreshed successfully.")
        return auth
    except Exception:
        logger.exception("Token refresh failed")
        return None
