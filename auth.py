"""OpenAI ChatGPT OAuth authentication via browser login."""

import base64
import hashlib
import json
import logging
import os
import secrets
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, urlparse, parse_qs

import httpx

logger = logging.getLogger(__name__)

AUTH_BASE = "https://auth.openai.com"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
TOKEN_URL = f"{AUTH_BASE}/oauth/token"
AUTHORIZE_URL = f"{AUTH_BASE}/oauth/authorize"
CALLBACK_PORT = 1455
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}/auth/callback"
SCOPES = "openid profile email offline_access"

CONFIG_DIR = os.path.join(os.path.dirname(__file__), ".config")
SECRET_DIR = os.path.join(CONFIG_DIR, "secret")
AUTH_PATH = os.path.join(SECRET_DIR, "auth.json")

TOKEN_REFRESH_DAYS = 8


# ---------------------------------------------------------------------------
# PKCE
# ---------------------------------------------------------------------------

def _generate_pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


# ---------------------------------------------------------------------------
# Token storage  (.config/secret/)
# ---------------------------------------------------------------------------

def _load_auth() -> dict | None:
    if os.path.exists(AUTH_PATH):
        with open(AUTH_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_auth(data: dict) -> None:
    os.makedirs(SECRET_DIR, exist_ok=True)
    with open(AUTH_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def is_logged_in() -> bool:
    data = _load_auth()
    return data is not None and bool(data.get("access_token"))


def get_tokens() -> dict | None:
    return _load_auth()


def clear_auth() -> None:
    if os.path.exists(AUTH_PATH):
        os.remove(AUTH_PATH)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def _decode_jwt_payload(token: str) -> dict:
    payload = token.split(".")[1]
    padding = 4 - len(payload) % 4
    if padding != 4:
        payload += "=" * padding
    return json.loads(base64.urlsafe_b64decode(payload))


def _is_token_expired(data: dict) -> bool:
    try:
        claims = _decode_jwt_payload(data["access_token"])
        return claims.get("exp", 0) < time.time()
    except Exception:
        return True


def _is_token_stale(data: dict) -> bool:
    if _is_token_expired(data):
        return True
    last = data.get("last_refresh")
    if not last:
        return True
    try:
        dt = datetime.fromisoformat(last)
        age = (datetime.now(timezone.utc) - dt).days
        return age >= TOKEN_REFRESH_DAYS
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Browser OAuth flow (PKCE authorization code)
# ---------------------------------------------------------------------------

class _CallbackHandler(BaseHTTPRequestHandler):
    """Handles the OAuth redirect callback on localhost."""

    auth_code: str | None = None
    received_state: str | None = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/auth/callback":
            self.send_response(404)
            self.end_headers()
            return

        params = parse_qs(parsed.query)
        _CallbackHandler.auth_code = params.get("code", [None])[0]
        _CallbackHandler.received_state = params.get("state", [None])[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(
            b"<html><body><h2>Login successful!</h2>"
            b"<p>You can close this tab and return to the terminal.</p>"
            b"</body></html>"
        )

    def log_message(self, format, *args):
        # Silence request logging
        pass


def browser_login() -> dict:
    """
    Open the default browser for OpenAI OAuth login.
    Starts a local server to receive the callback.
    Returns the saved auth dict.
    """
    verifier, challenge = _generate_pkce()
    state = secrets.token_urlsafe(32)

    # Reset handler state
    _CallbackHandler.auth_code = None
    _CallbackHandler.received_state = None

    # Start local callback server
    server = HTTPServer(("127.0.0.1", CALLBACK_PORT), _CallbackHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Build authorize URL and open browser
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "originator": "codex_cli_rs",
    }
    url = f"{AUTHORIZE_URL}?{urlencode(params)}"

    print(f"\nOpening browser for OpenAI login...")
    print(f"If the browser doesn't open, go to:\n{url}\n")
    webbrowser.open(url)

    # Wait for callback
    print("Waiting for login callback...")
    deadline = time.time() + 300  # 5 min timeout
    while time.time() < deadline:
        if _CallbackHandler.auth_code is not None:
            break
        time.sleep(0.5)

    server.shutdown()

    if _CallbackHandler.auth_code is None:
        raise TimeoutError("Login timed out — no callback received within 5 minutes.")

    if _CallbackHandler.received_state != state:
        raise RuntimeError("OAuth state mismatch — possible CSRF. Try again.")

    # Exchange code for tokens
    token_resp = _exchange_auth_code(_CallbackHandler.auth_code, verifier)
    return save_tokens(token_resp)


def _exchange_auth_code(code: str, code_verifier: str) -> dict:
    resp = httpx.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": CLIENT_ID,
            "code_verifier": code_verifier,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    resp.raise_for_status()
    return resp.json()


def save_tokens(token_resp: dict) -> dict:
    id_token = token_resp.get("id_token", "")
    claims = {}
    if id_token:
        try:
            claims = _decode_jwt_payload(id_token)
        except Exception:
            pass

    auth_claims = claims.get("https://api.openai.com/auth", {})
    data = {
        "access_token": token_resp.get("access_token", ""),
        "refresh_token": token_resp.get("refresh_token", ""),
        "id_token": id_token,
        "account_id": auth_claims.get("account_id", ""),
        "email": claims.get("email", ""),
        "plan": auth_claims.get("chatgpt_plan_type", ""),
        "last_refresh": datetime.now(timezone.utc).isoformat(),
    }
    _save_auth(data)
    return data


# ---------------------------------------------------------------------------
# API key exchange (id_token -> openai-api-key, like Codex CLI)
# ---------------------------------------------------------------------------

def get_api_key() -> str | None:
    """Return a usable OpenAI API key, exchanging the id_token if needed."""
    data = _load_auth()
    if not data:
        return None
    # Use cached key if present
    if data.get("api_key"):
        return data["api_key"]
    # Exchange id_token for API key
    id_token = data.get("id_token")
    if not id_token:
        return None
    try:
        resp = httpx.post(
            TOKEN_URL,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                "client_id": CLIENT_ID,
                "requested_token": "openai-api-key",
                "subject_token": id_token,
                "subject_token_type": "urn:ietf:params:oauth:token-type:id_token",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        api_key = resp.json().get("access_token", "")
        if api_key:
            data["api_key"] = api_key
            _save_auth(data)
            logger.info("API key obtained via token exchange.")
        return api_key or None
    except Exception:
        logger.exception("API key exchange failed")
        return None


# ---------------------------------------------------------------------------
# Token refresh
# ---------------------------------------------------------------------------

def refresh_if_needed() -> dict | None:
    data = _load_auth()
    if not data:
        return None
    if not _is_token_stale(data):
        return data
    return _refresh_tokens(data)


def _refresh_tokens(data: dict) -> dict | None:
    refresh_token = data.get("refresh_token")
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
        if token_resp.get("access_token"):
            data["access_token"] = token_resp["access_token"]
        if token_resp.get("refresh_token"):
            data["refresh_token"] = token_resp["refresh_token"]
        if token_resp.get("id_token"):
            data["id_token"] = token_resp["id_token"]
        data["last_refresh"] = datetime.now(timezone.utc).isoformat()
        data.pop("api_key", None)  # invalidate cached API key
        _save_auth(data)
        logger.info("Tokens refreshed successfully.")
        return data
    except Exception:
        logger.exception("Token refresh failed")
        return None
