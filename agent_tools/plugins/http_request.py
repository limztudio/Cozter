"""Plugin: call HTTP APIs with any method against public hosts.

``web_fetch`` is GET-only and returns readable text, so HTTP backends
(llama, meta, zai, ...) had no way to interact with REST APIs - no
POST/PUT/PATCH, no request headers, no request body. This plugin adds a
bounded JSON-friendly request tool. It reuses web_fetch's public-host
safety model (syntax checks plus DNS answers restricted to globally
routable addresses, re-validated on every redirect hop) and, unlike
web_fetch, *shows* 4xx/5xx response bodies, because that is where APIs
explain their errors.
"""

from __future__ import annotations

import asyncio
import json
import urllib.parse
from typing import Any, ClassVar

import aiohttp

from ..base import (
    AgentTool,
    coerce_int_arg,
    object_parameters,
    open_public_http_session,
    read_bounded_text,
    require_nonempty_string_arg,
    summarize_arg,
    validate_public_url,
)

_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECTS = 5
_TIMEOUT_SECONDS = 30
_DEFAULT_MAX_CHARS = 3_500
# A tool-call argument, not a file upload: keep request bodies modest.
_MAX_BODY_CHARS = 100_000
# Never echoed or proxied by this tool; the transport manages these.
_HOP_BY_HOP_HEADERS = frozenset({
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "proxy-connection",
    "transfer-encoding",
    "upgrade",
    "expect",
})
_TEXTUAL_TYPE_MARKERS = (
    "json", "xml", "html", "csv", "yaml", "javascript", "text", "x-www-form",
)


class _RequestRefused(Exception):
    """A deterministic refusal (bad target, redirect policy, content)."""


def _clean_headers(value: object) -> dict[str, str] | str:
    """Return validated custom headers, or a model-facing error string."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        return "Error: 'headers' must be an object of string values"
    headers: dict[str, str] = {}
    for key, val in value.items():
        if not isinstance(key, str) or not isinstance(val, str):
            continue
        stripped = key.strip()
        if not stripped or stripped.casefold() in _HOP_BY_HOP_HEADERS:
            continue
        headers[stripped] = val
    return headers


def _looks_like_json(body: str) -> bool:
    try:
        parsed = json.loads(body)
    except ValueError:
        return False
    return isinstance(parsed, (dict, list))


def _is_textual_content_type(content_type: str) -> bool:
    normalized = content_type.casefold()
    if not normalized:
        return True
    return (
        normalized.startswith("text/")
        or any(marker in normalized for marker in _TEXTUAL_TYPE_MARKERS)
    )


async def _request_following_redirects(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    headers: dict[str, str],
    body: str | None,
) -> tuple[int, str, str, str | None]:
    """Send the request, re-validating every redirect hop.

    Returns ``(status, final_url, content_type, text_body)`` where
    ``text_body`` is ``None`` for non-textual (binary) response bodies.
    """
    current_url = url
    current_method = method
    current_body = body
    redirects = 0
    while True:
        kwargs: dict[str, Any] = {
            "allow_redirects": False,
            "timeout": aiohttp.ClientTimeout(total=_TIMEOUT_SECONDS),
        }
        if current_body is not None:
            kwargs["data"] = current_body
        async with session.request(
            current_method, current_url, headers=headers, **kwargs,
        ) as response:
            status = response.status
            location = response.headers.get("location")
            if status in _REDIRECT_STATUSES and location:
                if redirects >= _MAX_REDIRECTS:
                    raise _RequestRefused(
                        "Error: too many redirects (more than"
                        f" {_MAX_REDIRECTS})",
                    )
                next_url = urllib.parse.urljoin(
                    str(response.url), location,
                )
                validation_error = validate_public_url(next_url)
                if validation_error:
                    raise _RequestRefused(validation_error)
                # 303 always becomes GET; browsers also downgrade
                # 301/302 for anything that is not a GET.
                if status == 303 or (
                    status in (301, 302) and current_method != "GET"
                ):
                    current_method = "GET"
                    current_body = None
                current_url = next_url
                redirects += 1
                continue

            content_type = response.headers.get("content-type", "")
            if not _is_textual_content_type(content_type):
                return status, str(response.url), content_type, None
            text = await read_bounded_text(response)
            return status, str(response.url), content_type, text


class HttpRequestTool(AgentTool):
    name = "http_request"
    description = "HTTP API calls; bounded reply."
    parameters: ClassVar[dict[str, Any]] = object_parameters(
        {
            "url": {
                "type": "string",
            },
            "method": {
                "type": "string",
                "enum": list(_METHODS),
            },
            "headers": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "body": {"type": "string"},
            "max_chars": {
                "type": "integer",
                "description": "max 3500.",
            },
        },
        ["url"],
    )

    async def run(self, workspace_path: str, args: dict) -> str:
        del workspace_path  # web tools don't need the workspace
        url, error = require_nonempty_string_arg(args, "url", strip=True)
        if error:
            return error
        assert url is not None  # non-None once error is None
        validation_error = validate_public_url(url)
        if validation_error:
            return validation_error

        method = args.get("method") or "GET"
        if not isinstance(method, str) or method.upper() not in _METHODS:
            return (
                "Error: 'method' must be one of"
                f" {', '.join(_METHODS)}"
            )
        method = method.upper()

        headers = _clean_headers(args.get("headers"))
        if isinstance(headers, str):
            return headers

        body: str | None = None
        raw_body = args.get("body")
        if raw_body is not None:
            if not isinstance(raw_body, str) or not raw_body:
                return "Error: 'body' must be a non-empty string"
            if len(raw_body) > _MAX_BODY_CHARS:
                return (
                    "Error: 'body' is too long"
                    f" ({len(raw_body)} chars; max {_MAX_BODY_CHARS})"
                )
            body = raw_body
            if not any(
                key.casefold() == "content-type" for key in headers
            ):
                headers["Content-Type"] = (
                    "application/json"
                    if _looks_like_json(body)
                    else "text/plain"
                )

        max_chars = coerce_int_arg(
            args.get("max_chars") or _DEFAULT_MAX_CHARS,
            default=_DEFAULT_MAX_CHARS,
            minimum=500,
            maximum=12_000,
        )

        try:
            async with open_public_http_session() as session:
                status, final_url, content_type, text = (
                    await _request_following_redirects(
                        session, method, url, headers, body,
                    )
                )
        except _RequestRefused as exc:
            return str(exc)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            return f"Request failed: {exc}"
        except Exception as exc:  # aiohttp raises bare Exception subclasses
            return f"Request failed: {exc}"

        header = (
            f"HTTP {status}\n{method} {final_url}\n"
            f"Content-Type: {content_type or 'unknown'}"
        )
        if text is None:
            return (
                f"{header}\n\n(binary response body of this content type"
                " is not shown)"
            )
        text = text.strip()
        if len(text) > max_chars:
            text = (
                text[:max_chars]
                + f"\n... [truncated, {len(text)} chars total;"
                " raise max_chars to fetch the rest;"
                " never treat this preview as full content;"
                " say PARTIAL + remainder when coverage is unclear]"
            )
        return f"{header}\n\n{text}" if text else header

    def summarize(self, args: dict) -> str:
        return summarize_arg("http_request", args, "url")


if __name__ == "__main__":
    HttpRequestTool.run_as_script()
