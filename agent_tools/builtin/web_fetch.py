"""web_fetch: download an HTTP/HTTPS URL and return readable text."""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.parse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
from aiohttp.abc import AbstractResolver
from aiohttp.resolver import DefaultResolver

from ..base import (
    AgentTool,
    HTTP_USER_AGENT_HEADERS,
    coerce_int_arg,
    html_to_text,
    object_parameters,
    read_bounded_text,
    require_nonempty_string_arg,
    summarize_arg,
)


_MAX_REDIRECTS = 10
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_HOST_LABEL_RE = re.compile(
    r"[A-Za-z0-9_](?:[A-Za-z0-9_-]{0,61}[A-Za-z0-9_])?"
)


class _NonPublicAddressError(OSError):
    """Raised when a request hostname resolves outside the public internet."""


class _PublicResolver(AbstractResolver):
    """Resolve DNS normally, while allowing only globally routable answers."""

    def __init__(self) -> None:
        self._resolver: AbstractResolver = DefaultResolver()

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: Any = socket.AF_UNSPEC,
    ) -> list[Any]:
        answers = await self._resolver.resolve(host, port, family=family)
        if not answers:
            raise _NonPublicAddressError(
                "URL host did not resolve to an address",
            )
        for answer in answers:
            address = answer.get("host") if isinstance(answer, dict) else None
            if not isinstance(address, str) or not _is_public_ip(address):
                raise _NonPublicAddressError(
                    "URL host resolves to a non-public address",
                )
        return answers

    async def close(self) -> None:
        await self._resolver.close()


def _is_public_ip(value: str) -> bool:
    """Return whether *value* is a globally routable, public IP address."""
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False

    # A scoped IPv6 address is never an appropriate public HTTP target.
    if isinstance(address, ipaddress.IPv6Address) and address.scope_id:
        return False

    return (
        address.is_global
        and not address.is_loopback
        and not address.is_private
        and not address.is_link_local
        and not address.is_reserved
        and not address.is_multicast
        and not address.is_unspecified
    )


def _is_valid_host(host: str) -> bool:
    """Check URL-host syntax without performing a DNS lookup."""
    if (
        not host
        or "%" in host
        or any(char.isspace() or ord(char) < 32 for char in host)
    ):
        return False

    try:
        ipaddress.ip_address(host)
    except ValueError:
        # A colon is only valid here as part of an IPv6 literal.
        if ":" in host:
            return False
        try:
            ascii_host = host.encode("idna").decode("ascii")
        except UnicodeError:
            return False
        if ascii_host.endswith("."):
            ascii_host = ascii_host[:-1]
        if not ascii_host or len(ascii_host) > 253:
            return False
        return all(
            _HOST_LABEL_RE.fullmatch(label) is not None
            for label in ascii_host.split(".")
        )
    return True


def _validate_public_url(url: str) -> str | None:
    """Return a model-facing error unless *url* is a safe public target."""
    try:
        parsed = urllib.parse.urlsplit(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return "Error: invalid URL host"

    if parsed.scheme not in ("http", "https"):
        return "Error: only http:// and https:// URLs are allowed"
    if parsed.username is not None or parsed.password is not None:
        return "Error: URL credentials are not allowed"
    if host is None or not _is_valid_host(host):
        return "Error: invalid URL host"

    # aiohttp bypasses custom resolvers for numeric hosts. Parse all numeric
    # spellings locally (including legacy decimal/octal IPv4 forms) so they
    # receive the same public-address policy before a connection is opened.
    default_port = 443 if parsed.scheme == "https" else 80
    try:
        ipaddress.ip_address(host)
    except ValueError:
        # Match aiohttp/yarl's IDNA handling before asking the socket layer
        # whether a hostname is actually a numeric spelling.
        numeric_host = host.encode("idna").decode("ascii")
    else:
        numeric_host = host
    try:
        numeric_answers = socket.getaddrinfo(
            numeric_host,
            port if port is not None else default_port,
            type=socket.SOCK_STREAM,
            flags=socket.AI_NUMERICHOST,
        )
    except (OSError, UnicodeError):
        return None

    for answer in numeric_answers:
        address = answer[4][0]
        if not isinstance(address, str) or not _is_public_ip(address):
            return "Error: URL host must be a publicly routable address"
    return None


@asynccontextmanager
async def _open_public_http_session() -> AsyncIterator[aiohttp.ClientSession]:
    """Open one HTTP session whose every DNS answer is safety-checked."""
    resolver = _PublicResolver()
    connector = aiohttp.TCPConnector(
        resolver=resolver,
        # Re-resolve each new connection, so aiohttp's cache cannot outlive
        # our validation of a hostname's current DNS answers.
        use_dns_cache=False,
    )
    try:
        async with aiohttp.ClientSession(
            headers=HTTP_USER_AGENT_HEADERS,
            connector=connector,
            trust_env=False,
        ) as session:
            yield session
    finally:
        # Older aiohttp releases do not close a caller-provided resolver.
        await resolver.close()


class WebFetchTool(AgentTool):
    name = "web_fetch"
    description = (
        "Fetch a public HTTP/HTTPS URL and return readable text. Use"
        " this after web_search to inspect a page."
    )
    parameters = object_parameters(
        {
            "url": {"type": "string"},
            "max_chars": {
                "type": "integer",
                "description": (
                    "Maximum characters to return, default 12000."
                ),
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
        validation_error = _validate_public_url(url)
        if validation_error:
            return validation_error

        max_chars = coerce_int_arg(
            args.get("max_chars") or 12_000,
            default=12_000,
            minimum=1_000,
            maximum=30_000,
        )

        final_url = url
        content_type = ""
        body = ""
        try:
            async with _open_public_http_session() as session:
                current_url = url
                redirects = 0
                while True:
                    async with session.get(
                        current_url,
                        allow_redirects=False,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as response:
                        location = response.headers.get("location")
                        if (
                            response.status in _REDIRECT_STATUSES
                            and location
                        ):
                            if redirects >= _MAX_REDIRECTS:
                                return "Fetch failed: too many redirects"
                            current_url = urllib.parse.urljoin(
                                str(response.url), location,
                            )
                            validation_error = _validate_public_url(
                                current_url,
                            )
                            if validation_error:
                                return validation_error
                            redirects += 1
                            continue

                        final_url = str(response.url)
                        content_type = response.headers.get(
                            "content-type", "",
                        )

                        if response.status >= 400:
                            return (
                                f"Fetch failed: HTTP {response.status} for"
                                f" {final_url}"
                            )

                        if not (
                            content_type.startswith("text/")
                            or "html" in content_type
                            or "json" in content_type
                            or "xml" in content_type
                            or content_type == ""
                        ):
                            return (
                                f"Fetched {final_url}, but content type is "
                                f"'{content_type}', not readable text."
                            )

                        body = await read_bounded_text(response)
                        break
        except Exception as exc:
            return f"Fetch failed: {exc}"

        title = ""
        title_match = re.search(
            r"<title[^>]*>(.*?)</title>",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if title_match:
            title = html_to_text(title_match.group(1))

        is_html = "html" in content_type.lower()
        text = html_to_text(body) if is_html else body
        text = text.strip()

        if len(text) > max_chars:
            text = (
                text[:max_chars]
                + f"\n... [truncated, {len(text)} chars total]"
            )

        header = f"URL: {final_url}"
        if title:
            header += f"\nTitle: {title}"

        return f"{header}\n\n{text}"

    def summarize(self, args: dict) -> str:
        return summarize_arg("web_fetch", args, "url")
