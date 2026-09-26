"""web_fetch: download an HTTP/HTTPS URL and return readable text."""

from __future__ import annotations

import asyncio
import re
import urllib.parse

import aiohttp

from ..base import (
    AgentTool,
    coerce_int_arg,
    html_to_text,
    object_parameters,
    open_public_http_session as _open_public_http_session,
    read_bounded_text,
    require_nonempty_string_arg,
    summarize_arg,
    truncate_with_marker,
    validate_public_url as _validate_public_url,
    with_fetch_cap_marker,
)


_MAX_REDIRECTS = 10
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_FETCH_ATTEMPTS = 2
_FETCH_RETRY_DELAY_SECONDS = 0.4
# Precompiled once: WebFetchTool.run executes per fetch, so an inline
# re.search would recompile the title pattern on every fetched page.
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


class _FetchRefused(Exception):
    """A deterministic fetch refusal that a retry cannot fix.

    Covers HTTP error statuses, refused content types, refused redirect
    targets, and redirect loops. Transport failures (connection resets,
    DNS hiccups) stay ordinary exceptions so the caller can retry them
    once. Real-work cap: requests run up to 3600s or until cancelled.
    """


async def _fetch_following_redirects(
    session: aiohttp.ClientSession,
    url: str,
) -> tuple[str, str, str]:
    """GET *url* with per-hop validation; return url/content-type/body.

    Every redirect target is re-validated against the public-address
    policy before it is followed, and deterministic refusals raise
    :class:`_FetchRefused` so the caller never retries them.
    """
    current_url = url
    redirects = 0
    while True:
        async with session.get(
            current_url,
            allow_redirects=False,
        ) as response:
            location = response.headers.get("location")
            if response.status in _REDIRECT_STATUSES and location:
                if redirects >= _MAX_REDIRECTS:
                    raise _FetchRefused("Fetch failed: too many redirects")
                current_url = urllib.parse.urljoin(
                    str(response.url), location,
                )
                validation_error = _validate_public_url(current_url)
                if validation_error:
                    raise _FetchRefused(validation_error)
                redirects += 1
                continue

            final_url = str(response.url)
            content_type = response.headers.get("content-type", "")
            normalized_content_type = content_type.casefold()

            if response.status >= 400:
                raise _FetchRefused(
                    f"Fetch failed: HTTP {response.status} for {final_url}",
                )

            if not (
                normalized_content_type.startswith("text/")
                or "html" in normalized_content_type
                or "json" in normalized_content_type
                or "xml" in normalized_content_type
                or content_type == ""
            ):
                raise _FetchRefused(
                    f"Fetched {final_url}, but content type is "
                    f"'{content_type}', not readable text.",
                )

            body, _fetch_capped = await read_bounded_text(response)
            return final_url, content_type, with_fetch_cap_marker(
                body, _fetch_capped,
            )


class WebFetchTool(AgentTool):
    name = "web_fetch"
    description = "Fetch a public URL as text."
    parameters = object_parameters(
        {
            "url": {"type": "string"},
            "max_chars": {
                "type": "integer",
                "description": "max 12000.",
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
            args.get("max_chars", 12_000),
            default=12_000,
            minimum=1_000,
            maximum=30_000,
        )

        # One retry covers transient transport failures (connection reset,
        # resolver hiccup); every other outcome is final. Real-work cap
        # of 3600s; cancel still stops instantly. One session serves both
        # attempts so the retry reuses the connection pool instead of
        # paying for a fresh connector + DNS validation pass.
        final_url = url
        content_type = ""
        body = ""
        try:
            async with _open_public_http_session() as session:
                for attempt in range(_FETCH_ATTEMPTS):
                    try:
                        final_url, content_type, body = (
                            await _fetch_following_redirects(session, url)
                        )
                        break
                    except _FetchRefused as exc:
                        return str(exc)
                    except aiohttp.ClientError as exc:
                        if attempt + 1 >= _FETCH_ATTEMPTS:
                            return f"Fetch failed: {exc}"
                        await asyncio.sleep(_FETCH_RETRY_DELAY_SECONDS)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        return f"Fetch failed: {exc}"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return f"Fetch failed: {exc}"

        title = ""
        title_match = _TITLE_RE.search(body)
        if title_match:
            title = html_to_text(title_match.group(1))

        is_html = "html" in content_type.casefold()
        text = html_to_text(body) if is_html else body
        text = text.strip()

        if len(text) > max_chars:
            text = truncate_with_marker(
                text, max_chars, "say PARTIAL + remainder when coverage is unclear",
            )

        header = f"URL: {final_url}"
        if title:
            header += f"\nTitle: {title}"

        return f"{header}\n\n{text}"

    def summarize(self, args: dict) -> str:
        return summarize_arg("web_fetch", args, "url")
