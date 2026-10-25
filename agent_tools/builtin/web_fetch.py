"""web_fetch: download an HTTP/HTTPS URL and return readable text."""

from __future__ import annotations

import asyncio
import urllib.parse

from .. import _http_types
from ..html_scan import first_title_contents

from ..base import (
    REDIRECT_STATUSES as _REDIRECT_STATUSES,
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
_FETCH_ATTEMPTS = 2
_FETCH_RETRY_DELAY_SECONDS = 0.4


class _FetchRefusedError(Exception):
    """A deterministic fetch refusal that a retry cannot fix.

    Covers HTTP error statuses, refused content types, refused redirect
    targets, and redirect loops. Transport failures (connection resets,
    DNS hiccups) stay ordinary exceptions so the caller can retry them
    once. Requests run until completion or cancellation.
    """


async def _fetch_following_redirects(
    session: _http_types.ClientSession,
    url: str,
) -> tuple[str, str, str, bool]:
    """GET *url* with per-hop validation; return url/type/body/cap state.

    Every redirect target is re-validated against the public-address
    policy before it is followed, and deterministic refusals raise
    :class:`_FetchRefusedError` so the caller never retries them.
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
                    raise _FetchRefusedError("Fetch failed: too many redirects")
                current_url = urllib.parse.urljoin(
                    str(response.url), location,
                )
                validation_error = _validate_public_url(current_url)
                if validation_error:
                    raise _FetchRefusedError(validation_error)
                redirects += 1
                continue

            final_url = str(response.url)
            content_type = response.headers.get("content-type", "")
            normalized_content_type = content_type.casefold()

            if response.status >= 400:
                raise _FetchRefusedError(
                    f"Fetch failed: HTTP {response.status} for {final_url}",
                )

            if not (
                normalized_content_type.startswith("text/")
                or "html" in normalized_content_type
                or "json" in normalized_content_type
                or "xml" in normalized_content_type
                or content_type == ""
            ):
                raise _FetchRefusedError(
                    f"Fetched {final_url}, but content type is "
                    f"'{content_type}', not readable text.",
                )

            body, fetch_capped = await read_bounded_text(response)
            return final_url, content_type, body, fetch_capped


class WebFetchTool(AgentTool):
    name = "web_fetch"
    # Preserve max_chars=30,000 plus the URL/title through shared dispatch.
    max_result_chars = 32 * 1024
    description = "Fetch a public URL as text."
    parameters = object_parameters(
        {
            "url": {"type": "string"},
            "max_chars": {
                "type": "integer",
                "description": "max 30000.",
            },
        },
        ["url"],
    )

    async def run(self, workspace_path: str, args: dict) -> str:
        del workspace_path
        url, error = require_nonempty_string_arg(args, "url", strip=True)
        if error:
            return error
        assert url is not None
        validation_error = _validate_public_url(url)
        if validation_error:
            return validation_error

        import aiohttp

        max_chars = coerce_int_arg(
            args.get("max_chars", 12_000),
            default=12_000,
            minimum=1_000,
            maximum=30_000,
        )

        final_url = url
        content_type = ""
        body = ""
        fetch_capped = False
        try:
            async with _open_public_http_session() as session:
                for attempt in range(_FETCH_ATTEMPTS):
                    try:
                        final_url, content_type, body, fetch_capped = (
                            await _fetch_following_redirects(session, url)
                        )
                        break
                    except _FetchRefusedError as exc:
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

        # Format is pure CPU work on capped bytes.
        return await asyncio.to_thread(
            _format_fetch_result, final_url, content_type, body, fetch_capped, max_chars,
        )

    def summarize(self, args: dict) -> str:
        return summarize_arg("web_fetch", args, "url")


def _format_fetch_result(
    final_url: str, content_type: str, body: str, fetch_capped: bool, max_chars: int,
) -> str:
    """Format one byte-capped fetch without doing I/O or changing shared state."""
    title = html_to_text(first_title_contents(body))
    is_html = "html" in content_type.casefold()
    text = html_to_text(body) if is_html else body
    text = text.strip()
    # Mark after stripping HTML: a capped raw script/style/comment can
    # otherwise swallow the footer along with its hidden contents.
    text = with_fetch_cap_marker(text, fetch_capped)
    if len(text) > max_chars:
        text = truncate_with_marker(
            text, max_chars, "say PARTIAL + remainder when coverage is unclear",
        )
    header = f"URL: {final_url}"
    if title:
        header += f"\nTitle: {title}"
    return f"{header}\n\n{text}"
