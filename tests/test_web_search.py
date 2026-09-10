from __future__ import annotations

import asyncio
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest import mock

from Cozter.agent_tools.builtin import web_search
from Cozter.agent_tools.builtin.web_search import WebSearchTool

_HTML_BODY = """
<div class="results">
  <div class="result results_links">
    <a rel="nofollow" class="result__a"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&amp;rut=x"
       >First <b>Result</b></a>
  </div>
  <a class="result__a" href="https://direct.example.com/b">Second Result</a>
  <a class="result__a"
     href="//duckduckgo.com/y.js?ad_domain=ads.example.com&amp;ad_provider=p"
     >Sponsored Result</a>
</div>
"""

_LITE_BODY = """
<table>
  <tr><td>1.</td><td>
    <a rel="nofollow"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Flite.example%2Fone">Lite
    One</a>
  </td></tr>
  <tr><td>2.</td><td>
    <a rel="nofollow" href="https://plain.example/two">Lite Two</a>
  </td></tr>
</table>
<a href="//lite.duckduckgo.com/lite/">Home</a>
"""


class _Content:
    def __init__(self, body: bytes) -> None:
        self.body = body

    async def read(self, limit: int) -> bytes:
        chunk, self.body = self.body[:limit], self.body[limit:]
        return chunk


class _Response:
    charset = "utf-8"

    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self.content = _Content(body)


class _FakeNet:
    """Queue of per-request outcomes: exceptions or (status, body)."""

    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.urls: list[str] = []

    def __call__(self, url: str, *, timeout: int,
                 allow_redirects: bool = True):
        del timeout, allow_redirects  # unused; the tool passes them
        self.urls.append(url)
        outcome = self.outcomes.pop(0)
        return self._enter(outcome)

    @asynccontextmanager
    async def _enter(self, outcome: object):
        if isinstance(outcome, Exception):
            raise outcome
        status, body = outcome  # type: ignore[misc]
        yield _Response(status, body)


def _run_search(net: _FakeNet, **args: str) -> str:
    async def execute() -> str:
        with mock.patch.object(
            web_search, "open_http_response", net,
        ), mock.patch.object(web_search, "_RETRY_DELAY_SECONDS", 0):
            return await WebSearchTool().run(".", args)

    return asyncio.run(execute())


class WebSearchToolTests(unittest.TestCase):
    def test_parses_html_results_and_drops_ads(self) -> None:
        net = _FakeNet((200, _HTML_BODY.encode()))
        result = _run_search(net, query="cozter release notes")
        self.assertIn("1. First Result", result)
        self.assertIn("https://example.com/a", result)
        self.assertIn("2. Second Result", result)
        self.assertIn("https://direct.example.com/b", result)
        self.assertNotIn("Sponsored", result)
        self.assertNotIn("ad_domain", result)

    def test_encodes_query_and_uses_html_endpoint_first(self) -> None:
        net = _FakeNet((200, _HTML_BODY.encode()))
        _run_search(net, query="two words")
        self.assertTrue(net.urls[0].startswith("https://html.duckduckgo.com"))
        self.assertIn("q=two+words", net.urls[0])

    def test_falls_back_to_lite_endpoint(self) -> None:
        net = _FakeNet(
            OSError("connection reset"),
            # html retry answers 200 with an empty shell -> move on
            (200, b"<html><body></body></html>"),
            (200, _LITE_BODY.encode()),
        )
        result = _run_search(net, query="anything")
        self.assertIn("1. Lite One", result)
        self.assertIn("https://lite.example/one", result)
        self.assertIn("2. Lite Two", result)
        self.assertTrue(
            net.urls[2].startswith("https://lite.duckduckgo.com"),
        )
        # The lite page's own navigation links must not become results.
        self.assertNotIn("Home", result)

    def test_retries_same_endpoint_before_moving_on(self) -> None:
        net = _FakeNet(
            SimpleNamespace(status=503, content=_Content(b"")),
            (200, _HTML_BODY.encode()),
        )
        result = _run_search(net, query="anything")
        self.assertIn("First Result", result)
        self.assertEqual(net.urls[0], net.urls[1])

    def test_all_attempts_failing_reports_each_host_once(self) -> None:
        net = _FakeNet(
            OSError("down"),
            OSError("down"),
            OSError("down"),
            OSError("down"),
        )
        result = _run_search(net, query="anything")
        self.assertTrue(result.startswith("Search failed:"))
        self.assertEqual(result.count("down"), 2)  # one per endpoint

    def test_empty_pages_with_200_report_no_results(self) -> None:
        net = _FakeNet(
            (200, b"<html><body></body></html>"),
            (200, b"<html><body></body></html>"),
            (200, b""),
            (200, b""),
        )
        result = _run_search(net, query="anything")
        self.assertEqual(result, "No search results found.")

    def test_max_results_bounds_output(self) -> None:
        body = _HTML_BODY.replace(
            'href="https://direct.example.com/b"',
            'href="https://direct.example.com/c"',
        ) + '<a class="result__a" href="https://d.example/d">Third</a>'
        net = _FakeNet((200, body.encode()))
        result = _run_search(net, query="anything", max_results="2")
        self.assertIn("2. ", result)
        self.assertNotIn("3. ", result)

    def test_invalid_args(self) -> None:
        self.assertTrue(
            _run_search(_FakeNet(), query="   ").startswith("Error:"),
        )
        self.assertTrue(_run_search(_FakeNet()).startswith("Error:"))


if __name__ == "__main__":
    unittest.main()
