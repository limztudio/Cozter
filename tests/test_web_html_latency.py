"""Malformed fetched HTML cannot trigger repeated whole-suffix regex scans."""

import asyncio
import json
import os
import subprocess
import sys
import threading
import unittest
from contextlib import asynccontextmanager
from unittest import mock

from Cozter.agent_tools.builtin import web_search
from Cozter.agent_tools.builtin import web_fetch
from Cozter.agent_tools.html_scan import first_title_contents, iter_anchor_contents


class WebHtmlScanTests(unittest.TestCase):
    def test_title_keeps_case_entities_markup_and_quoted_greater_than(self):
        body = '<TiTlE data-note="a > b">One &amp; <b>two</b></TiTlE><title>Later</title>'
        self.assertEqual(first_title_contents(body), "One &amp; <b>two</b>")
        self.assertEqual(first_title_contents("<titlecard>Wrong</titlecard>"), "")
        self.assertEqual(first_title_contents("<title>unfinished"), "")
        self.assertEqual(first_title_contents('<title data-note="unfinished'), "")

    def test_anchor_attributes_are_quote_aware_and_make_forward_progress(self):
        body = (
            '<A disabled title="not href=\'fake\' > text" HREF="https://one.example/a?x=1&amp;y=2">One</A>'
            "<a href='https://two.example/b'>Two <b>bold</b></a>"
            '<a class="navigation">No href</a>'
            '<a href=https://three.example/c>Three</a>'
        )
        self.assertEqual(list(iter_anchor_contents(body)), [
            ("https://one.example/a?x=1&amp;y=2", "One"),
            ("https://two.example/b", "Two <b>bold</b>"),
            ("https://three.example/c", "Three"),
        ])

    def test_search_still_filters_unwraps_deduplicates_and_respects_result_limit(self):
        body = (
            '<a href="https://duckduckgo.com/">Navigation</a>'
            '<a href="//duckduckgo.com/y.js?ad_domain=ads.example">Advertisement</a>'
            '<a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fone.example%2Fa&amp;rut=x">One &amp; two</a>'
            '<a href="https://one.example/a">Duplicate</a>'
            '<a href="https://two.example/b">Another</a>'
        )
        self.assertEqual(web_search._parse_results(body, 1), [
            "1. One & two\n   https://one.example/a",
        ])
        self.assertEqual(web_search._parse_results(body, 5), [
            "1. One & two\n   https://one.example/a", "2. Another\n   https://two.example/b",
        ])

    def test_anchor_scan_cap_still_marks_uninspected_results(self):
        body = '<a href="https://duckduckgo.com/">Navigation</a>' * 201
        body += '<a href="https://later.example/">Uninspected result</a>'
        result = "\n".join(web_search._parse_results(body, 5))
        self.assertIn("parser scan capped at 200 anchors", result)
        self.assertIn("PARTIAL + remainder", result)
        self.assertNotIn("later.example", result)

    def test_five_megabyte_malformed_documents_finish_in_killable_process(self):
        # A child process bounds the regression test itself if a future change
        # reintroduces a regex that holds the GIL for minutes on these inputs.
        # Ten seconds includes Python/import startup and permits slow hosts;
        # this is intentionally not a subsecond wall-clock performance test.
        script = """
import json
from Cozter.agent_tools.html_scan import first_title_contents
from Cozter.agent_tools.builtin.web_search import _parse_results
size = 5 * 1024 * 1024
cases = [
    ('title_open', '<title ', first_title_contents),
    ('title_body', '<title>', first_title_contents),
    ('anchor_open', '<a ', lambda body: _parse_results(body, 5)),
    ('anchor_body', '<a href="https://one.example/">', lambda body: _parse_results(body, 5)),
    ('anchor_no_href', '<a class="empty">', lambda body: _parse_results(body, 5)),
]
checked = []
for name, unit, parse in cases:
    body = unit * (size // len(unit))
    result = parse(body)
    assert not result, (name, result)
    checked.append(name)
print(json.dumps(checked))
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True,
            env=dict(os.environ), timeout=10, check=True,
        )
        self.assertEqual(json.loads(result.stdout), [
            "title_open", "title_body", "anchor_open", "anchor_body", "anchor_no_href",
        ])


class WebHtmlOffloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_format_allows_loop_progress_and_keeps_network_closed_on_cancel(self):
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        closed = asyncio.Event()

        @asynccontextmanager
        async def session():
            try:
                yield object()
            finally:
                closed.set()

        def format_result(*_args):
            started.set()
            try:
                if not release.wait(5):
                    raise AssertionError("event loop did not release the CPU worker")
                return "formatted"
            finally:
                finished.set()

        with (
            mock.patch.object(web_fetch, "_open_public_http_session", session),
            mock.patch.object(web_fetch, "_fetch_following_redirects", mock.AsyncMock(return_value=(
                "https://public.example/", "text/html", "<p>bounded body</p>", False,
            ))),
            mock.patch.object(web_fetch, "_format_fetch_result", format_result),
        ):
            task = asyncio.create_task(web_fetch.WebFetchTool().run("", {"url": "https://public.example/"}))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 5))
                # Only the event-loop coroutine can reach this cancellation
                # while the CPU worker waits for its release barrier.
                self.assertTrue(closed.is_set())
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
                self.assertTrue(await asyncio.to_thread(finished.wait, 5))
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_search_parse_allows_loop_progress_and_closes_both_endpoint_contexts(self):
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        closed = set()

        class Response:
            status = 200

        @asynccontextmanager
        async def response(url):
            try:
                if "lite.duckduckgo.com" in url:
                    await asyncio.Event().wait()
                yield Response()
            finally:
                closed.add(url)

        def parse_results(*_args):
            started.set()
            try:
                if not release.wait(5):
                    raise AssertionError("event loop did not release the CPU worker")
                return ["1. Result\n   https://example.com/"]
            finally:
                finished.set()

        with (
            mock.patch.object(web_search, "open_http_response", response),
            mock.patch.object(web_search, "read_bounded_text", mock.AsyncMock(return_value=("bounded", False))),
            mock.patch.object(web_search, "_parse_results", parse_results),
        ):
            task = asyncio.create_task(web_search.WebSearchTool().run("", {"query": "offline fixture"}))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 5))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(len(closed), 2)
            finally:
                release.set()
                self.assertTrue(await asyncio.to_thread(finished.wait, 5))
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
