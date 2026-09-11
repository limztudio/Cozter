from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from Cozter.agent_tools.plugins import http_request as http_request_module
from Cozter.agent_tools.plugins.calculator import CalculatorTool
from Cozter.agent_tools.plugins.git_info import GitInfoTool
from Cozter.agent_tools.plugins.http_request import HttpRequestTool
from Cozter.agent_tools.plugins.memory import MemoryTool
from Cozter.agent_tools.plugins.notes import NotesTool


def _run(coro):
    return asyncio.run(coro)


class CalculatorToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = CalculatorTool()

    def eval(self, expression: str) -> str:
        return _run(self.tool.run(".", {"expression": expression}))

    def test_arithmetic_precedence_and_parentheses(self) -> None:
        self.assertEqual(self.eval("2 + 3 * 4"), "14")
        self.assertEqual(self.eval("(2 + 3) * 4"), "20")
        self.assertEqual(self.eval("2 ** 10"), "1024")
        self.assertEqual(self.eval("7 // 2"), "3")
        self.assertEqual(self.eval("7 % 3"), "1")
        self.assertEqual(self.eval("-5 + 2"), "-3")
        self.assertEqual(self.eval("10 / 4"), "2.5")

    def test_functions_and_constants(self) -> None:
        self.assertEqual(self.eval("sqrt(16)"), "4")
        self.assertEqual(self.eval("floor(2.7) + ceil(2.1)"), "5")
        self.assertEqual(self.eval("min(3, 1, 2) + max(3, 1, 2)"), "4")
        self.assertEqual(self.eval("round(0.12345, 3)"), "0.123")
        self.assertEqual(self.eval("factorial(5)"), "120")
        self.assertEqual(self.eval("round(pi, 5)"), "3.14159")
        self.assertEqual(self.eval("round(e, 3)"), "2.718")
        self.assertEqual(self.eval("gcd(12, 18)"), "6")

    def test_rejects_non_numeric_and_unknown_syntax(self) -> None:
        for expression in (
            "__import__('os').system('true')",
            "().__class__",
            "[1, 2]",
            "'a' + 'b'",
            "x + 1",
            "1 < 2",
            "1 if 2 else 3",
            "lambda: 1",
            "f'{1}'",
            "abs(-1) if True else 0",
            "sqrt",
        ):
            with self.subTest(expression=expression):
                self.assertTrue(
                    self.eval(expression).startswith("Error:"),
                    msg=expression,
                )

    def test_rejects_dangerous_magnitudes(self) -> None:
        self.assertIn("Error", self.eval("9 ** 9 ** 9"))
        self.assertIn("Error", self.eval("factorial(5000)"))
        self.assertIn("Error", self.eval("2 ** 999999999"))

    def test_error_cases(self) -> None:
        self.assertEqual(self.eval("1 / 0"), "Error: division by zero")
        self.assertEqual(self.eval("1 // 0"), "Error: division by zero")
        self.assertIn("Error", self.eval("sqrt(-1)"))
        self.assertIn("Error", self.eval("(2+"))
        self.assertIn("Error", self.eval("sqrt(1, 2, 3)"))

    def test_missing_expression_arg(self) -> None:
        result = _run(self.tool.run(".", {}))
        self.assertTrue(result.startswith("Error:"))


class NotesToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = NotesTool()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = self._tmp.name

    def invoke(self, action: str, text: str | None = None) -> str:
        args: dict[str, str] = {"action": action}
        if text is not None:
            args["text"] = text
        return _run(self.tool.run(self.workspace, args))

    def notes_path(self) -> str:
        return os.path.join(self.workspace, ".cozter", "notes.md")

    def test_append_creates_timestamped_entry(self) -> None:
        result = self.invoke("append", "Investigated grep; found timeout bug")
        self.assertIn("Noted", result)
        with open(self.notes_path(), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("## ", content)
        self.assertIn("Investigated grep", content)

    def test_append_requires_text(self) -> None:
        self.assertTrue(self.invoke("append").startswith("Error:"))
        self.assertTrue(self.invoke("append", "   ").startswith("Error:"))

    def test_append_requires_known_action(self) -> None:
        self.assertTrue(self.invoke("delete").startswith("Error:"))

    def test_read_reports_empty_then_entries(self) -> None:
        self.assertIn("empty", self.invoke("read"))
        self.invoke("append", "first finding")
        self.invoke("append", "second finding")
        notes = self.invoke("read")
        self.assertIn("first finding", notes)
        self.assertIn("second finding", notes)
        self.assertIn("End of notes", notes)

    def test_clear_removes_notes(self) -> None:
        self.invoke("append", "temporary")
        self.assertIn("cleared", self.invoke("clear"))
        self.assertFalse(os.path.exists(self.notes_path()))
        self.assertIn("already empty", self.invoke("clear"))

    def test_trim_keeps_newest_entries_within_budget(self) -> None:
        # Write enough entries that the 64 KiB ceiling forces a trim.
        for index in range(400):
            self.invoke("append", f"entry {index} " + "x" * 300)
        with open(self.notes_path(), "rb") as f:
            size = len(f.read())
        notes = self.invoke("read")
        self.assertLessEqual(size, 64 * 1024 + 16 * 1024)
        self.assertIn("entry 399", notes)  # newest entry always kept

    def test_tail_read_and_clipped_entry_mark_preview_and_partial(self) -> None:
        for index in range(60):
            self.invoke("append", f"tail entry {index} " + "z" * 200)
        notes = self.invoke("read")
        self.assertIn("older characters omitted", notes)
        self.assertIn("PARTIAL + remainder", notes)
        clipped = self.invoke("append", "q" * 3_000)
        self.assertIn("clipped", clipped)
        self.assertIn("PARTIAL + remainder", clipped)

    def test_oversized_legacy_file_keeps_newest_entries(self) -> None:
        # A pre-existing oversized notes file must not push the newest
        # entries out of the window a later append reads.
        os.makedirs(os.path.dirname(self.notes_path()), exist_ok=True)
        with open(self.notes_path(), "w", encoding="utf-8") as f:
            f.write("## 2000-01-01 00:00:00\n" + "old " * 30_000 + "\n")
            f.write("## 2099-01-01 00:00:00\nnewest entry\n\n")
        result = self.invoke("append", "fresh append")
        self.assertIn("Noted", result)
        with open(self.notes_path(), encoding="utf-8") as f:
            content = f.read()
        self.assertIn("newest entry", content)
        self.assertIn("fresh append", content)


class GitInfoToolTests(unittest.TestCase):
    def setUp(self) -> None:
        if not GitInfoToolTests._git_available():
            self.skipTest("git not available")
        self.tool = GitInfoTool()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = self._tmp.name
        self._git("init", "-q")
        self._git("config", "user.email", "test@example.com")
        self._git("config", "user.name", "Cozter Tests")

    @staticmethod
    def _git_available() -> bool:
        try:
            subprocess.run(
                ["git", "--version"],
                capture_output=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return False
        return True

    def _git(self, *args: str) -> None:
        subprocess.run(
            ["git", "-C", self.workspace, *args],
            capture_output=True,
            check=True,
        )

    def invoke(self, **args: object) -> str:
        return _run(self.tool.run(self.workspace, args))

    def _commit(self, filename: str, content: str, message: str) -> None:
        with open(os.path.join(self.workspace, filename), "w") as f:
            f.write(content)
        self._git("add", filename)
        self._git("commit", "-q", "-m", message)

    def test_status_shows_branch_and_changes(self) -> None:
        self._commit("a.txt", "hello\n", "initial commit")
        with open(os.path.join(self.workspace, "a.txt"), "a") as f:
            f.write("more\n")
        status = self.invoke(action="status")
        self.assertIn("##", status)  # branch line
        self.assertIn("a.txt", status)

    def test_log_lists_commit_subjects(self) -> None:
        self._commit("a.txt", "one\n", "first subject")
        self._commit("b.txt", "two\n", "second subject")
        log = self.invoke(action="log", limit=5)
        self.assertIn("first subject", log)
        self.assertIn("second subject", log)

    def test_diff_summary_and_patch(self) -> None:
        self._commit("a.txt", "one\n", "initial commit")
        with open(os.path.join(self.workspace, "a.txt"), "a") as f:
            f.write("two\n")
        summary = self.invoke(action="diff")
        self.assertIn("a.txt", summary)
        patch = self.invoke(action="diff", patch=True)
        self.assertIn("+two", patch)

    def test_diff_without_head_falls_back_to_index(self) -> None:
        # Zero-commit repository: no HEAD exists yet.
        with open(os.path.join(self.workspace, "a.txt"), "w") as f:
            f.write("staged\n")
        self._git("add", "a.txt")
        result = self.invoke(action="diff")
        self.assertFalse(result.startswith("Error:"))
        self.assertIn("a.txt", result)

    def test_non_repository_reports_git_error(self) -> None:
        empty = tempfile.mkdtemp()
        self.addCleanup(os.rmdir, empty)
        result = _run(self.tool.run(empty, {"action": "status"}))
        self.assertTrue(result.startswith("Error:"))

    def test_invalid_action_rejected(self) -> None:
        self.assertTrue(
            self.invoke(action="push").startswith("Error:"),
        )

    def test_diff_truncation_marks_preview_and_partial(self) -> None:
        self._commit("big.txt", "y" * 20_000 + "\n", "big commit")
        with open(os.path.join(self.workspace, "big.txt"), "a") as f:
            f.write("z" * 20_000 + "\n")
        result = self.invoke(action="diff", patch=True)
        self.assertIn("truncated", result)
        self.assertIn("never treat this preview as full content", result)
        self.assertIn("PARTIAL + remainder", result)

    def test_path_escaping_rejected(self) -> None:
        self.assertTrue(
            self.invoke(action="diff", path="../outside").startswith(
                "Error:",
            ),
        )


class MemoryToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = MemoryTool()
        self.ws = tempfile.mkdtemp()
        self.addCleanup(
            shutil.rmtree, self.ws, ignore_errors=True,
        )

    def invoke(self, **args: object) -> str:
        return _run(self.tool.run(self.ws, dict(args)))

    def write_session(
        self,
        session_id: str,
        *,
        name: str,
        created: str,
        messages: list[dict] | None = None,
        summary: str | None = None,
        long_term: list[str] | None = None,
    ) -> None:
        data: dict = {
            "id": session_id,
            "name": name,
            "created": created,
            "messages": messages or [],
            "long_term": long_term or [],
        }
        if summary:
            data["summary"] = summary
        sessions = os.path.join(self.ws, ".cozter", "sessions")
        os.makedirs(sessions, exist_ok=True)
        with open(
            os.path.join(sessions, f"{session_id}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(data, f)

    def write_colony(self, items: list[str]) -> None:
        cozter = os.path.join(self.ws, ".cozter")
        os.makedirs(cozter, exist_ok=True)
        with open(
            os.path.join(cozter, "colony.json"), "w", encoding="utf-8",
        ) as f:
            json.dump({"items": items, "compact_count": 0}, f)

    def test_list_orders_sessions_newest_first(self) -> None:
        self.write_session(
            "aaaaaaaa-1111", name="Old work", created="2025-05-01T09:00:00",
            messages=[{"role": "user", "content": "hi"}],
        )
        self.write_session(
            "bbbbbbbb-2222", name="Meta fix", created="2025-06-01T10:00:00",
            messages=[{"role": "user", "content": "a"}, {"role": "assistant",
                      "content": "b"}],
            summary="Fixed the host.", long_term=["Key lives on meta.ai"],
        )
        self.write_colony(["Deploys are Tuesdays."])
        result = self.invoke(action="list")
        self.assertIn("Sessions (newest first):", result)
        self.assertLess(result.index("Meta fix"), result.index("Old work"))
        self.assertIn("2 messages", result)
        self.assertIn("summary", result)
        self.assertIn("1 long-term", result)
        self.assertIn("Colony: 1 items", result)

    def test_list_empty_workspace(self) -> None:
        self.assertEqual(
            self.invoke(action="list"),
            "No sessions recorded in this workspace yet.",
        )

    def test_search_finds_message_summary_longterm_and_colony(self) -> None:
        self.write_session(
            "cccccccc-3333", name="Release", created="2025-06-02T08:00:00",
            messages=[{"role": "assistant",
                       "content": "The deploy window is Tuesday 09:00 UTC."}],
        )
        self.write_session(
            "dddddddd-4444", name="Ops", created="2025-06-01T08:00:00",
            summary="Agreed on the DEPLOY WINDOW change.",
            long_term=["deploy window owns the calendar"],
        )
        self.write_colony(["Deploy window: Tue 09:00 UTC."])
        result = self.invoke(action="search", query="deploy window")
        self.assertIn("Found 4 match(es)", result)
        self.assertIn("[Release · 2025-06-02] Assistant:", result)
        self.assertIn("[Ops · 2025-06-01] Summary:", result)
        self.assertIn("[Ops · 2025-06-01] Long-term:", result)
        self.assertIn("[Colony]", result)

    def test_search_is_case_insensitive_and_excerpts_match(self) -> None:
        self.write_session(
            "eeeeeeee-5555", name="Notes", created="2025-06-03T08:00:00",
            messages=[{"role": "user",
                       "content": "x" * 200 + " RedisCache lives on port 6379"}],
        )
        result = self.invoke(action="search", query="redis")
        self.assertIn("Found 1 match(es)", result)
        self.assertIn("…", result)  # long content is excerpted
        self.assertIn("port 6379", result)

    def test_search_limit_and_omission_hint(self) -> None:
        for offset in range(3):
            self.write_session(
                f"ffff000{offset}-6666",
                name=f"S{offset}",
                created=f"2025-06-0{offset + 1}T08:00:00",
                messages=[{"role": "user",
                           "content": "quetzal migration notes"}],
            )
        result = self.invoke(action="search", query="quetzal", limit=2)
        self.assertIn("Found 3 match(es)", result)
        self.assertEqual(result.count("quetzal migration notes"), 2)
        self.assertIn("showing the 2 newest", result)
        self.assertIn("PARTIAL + remainder", result)
        self.assertIn("never treat this preview", result)

    def test_search_requires_query(self) -> None:
        self.assertTrue(
            self.invoke(action="search").startswith("Error:"),
        )

    def test_search_no_matches(self) -> None:
        self.write_session(
            "aaaa7777-8888", name="Empty", created="2025-06-01T08:00:00",
        )
        self.assertIn(
            "No matches", self.invoke(action="search", query="zzz"),
        )

    def test_read_by_name_prefix_and_last(self) -> None:
        self.write_session(
            "11111111-9999", name="Alpha", created="2025-05-01T08:00:00",
            messages=[{"role": "user", "content": "alpha says hi"}],
        )
        self.write_session(
            "22222222-0000", name="Beta", created="2025-06-01T08:00:00",
            messages=[
                {"role": "user", "content": "beta q"},
                {"role": "assistant", "content": "beta a"},
            ],
        )
        by_name = self.invoke(action="read", session="Beta")
        self.assertIn("Session: Beta (id 22222222", by_name)
        self.assertIn("2. Assistant: beta a", by_name)
        by_prefix = self.invoke(action="read", session="11111111")
        self.assertIn("alpha says hi", by_prefix)
        by_last = self.invoke(action="read", session="last")
        self.assertIn("Session: Beta", by_last)

    def test_read_message_limit_and_line_cap(self) -> None:
        self.write_session(
            "33333333-1111",
            name="Long",
            created="2025-06-01T08:00:00",
            messages=[
                {"role": "user", "content": f"msg {i} " + "y" * 400}
                for i in range(30)
            ],
        )
        result = self.invoke(action="read", session="Long", limit=3)
        self.assertIn("30 message(s), showing last 3", result)
        self.assertIn("28.", result)
        self.assertNotIn("1. User:", result)
        self.assertIn("…", result)  # per-line cap applied
        self.assertIn("[line clipped]", result)

    def test_colony_cap_marks_preview_and_partial(self) -> None:
        self.write_colony([f"item {i}" for i in range(120)])
        result = self.invoke(action="search", query="older colony", limit=20)
        self.assertIn("older colony item(s)", result)
        self.assertIn("PARTIAL + remainder", result)

    def test_read_missing_and_ambiguous_targets(self) -> None:
        self.write_session(
            "44444444-2222", name="One", created="2025-06-01T08:00:00",
        )
        self.write_session(
            "44445555-3333", name="Two", created="2025-06-02T08:00:00",
        )
        self.assertTrue(
            self.invoke(action="read", session="nope").startswith(
                "Error: no session named",
            ),
        )
        ambiguous = self.invoke(action="read", session="4444")
        self.assertIn("matches 2 sessions", ambiguous)

    def test_read_requires_session_arg(self) -> None:
        self.assertTrue(
            self.invoke(action="read").startswith("Error:"),
        )

    def test_invalid_action_rejected(self) -> None:
        self.assertTrue(
            self.invoke(action="write").startswith("Error:"),
        )

    def test_corrupt_session_file_is_skipped(self) -> None:
        sessions = os.path.join(self.ws, ".cozter", "sessions")
        os.makedirs(sessions, exist_ok=True)
        with open(
            os.path.join(sessions, "badfile.json"), "w", encoding="utf-8",
        ) as f:
            f.write("{not json")
        self.write_session(
            "55555555-4444", name="Good", created="2025-06-01T08:00:00",
            messages=[{"role": "user", "content": "survivor"}],
        )
        result = self.invoke(action="list")
        self.assertIn("Good", result)
        self.assertNotIn("badfile", result)


class _FakeContent:
    def __init__(self, body: bytes) -> None:
        self.body = body

    async def read(self, limit: int) -> bytes:
        chunk, self.body = self.body[:limit], self.body[limit:]
        return chunk


class _FakeResponse:
    charset = "utf-8"

    def __init__(
        self,
        *,
        status: int,
        url: str,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
        reason: str = "",
    ) -> None:
        self.status = status
        self.url = url
        # aiohttp headers are case-insensitive (CIMultiDict); emulate that.
        self.headers = {
            key.lower(): value for key, value in (headers or {}).items()
        }
        self.reason = reason
        self.content = _FakeContent(body)

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def request(self, method: str, url: str, **kwargs: object):
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.responses.pop(0)


class HttpRequestToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = HttpRequestTool()

    def run_with(self, responses: list[_FakeResponse], **args: object):
        session = _FakeSession(responses)

        @asynccontextmanager
        async def fake_open():
            yield session

        with mock.patch.object(
            http_request_module, "open_public_http_session", fake_open,
        ):
            result = _run(self.tool.run(".", dict(args)))
        return result, session

    def test_get_returns_status_and_json_body(self) -> None:
        result, session = self.run_with(
            [
                _FakeResponse(
                    status=200,
                    url="https://api.example.com/v1/things",
                    headers={"Content-Type": "application/json"},
                    body=b'{"ok": true}',
                    reason="OK",
                ),
            ],
            url="https://api.example.com/v1/things",
        )
        self.assertIn("HTTP 200", result)
        self.assertIn("GET https://api.example.com/v1/things", result)
        self.assertIn('{"ok": true}', result)
        self.assertEqual(len(session.calls), 1)
        self.assertTrue(session.calls[0]["allow_redirects"] is False)

    def test_error_status_body_is_shown(self) -> None:
        result, _ = self.run_with(
            [
                _FakeResponse(
                    status=404,
                    url="https://api.example.com/v1/missing",
                    headers={"Content-Type": "application/json"},
                    body=b'{"error": "not found"}',
                    reason="Not Found",
                ),
            ],
            url="https://api.example.com/v1/missing",
        )
        self.assertIn("HTTP 404", result)
        self.assertIn('{"error": "not found"}', result)

    def test_private_host_refused_without_request(self) -> None:
        result, session = self.run_with(
            [], url="http://127.0.0.1:8080/admin",
        )
        self.assertIn("publicly routable", result)
        self.assertEqual(session.calls, [])

    def test_post_sniffs_json_content_type(self) -> None:
        result, session = self.run_with(
            [
                _FakeResponse(
                    status=201,
                    url="https://api.example.com/v1/things",
                    headers={"Content-Type": "application/json"},
                    body=b'{"id": 7}',
                ),
            ],
            url="https://api.example.com/v1/things",
            method="POST",
            body='{"name": "x"}',
        )
        self.assertIn("HTTP 201", result)
        call = session.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["data"], '{"name": "x"}')
        headers = call["headers"]
        self.assertEqual(
            headers.get("Content-Type"), "application/json",
        )

    def test_custom_headers_pass_hop_by_hop_filtered(self) -> None:
        _, session = self.run_with(
            [
                _FakeResponse(
                    status=200,
                    url="https://api.example.com/",
                    headers={"Content-Type": "text/plain"},
                    body=b"ok",
                ),
            ],
            url="https://api.example.com/",
            headers={
                "Authorization": "Bearer tok",
                "Host": "evil.example",
                "Connection": "close",
                "X-Trace": "1",
            },
        )
        headers = session.calls[0]["headers"]
        self.assertEqual(headers.get("Authorization"), "Bearer tok")
        self.assertEqual(headers.get("X-Trace"), "1")
        self.assertNotIn("Host", headers)
        self.assertNotIn("Connection", headers)

    def test_redirect_downgrades_post_to_get(self) -> None:
        result, session = self.run_with(
            [
                _FakeResponse(
                    status=303,
                    url="https://api.example.com/old",
                    headers={"Location": "https://api.example.com/new"},
                ),
                _FakeResponse(
                    status=200,
                    url="https://api.example.com/new",
                    headers={"Content-Type": "application/json"},
                    body=b'{"moved": true}',
                ),
            ],
            url="https://api.example.com/old",
            method="POST",
            body='{"a": 1}',
        )
        self.assertIn('{"moved": true}', result)
        self.assertEqual(session.calls[0]["method"], "POST")
        self.assertEqual(session.calls[1]["method"], "GET")
        self.assertNotIn("data", session.calls[1])
        self.assertEqual(session.calls[1]["url"], "https://api.example.com/new")

    def test_redirect_to_private_host_refused(self) -> None:
        result, session = self.run_with(
            [
                _FakeResponse(
                    status=302,
                    url="https://api.example.com/old",
                    headers={"Location": "http://10.0.0.5/steal"},
                ),
            ],
            url="https://api.example.com/old",
        )
        self.assertIn("publicly routable", result)
        self.assertEqual(len(session.calls), 1)  # redirect was not followed

    def test_too_many_redirects_refused(self) -> None:
        responses = [
            _FakeResponse(
                status=302,
                url=f"https://api.example.com/hop{i}",
                headers={
                    "Location": f"https://api.example.com/hop{i + 1}",
                },
            )
            for i in range(6)
        ]
        result, _ = self.run_with(
            responses, url="https://api.example.com/hop0",
        )
        self.assertIn("too many redirects", result)

    def test_binary_content_type_hidden(self) -> None:
        result, _ = self.run_with(
            [
                _FakeResponse(
                    status=200,
                    url="https://api.example.com/image.png",
                    headers={"Content-Type": "image/png"},
                    body=b"\x89PNG...",
                ),
            ],
            url="https://api.example.com/image.png",
        )
        self.assertIn("HTTP 200", result)
        self.assertIn("binary response body", result)
        self.assertNotIn("PNG", result)

    def test_transport_failure_reported(self) -> None:
        @asynccontextmanager
        async def failing_open():
            raise asyncio.TimeoutError()
            yield  # pragma: no cover - makes this an async generator

        with mock.patch.object(
            http_request_module, "open_public_http_session", failing_open,
        ):
            result = _run(
                self.tool.run(".", {"url": "https://api.example.com/"}),
            )
        self.assertTrue(result.startswith("Request failed:"))

    def test_argument_validation(self) -> None:
        result, _ = self.run_with([], url="ftp://api.example.com/x")
        self.assertIn("only http:// and https://", result)
        result, _ = self.run_with(
            [], url="https://api.example.com/", method="TRACE",
        )
        self.assertIn("'method' must be one of", result)
        result, _ = self.run_with(
            [], url="https://api.example.com/", headers="nope",
        )
        self.assertIn("'headers' must be an object", result)
        result, _ = self.run_with(
            [], url="https://api.example.com/", body="",
        )
        self.assertIn("'body' must be a non-empty string", result)
        result, _ = self.run_with([], url="")
        self.assertTrue(result.startswith("Error:"))

    def test_body_truncation_marks_preview_and_partial(self) -> None:
        result, _ = self.run_with(
            [
                _FakeResponse(
                    status=200,
                    url="https://api.example.com/big",
                    headers={"Content-Type": "application/json"},
                    body=("x" * 5_000).encode(),
                ),
            ],
            url="https://api.example.com/big",
            max_chars=500,
        )
        self.assertIn("truncated", result)
        self.assertIn("never treat this preview as full content", result)
        self.assertIn("PARTIAL + remainder", result)
        self.assertIn("raise max_chars", result)


if __name__ == "__main__":
    unittest.main()

