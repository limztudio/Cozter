"""Isolated runtime coverage and behavioral regressions for the tools audit."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import aiohttp

from Cozter import agent_tools
from Cozter.agent_tools.builtin.apply_patch import ApplyPatchTool
from Cozter.agent_tools.builtin import web_fetch, web_search
from Cozter.agent_tools.plugins import git_ops, git_sync, http_request, notes
from Cozter.agent_tools.plugins.git_info import GitInfoTool
from Cozter.backends_agent import claude_background_guard as guard


BUILTIN_NAMES = frozenset({
    "apply_patch", "bash", "copy_file", "delete_file", "edit_file", "glob",
    "grep", "list_dir", "list_skills", "make_dir", "move_file", "multi_edit", "read_file", "read_files", "read_skill",
    "tree", "web_fetch", "web_search", "write_file",
})
PLUGIN_NAMES = frozenset({
    "calculator", "current_time", "git_info", "git_ops", "git_sync",
    "http_request", "memory", "notes",
})


class ToolIoFailureAndShellGuardTests(unittest.TestCase):
    def test_unreadable_notes_refuse_append_and_preserve_existing_state(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            target = Path(workspace, ".cozter", "notes.md")
            target.parent.mkdir()
            target.write_text("durable old notes\n", encoding="utf-8")
            with mock.patch.object(notes, "open", create=True,
                                   side_effect=PermissionError("read denied")):
                result = asyncio.run(notes.NotesTool().run(workspace, {
                    "action": "append", "text": "new entry",
                }))
            self.assertIn("could not read notes", result)
            self.assertEqual(target.read_text(encoding="utf-8"), "durable old notes\n")

    def test_unreadable_notes_report_read_failure(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            with mock.patch.object(notes, "open", create=True,
                                   side_effect=PermissionError("read denied")):
                result = asyncio.run(notes.NotesTool().run(workspace, {"action": "read"}))
            self.assertIn("could not read notes", result)
            self.assertNotIn("Notes are empty", result)

    def test_multifile_patch_reports_every_file_after_io_failure(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "blocked").write_text("parent is a file", encoding="utf-8")
            patch = "".join(
                f"--- /dev/null\n+++ b/{name}\n@@ -0,0 +1 @@\n+{content}\n"
                for name, content in (
                    ("first.txt", "first content"),
                    ("blocked/middle.txt", "middle content"),
                    ("last.txt", "last content"),
                )
            )
            result = asyncio.run(ApplyPatchTool().run(workspace, {"patch": patch}))
            self.assertIn("first.txt: created", result)
            self.assertIn("blocked/middle.txt: failed", result)
            self.assertIn("last.txt: created", result)
            self.assertEqual(Path(workspace, "first.txt").read_text(), "first content\n")
            self.assertEqual(Path(workspace, "last.txt").read_text(), "last content\n")
            self.assertEqual(Path(workspace, "blocked").read_text(), "parent is a file")

    def test_failed_head_show_does_not_retry_with_diff_only_flag(self) -> None:
        runner = mock.AsyncMock(return_value=("", "fatal: ambiguous argument 'HEAD'", 128))
        with mock.patch("Cozter.agent_tools.plugins.git_info._git_once", runner):
            result = asyncio.run(GitInfoTool().run(".", {"action": "show"}))
        self.assertTrue(result.startswith("Error: git show:"), result)
        self.assertEqual(runner.await_count, 1)
        self.assertNotIn("--cached", runner.await_args.args[0])

    def test_multifile_patch_preserves_failed_replacement_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            first = Path(workspace, "first.txt")
            failed = Path(workspace, "failed.txt")
            last = Path(workspace, "last.txt")
            for path in (first, failed, last):
                path.write_text("original\n", encoding="utf-8")
            real_replace = os.replace
            def fail_one_replace(source: str, destination: str) -> None:
                if destination == str(failed):
                    raise PermissionError("replace denied")
                real_replace(source, destination)
            patch = "".join(
                f"--- a/{path.name}\n+++ b/{path.name}\n@@ -1 +1 @@\n-original\n+updated\n"
                for path in (first, failed, last)
            )
            with mock.patch("Cozter.agent_tools.base.os.replace", fail_one_replace):
                result = asyncio.run(ApplyPatchTool().run(workspace, {"patch": patch}))
            self.assertIn("first.txt: applied", result)
            self.assertIn("failed.txt: failed (replace denied)", result)
            self.assertIn("last.txt: applied", result)
            self.assertEqual(first.read_text(), "updated\n")
            self.assertEqual(failed.read_text(), "original\n")
            self.assertEqual(last.read_text(), "updated\n")
            self.assertEqual({path.name for path in Path(workspace).iterdir()},
                             {"first.txt", "failed.txt", "last.txt"})

    def test_guard_rejects_launchers_after_supported_wrapper_options(self) -> None:
        for command in (
            "exec -- nohup task", "exec -c -l nohup task",
            "exec -a custom-name nohup task", "command -p nohup task",
            "command -- claude --bg task", "command -p -- disown",
        ):
            with self.subTest(command=command):
                self.assertIsNotNone(guard.background_launch_mechanism({"command": command}))
        for command in ("command -v nohup", "command -V disown", "exec -a nohup printf ok"):
            with self.subTest(command=command):
                self.assertIsNone(guard.background_launch_mechanism({"command": command}))


class _FixtureHandler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        if self.path.startswith("/search"):
            body = b'<a href="https://example.org/a">Fixture result</a>'
            content_type = "text/html"
        elif self.path == "/page":
            body = b"<html><title>Fixture title</title><p>Fixture body</p></html>"
            content_type = "text/html"
        else:
            body = b'{"fixture":"GET"}'
            content_type = "application/json"
        self._respond(body, content_type)

    def _respond(self, body: bytes, content_type: str) -> None:
        # Bounded tools may hang up early; that is not a fixture error.
        with suppress(ConnectionError):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _write_request(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        self._respond(json.dumps({"method": self.command, "body": body}).encode(),
                      "application/json")

    do_POST = _write_request
    do_PUT = _write_request
    do_PATCH = _write_request
    do_DELETE = _write_request


@asynccontextmanager
async def _fixture_session():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        yield session


class IsolatedToolsRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_registered_builtin_and_public_plugin(self) -> None:
        actual_builtin = {tool.name for tool in agent_tools._TOOLS if not tool.is_plugin}
        actual_plugins = {tool.name for tool in agent_tools._TOOLS if tool.is_plugin}
        self.assertEqual(actual_builtin, BUILTIN_NAMES)
        self.assertEqual(actual_plugins, PLUGIN_NAMES)
        exercised: set[str] = set()
        exercised_git_ops: set[str] = set()
        server = ThreadingHTTPServer(("127.0.0.1", 0), _FixtureHandler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"
        try:
            with tempfile.TemporaryDirectory() as workspace:
                async def invoke(name: str, args: dict, expected: str,
                                 permission: str = "auto") -> str:
                    events: list[dict] = []
                    output = await agent_tools.execute_tool(
                        name, args, workspace, permission, events.append,
                    )
                    self.assertIn(expected, output, f"{name}: {output}")
                    self.assertEqual([event["type"] for event in events],
                                     ["tool_use", "tool_result"])
                    exercised.add(name)
                    if name == "git_ops":
                        exercised_git_ops.add(args["action"])
                    return output

                await invoke("make_dir", {"path": "src"}, "Directory ready")
                await invoke("write_file", {"path": "src/a.txt", "content": "alpha\nbeta\n"}, "Wrote")
                await invoke("read_file", {"path": "src/a.txt", "offset": 1, "limit": 1}, "beta")
                await invoke("read_files", {"files": [{"path": "src/a.txt", "limit": 1}]}, "alpha")
                await invoke("edit_file", {"path": "src/a.txt", "old_string": "alpha", "new_string": "gamma"}, "Replaced")
                await invoke("multi_edit", {"path": "src/a.txt", "edits": [{"old_string": "beta", "new_string": "delta"}]}, "Applied")
                await invoke("copy_file", {"source": "src/a.txt", "destination": "src/copy.txt"}, "Copied")
                await invoke("move_file", {"source": "src/copy.txt", "destination": "src/moved.txt"}, "Moved")
                await invoke("delete_file", {"path": "src/moved.txt"}, "Deleted")
                await invoke("apply_patch", {"patch": "--- a/src/a.txt\n+++ b/src/a.txt\n@@ -1,2 +1,2 @@\n-gamma\n+epsilon\n delta\n"}, "applied")
                await invoke("list_dir", {"path": "src"}, "a.txt")
                await invoke("tree", {}, "src/")
                await invoke("glob", {"pattern": "**/*.txt"}, "src/a.txt")
                await invoke("grep", {"pattern": "epsilon", "glob": "**/*.txt"}, "src/a.txt:1: epsilon")
                await invoke("bash", {"command": "echo audit-builtin-shell"}, "audit-builtin-shell", "full")
                await invoke("calculator", {"expression": "(2+3)*sqrt(16)"}, "20")
                await invoke("current_time", {"timezone": "UTC"}, "+00:00")
                await invoke("notes", {"action": "append", "text": "isolated fixture note"}, "Noted")
                await invoke("notes", {"action": "read"}, "isolated fixture note")
                await invoke("notes", {"action": "clear"}, "Notes cleared")
                skill_directory = Path(workspace, ".cozter", "skills")
                skill_directory.mkdir(exist_ok=True)
                (skill_directory / "runtime-audit.md").write_text(
                    "---\nname: runtime-audit\ndescription: Isolated tools audit workflow.\n---\n"
                    "Use the isolated-skill-instructions fixture.\n", encoding="utf-8",
                )
                await invoke("list_skills", {}, "runtime-audit")
                await invoke("read_skill", {"name": "runtime-audit"}, "isolated-skill-instructions")
                sessions = Path(workspace, ".cozter", "sessions")
                sessions.mkdir(parents=True, exist_ok=True)
                session = {"id": "audit-session", "name": "Fixture session", "created": "2026-10-13",
                           "summary": "fixture summary", "long_term": ["fixture memory"],
                           "messages": [{"role": "user", "content": "fixture transcript"}]}
                (sessions / "audit-session.json").write_text(json.dumps(session), encoding="utf-8")
                Path(workspace, ".cozter", "colony.json").write_text(json.dumps({"items": ["fixture colony"]}), encoding="utf-8")
                await invoke("memory", {"action": "list"}, "Fixture session")
                await invoke("memory", {"action": "search", "query": "fixture"}, "Found 4 match")
                await invoke("memory", {"action": "read", "session": "last"}, "fixture transcript")
                if shutil.which("git") is None:
                    self.fail("git is required for the complete shipped-plugin runtime audit")
                env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                       "GIT_TERMINAL_PROMPT": "0"}
                def git(*args: str) -> None:
                    subprocess.run(["git", "-C", workspace, *args], env=env,
                                   capture_output=True, check=True, timeout=10)
                def prepare_git() -> None:
                    git("init", "-q", "--initial-branch=audit-main")
                    git("config", "user.name", "Isolated Tools Audit")
                    git("config", "user.email", "audit@example.invalid")
                    git("config", "core.hooksPath", os.devnull)
                await asyncio.to_thread(prepare_git)
                with mock.patch.dict(os.environ, env):
                    await invoke("git_ops", {"action": "add", "path": "src/a.txt"}, "OK")
                    await invoke("git_ops", {"action": "commit", "message": "isolated audit fixture"}, "isolated audit fixture")
                    await invoke("git_ops", {"action": "unstage", "path": "src/a.txt"}, "OK")
                    await invoke("git_ops", {"action": "branch-create", "branch": "audit-side"}, "OK")
                    await invoke("git_ops", {"action": "branch-rename", "branch": "audit-side", "new_branch": "audit-renamed"}, "OK")
                    await invoke("git_ops", {"action": "checkout", "branch": "audit-renamed"}, "audit-renamed")
                    await invoke("git_ops", {"action": "checkout", "branch": "audit-main"}, "audit-main")
                    await invoke("git_ops", {"action": "reset", "mode": "soft", "ref": "HEAD"}, "OK")
                    await invoke("git_ops", {"action": "merge", "branch": "audit-renamed", "ff_only": True}, "up to date")
                    await invoke("git_ops", {"action": "rebase", "branch": "audit-renamed"}, "up to date")
                    # No rebase exists in the fixture: the expected refusal also
                    # exercises the public abort path without inventing a conflict.
                    await invoke("git_ops", {"action": "rebase-abort"}, "Error: git rebase-abort")
                    await invoke("git_ops", {"action": "tag-create", "tag": "audit-tag"}, "OK")
                    await invoke("git_ops", {"action": "tag-delete", "tag": "audit-tag"}, "audit-tag")
                    source = Path(workspace, "src", "a.txt")
                    source.write_text("stash fixture\n", encoding="utf-8")
                    await invoke("git_ops", {"action": "stash-push", "message": "fixture stash"}, "Saved working directory")
                    await invoke("git_ops", {"action": "stash-apply"}, "src/a.txt")
                    await invoke("git_ops", {"action": "stash-drop"}, "Dropped")
                    await invoke("git_ops", {"action": "stash-push", "message": "fixture second stash"}, "Saved working directory")
                    await invoke("git_ops", {"action": "stash-pop"}, "Dropped")
                    await invoke("git_ops", {"action": "discard", "path": "src/a.txt"}, "OK")
                    self.assertEqual(source.read_text(), "epsilon\ndelta\n")
                    Path(workspace, "disposable.txt").write_text("temporary", encoding="utf-8")
                    await invoke("git_ops", {"action": "clean", "path": "disposable.txt", "force": True}, "disposable.txt")
                    self.assertFalse(Path(workspace, "disposable.txt").exists())
                    await invoke("git_ops", {"action": "branch-delete", "branch": "audit-renamed"}, "audit-renamed")
                    for action, extra, expected in (
                        ("status", {}, "##"), ("log", {}, "isolated audit fixture"),
                        ("diff", {}, "No changes"), ("branches", {}, "*"),
                        ("tags", {}, "No tags"), ("stashes", {}, "No stashes"),
                        ("show", {}, "isolated audit fixture"),
                        ("blame", {"path": "src/a.txt", "line": 1}, "epsilon"),
                    ):
                        await invoke("git_info", {"action": action, **extra}, expected)
                    await invoke("git_sync", {"action": "remotes"}, "OK", "full")
                    with mock.patch.object(git_sync, "_run_git", mock.AsyncMock()) as sync_runner:
                        for action in ("fetch", "pull", "push"):
                            argv = git_sync.GitSyncTool()._build_argv(
                                action, {"remote": "origin", "branch": "audit-main"},
                            )
                            self.assertIsInstance(argv, list)
                            await invoke("git_sync", {"action": action}, "requires full")
                        sync_runner.assert_not_awaited()
                self.assertEqual(exercised_git_ops, set(git_ops._ACTIONS))
                # Real loopback fixture; only this test's session and address gate
                # are replaced. Production public-address policy is verified below.
                with (
                    mock.patch.object(web_fetch, "_validate_public_url", return_value=None),
                    mock.patch.object(web_fetch, "_open_public_http_session", _fixture_session),
                    mock.patch.object(web_search, "_SEARCH_ENDPOINTS", (
                        base_url + "/search-a?{qs}", base_url + "/search-b?{qs}",
                    )),
                    mock.patch.object(http_request, "validate_public_url", return_value=None),
                    mock.patch.object(http_request, "open_public_http_session", _fixture_session),
                ):
                    await invoke("web_fetch", {"url": base_url + "/page"}, "Fixture body")
                    await invoke("web_search", {"query": "fixture query"}, "Fixture result")
                    for method in ("GET", "POST", "PUT", "PATCH", "DELETE"):
                        args = {"url": base_url + "/api", "method": method}
                        if method != "GET":
                            args["body"] = '{"fixture": true}'
                        await invoke("http_request", args, method)
                await invoke("web_fetch", {"url": base_url + "/page"}, "publicly routable")
                await invoke("http_request", {"url": base_url + "/api"}, "publicly routable")
                self.assertEqual(exercised, BUILTIN_NAMES | PLUGIN_NAMES)
        finally:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            thread.join(timeout=5)

    async def test_every_public_plugin_module_entrypoint(self) -> None:
        cases = {
            "calculator": ({"expression": "6*7"}, "42"),
            "current_time": ({"timezone": "UTC"}, "+00:00"),
            "notes": ({"action": "append", "text": "CLI isolated note"}, "Noted"),
            "memory": ({"action": "list"}, "No sessions recorded"),
            "git_info": ({"action": "status"}, "##"),
            "git_ops": ({"action": "add", "path": "fixture.txt"}, "OK"),
            "git_sync": ({"action": "remotes"}, "OK"),
            "http_request": ({"url": "http://127.0.0.1/"}, "publicly routable"),
        }
        self.assertEqual(set(cases), PLUGIN_NAMES)
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "fixture.txt").write_text("CLI fixture\n", encoding="utf-8")
            package_parent = str(Path(__file__).resolve().parents[2])
            inherited_pythonpath = os.environ.get("PYTHONPATH", "")
            pythonpath = os.pathsep.join(
                entry for entry in (package_parent, inherited_pythonpath) if entry
            )
            env = {**os.environ, "PYTHONPATH": pythonpath,
                   "PYTHONIOENCODING": "utf-8", "GIT_CONFIG_NOSYSTEM": "1",
                   "GIT_CONFIG_GLOBAL": os.devnull, "GIT_TERMINAL_PROMPT": "0"}
            await asyncio.to_thread(
                subprocess.run, ["git", "-C", workspace, "init", "-q"], env=env,
                capture_output=True, check=True, timeout=10,
            )
            for module, (args, expected) in cases.items():
                with self.subTest(module=module):
                    result = await asyncio.to_thread(
                        subprocess.run,
                        [sys.executable, "-m", f"Cozter.agent_tools.plugins.{module}", json.dumps(args)],
                        cwd=workspace, env=env, capture_output=True, text=True,
                        encoding="utf-8", timeout=20,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stderr, "")
                    self.assertIn(expected, result.stdout)


if __name__ == "__main__":
    unittest.main()
