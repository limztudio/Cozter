import asyncio
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from Cozter import agent_tools
from Cozter.agent_tools.base import (
    _path_matches_glob,
    apply_string_replacement,
    coerce_int_arg,
    read_bounded_text,
    validate_replacement_strings,
)
from Cozter.agent_tools.builtin.apply_patch import ApplyPatchTool
from Cozter.agent_tools.builtin.bash import BashTool
from Cozter.agent_tools.builtin.edit_file import EditFileTool
from Cozter.agent_tools.builtin.glob import GlobTool
from Cozter.agent_tools.builtin.grep import GrepTool
from Cozter.agent_tools.builtin.multi_edit import MultiEditTool
from Cozter.agent_tools.builtin.tree import TreeTool


class AgentToolHelperTests(unittest.TestCase):
    def test_coerce_int_arg_defaults_and_clamps(self) -> None:
        self.assertEqual(
            coerce_int_arg("bad", default=10, minimum=1, maximum=20),
            10,
        )
        self.assertEqual(
            coerce_int_arg("-5", default=10, minimum=1, maximum=20),
            1,
        )
        self.assertEqual(
            coerce_int_arg("99", default=10, minimum=1, maximum=20),
            20,
        )

    def test_replacement_helpers_validate_and_apply(self) -> None:
        self.assertEqual(
            validate_replacement_strings("", "new"),
            "'old_string' must not be empty",
        )
        self.assertEqual(
            validate_replacement_strings("same", "same"),
            "old_string and new_string are identical; nothing to change",
        )
        self.assertEqual(
            apply_string_replacement(
                "a b a", "a", "x", replace_all=False,
            ),
            ("a b a", 2, 0),
        )
        self.assertEqual(
            apply_string_replacement(
                "a b a", "a", "x", replace_all=True,
            ),
            ("x b x", 2, 2),
        )

    def test_read_bounded_text_accumulates_partial_reads(self) -> None:
        class ChunkedContent:
            def __init__(self, chunks: list[bytes]) -> None:
                self.chunks = chunks

            async def read(self, limit: int) -> bytes:
                if not self.chunks:
                    return b""
                chunk = self.chunks.pop(0)
                self.chunks[:0] = [chunk[limit:]] if len(chunk) > limit else []
                return chunk[:limit]

        class Response:
            charset = "utf-8"

            def __init__(self, chunks: list[bytes]) -> None:
                self.content = ChunkedContent(chunks)

        async def run() -> None:
            response = Response([b"first-", b"second-", b"third"])
            self.assertEqual(
                await read_bounded_text(response),  # type: ignore[arg-type]
                "first-second-third",
            )

        asyncio.run(run())

    def test_read_bounded_text_stops_at_byte_cap(self) -> None:
        class Content:
            async def read(self, limit: int) -> bytes:
                return b"abcdefgh"[:limit]

        class Response:
            content = Content()
            charset = "utf-8"

        async def run() -> None:
            with mock.patch("Cozter.agent_tools.base._MAX_FETCH_BYTES", 5):
                self.assertEqual(
                    await read_bounded_text(Response()),  # type: ignore[arg-type]
                    "abcde",
                )

        asyncio.run(run())

    def test_path_glob_handles_many_repeated_globstars(self) -> None:
        """Repeated ``**`` patterns must not cause exponential matching."""
        path = "/".join([*(f"part{i}" for i in range(40)), "target.py"])
        pattern = "/".join([*("**" for _ in range(40)), "target.py"])

        self.assertTrue(_path_matches_glob(path, pattern))
        self.assertFalse(_path_matches_glob(path, pattern[:-2] + "txt"))


class BuiltinEditToolTests(unittest.TestCase):
    def test_edit_file_uses_shared_replacement_logic(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "note.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("alpha beta")

                result = await EditFileTool().run(
                    tmp,
                    {
                        "path": "note.txt",
                        "old_string": "beta",
                        "new_string": "gamma",
                    },
                )

                self.assertEqual(result, "Replaced 1 occurrence in note.txt")
                with open(path, encoding="utf-8") as f:
                    self.assertEqual(f.read(), "alpha gamma")

        asyncio.run(run())


class BashToolTests(unittest.TestCase):
    @staticmethod
    def _process_is_running(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        if os.name == "posix":
            try:
                with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
                    stat = f.read()
            except OSError:
                return True
            parts = stat.split()
            if len(parts) > 2 and parts[2] == "Z":
                return False
        return True

    @unittest.skipIf(os.name == "nt", "POSIX process group behavior")
    def test_timeout_kills_child_process_group(self) -> None:
        async def run() -> tuple[str, int]:
            with tempfile.TemporaryDirectory() as tmp:
                pid_path = os.path.join(tmp, "child.pid")
                result = await BashTool().run(
                    tmp,
                    {
                        "command": "sleep 30 & echo $! > child.pid; wait",
                        "timeout": 1,
                    },
                )
                with open(pid_path, encoding="utf-8") as f:
                    child_pid = int(f.read().strip())
                return result, child_pid

        result, child_pid = asyncio.run(run())
        self.assertIn("timed out after 1s", result)

        deadline = time.monotonic() + 2
        while self._process_is_running(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)

        self.assertFalse(
            self._process_is_running(child_pid),
            f"child process {child_pid} survived bash tool timeout",
        )


class PluginScriptTests(unittest.TestCase):
    def test_plugin_module_invocation_does_not_preload_target(self) -> None:
        package_parent = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "Cozter.agent_tools.plugins.current_time",
                '{"timezone":"UTC"}',
            ],
            cwd=package_parent,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("RuntimeWarning", proc.stderr)
        self.assertIn("+00:00", proc.stdout.strip())


class DiscoveryToolTests(unittest.TestCase):
    def test_glob_skips_generated_dirs_unless_explicit(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                os.makedirs(os.path.join(tmp, ".venv", "pkg"))
                with open(os.path.join(tmp, "app.py"), "w", encoding="utf-8") as f:
                    f.write("print('app')\n")
                with open(
                    os.path.join(tmp, ".venv", "pkg", "hidden.py"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write("print('hidden')\n")

                result = await GlobTool().run(tmp, {"pattern": "**/*.py"})
                self.assertIn("app.py", result.splitlines())
                self.assertNotIn(".venv/pkg/hidden.py", result)

                explicit = await GlobTool().run(
                    tmp, {"pattern": ".venv/**/*.py"},
                )
                self.assertIn(".venv/pkg/hidden.py", explicit.splitlines())

        asyncio.run(run())

    def test_grep_skips_generated_dirs_unless_path_targets_them(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                os.makedirs(os.path.join(tmp, ".cozter", "sessions"))
                with open(os.path.join(tmp, "app.py"), "w", encoding="utf-8") as f:
                    f.write("needle in app\n")
                with open(
                    os.path.join(tmp, ".cozter", "sessions", "state.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write("needle in state\n")

                result = await GrepTool().run(tmp, {"pattern": "needle"})
                self.assertIn("app.py:1: needle in app", result)
                self.assertNotIn(".cozter/sessions/state.json", result)

                explicit = await GrepTool().run(
                    tmp, {"pattern": "needle", "path": ".cozter"},
                )
                self.assertIn(
                    ".cozter/sessions/state.json:1: needle in state",
                    explicit,
                )

        asyncio.run(run())

    def test_multi_edit_rejects_ambiguous_edit_without_partial_write(
        self,
    ) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "note.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("alpha beta beta")

                result = await MultiEditTool().run(
                    tmp,
                    {
                        "path": "note.txt",
                        "edits": [
                            {
                                "old_string": "alpha",
                                "new_string": "omega",
                            },
                            {
                                "old_string": "beta",
                                "new_string": "gamma",
                            },
                        ],
                    },
                )

                self.assertEqual(
                    result,
                    "Edit 1: old_string appears 2 times;"
                    " include more context or set replace_all=true.",
                )
                with open(path, encoding="utf-8") as f:
                    self.assertEqual(f.read(), "alpha beta beta")

        asyncio.run(run())


class ConfirmPermissionGateTests(unittest.TestCase):
    """confirm exposes read-only tools only; execute_tool is the backstop."""

    def _execute(
        self, name: str, args: dict, approval: str, ws: str,
    ) -> str:
        events: list[dict] = []
        return asyncio.run(
            agent_tools.execute_tool(name, args, ws, approval, events.append)
        )

    def test_confirm_blocks_state_changing_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._execute(
                "write_file", {"path": "x.txt", "content": "hi"},
                "confirm", tmp,
            )
            self.assertTrue(result.startswith("Blocked"), result)
            self.assertFalse(os.path.exists(os.path.join(tmp, "x.txt")))

    def test_confirm_allows_read_only_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._execute("list_dir", {"path": "."}, "confirm", tmp)
            self.assertFalse(result.startswith("Blocked"), result)

    def test_auto_allows_state_changing_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._execute(
                "write_file", {"path": "x.txt", "content": "hi"},
                "auto", tmp,
            )
            self.assertFalse(result.startswith("Blocked"), result)
            self.assertTrue(os.path.exists(os.path.join(tmp, "x.txt")))

    def test_read_only_schema_excludes_mutating_tools(self) -> None:
        names = {
            e["function"]["name"]
            for e in agent_tools.READ_ONLY_TOOL_SCHEMA
        }
        self.assertTrue(names.issubset(agent_tools.READ_ONLY_TOOL_NAMES))
        self.assertIn("read_file", names)
        self.assertNotIn("write_file", names)
        self.assertNotIn("bash", names)


class ExecuteToolTimeoutTests(unittest.TestCase):
    def test_execute_tool_enforces_configured_timeout(self) -> None:
        class SlowTool:
            file_action = None

            async def run(self, workspace_path: str, args: dict) -> str:
                del workspace_path, args
                await asyncio.sleep(60)
                return "finished"

        async def run() -> tuple[str, list[dict]]:
            events: list[dict] = []
            return (
                await agent_tools.execute_tool(
                    "slow_test", {}, "/tmp", "auto", events.append,
                ),
                events,
            )

        original_tools = agent_tools._BY_NAME
        original_timeout = agent_tools.tool_timeout
        agent_tools._BY_NAME = {**original_tools, "slow_test": SlowTool()}
        agent_tools.tool_timeout = lambda: 0.01
        try:
            result, events = asyncio.run(run())
        finally:
            agent_tools._BY_NAME = original_tools
            agent_tools.tool_timeout = original_timeout

        self.assertIn("Tool slow_test timed out after 0.01s", result)
        self.assertEqual(events[0]["type"], "tool_use")
        self.assertEqual(events[-1]["type"], "tool_result")
        self.assertEqual(events[-1]["output"], result)


class ParseOpenAICallTests(unittest.TestCase):
    def test_string_arguments(self) -> None:
        name, args = agent_tools.parse_openai_call(
            {"function": {"name": "read_file",
                          "arguments": '{"path": "a.py"}'}},
        )
        self.assertEqual(name, "read_file")
        self.assertEqual(args, {"path": "a.py"})

    def test_dict_arguments_are_accepted(self) -> None:
        # GLM / Z.ai and some local runtimes return an already-parsed
        # object instead of a JSON string; it must not crash.
        _, args = agent_tools.parse_openai_call(
            {"function": {"name": "x", "arguments": {"path": "b.py"}}},
        )
        self.assertEqual(args, {"path": "b.py"})

    def test_missing_or_bad_arguments_yield_empty(self) -> None:
        for raw in (None, "", "not json", "[1, 2]"):
            _, args = agent_tools.parse_openai_call(
                {"function": {"name": "x", "arguments": raw}},
            )
            self.assertEqual(args, {}, f"raw={raw!r}")


class TreeToolTests(unittest.TestCase):
    @staticmethod
    def _touch(path: str) -> None:
        with open(path, "w", encoding="utf-8"):
            pass

    def test_tree_shows_structure_and_skips_noise(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                os.makedirs(os.path.join(tmp, "src", "pkg"))
                os.makedirs(os.path.join(tmp, ".git", "objects"))
                self._touch(os.path.join(tmp, "README.md"))
                self._touch(os.path.join(tmp, "src", "main.py"))
                self._touch(os.path.join(tmp, "src", "pkg", "util.py"))
                out = await TreeTool().run(tmp, {})
                for marker in ("src/", "main.py", "pkg/", "util.py",
                               "README.md"):
                    self.assertIn(marker, out)
                self.assertNotIn(".git", out)  # noise dir skipped

        asyncio.run(run())

    def test_tree_depth_limits_recursion(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                os.makedirs(os.path.join(tmp, "a", "b", "c"))
                self._touch(os.path.join(tmp, "a", "b", "c", "deep.py"))
                out = await TreeTool().run(tmp, {"depth": 1})
                self.assertIn("a/", out)
                self.assertNotIn("deep.py", out)  # beyond depth 1

        asyncio.run(run())


class ApplyPatchToolTests(unittest.TestCase):
    @staticmethod
    def _write(path: str, text: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def _run(self, ws: str, patch: str) -> str:
        return asyncio.run(ApplyPatchTool().run(ws, {"patch": patch}))

    def test_modify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "foo.txt")
            self._write(p, "line1\nline2\nline3\n")
            out = self._run(tmp, (
                "--- a/foo.txt\n+++ b/foo.txt\n@@ -1,3 +1,3 @@\n"
                " line1\n-line2\n+line2-changed\n line3\n"
            ))
            self.assertIn("applied", out)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), "line1\nline2-changed\nline3\n")

    def test_create(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = self._run(tmp, (
                "--- /dev/null\n+++ b/new/dir/created.txt\n"
                "@@ -0,0 +1,2 @@\n+hello\n+world\n"
            ))
            self.assertIn("created", out)
            created = os.path.join(tmp, "new", "dir", "created.txt")
            with open(created, encoding="utf-8") as f:
                self.assertEqual(f.read(), "hello\nworld\n")

    def test_create_does_not_overwrite_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "existing.txt")
            self._write(p, "keep me\n")
            out = self._run(tmp, (
                "--- /dev/null\n+++ b/existing.txt\n"
                "@@ -0,0 +1 @@\n+replacement\n"
            ))
            self.assertIn("already exists", out)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), "keep me\n")

    def test_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "gone.txt")
            self._write(p, "bye\n")
            out = self._run(
                tmp, "--- a/gone.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-bye\n",
            )
            self.assertIn("deleted", out)
            self.assertFalse(os.path.exists(p))

    def test_delete_requires_matching_hunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "keep.txt")
            self._write(p, "actual content\n")
            out = self._run(tmp, (
                "--- a/keep.txt\n+++ /dev/null\n"
                "@@ -1 +0,0 @@\n-expected content\n"
            ))
            self.assertIn("did not apply", out)
            self.assertIn("not deleted", out)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), "actual content\n")

    def test_multi_hunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "m.txt")
            self._write(p, "\n".join(f"L{i}" for i in range(1, 11)) + "\n")
            out = self._run(tmp, (
                "--- a/m.txt\n+++ b/m.txt\n"
                "@@ -1,2 +1,2 @@\n L1\n-L2\n+L2x\n"
                "@@ -9,2 +9,2 @@\n L9\n-L10\n+L10x\n"
            ))
            self.assertIn("2 hunk", out)
            with open(p, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("L2x", content)
            self.assertIn("L10x", content)
            self.assertNotIn("\nL2\n", content)

    def test_context_not_found_leaves_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "c.txt")
            self._write(p, "alpha\nbeta\n")
            out = self._run(tmp, (
                "--- a/c.txt\n+++ b/c.txt\n@@ -1,2 +1,2 @@\n"
                " nonexistent-context\n-beta\n+gamma\n"
            ))
            self.assertIn("did not apply", out)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), "alpha\nbeta\n")

    def test_fuzzy_trailing_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "w.txt")
            # File has trailing spaces the patch context omits: the fuzzy
            # fallback still applies (context trailing ws normalizes to the
            # patch's form).
            self._write(p, "keep  \ndrop\n")
            out = self._run(tmp, (
                "--- a/w.txt\n+++ b/w.txt\n@@ -1,2 +1,1 @@\n keep\n-drop\n"
            ))
            self.assertIn("applied", out)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), "keep\n")

    def test_file_header_markers_inside_hunk_are_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "markers.txt")
            self._write(p, "-- old marker\nplain\n")
            out = self._run(tmp, (
                "--- a/markers.txt\n+++ b/markers.txt\n"
                "@@ -1,2 +1,2 @@\n"
                "--- old marker\n"
                "+++ new marker\n"
                " plain\n"
            ))
            self.assertIn("applied", out)
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), "++ new marker\nplain\n")


if __name__ == "__main__":
    unittest.main()
