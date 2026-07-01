import asyncio
import os
import subprocess
import sys
import tempfile
import unittest

from Cozter.agent_tools.base import (
    apply_string_replacement,
    coerce_int_arg,
    validate_replacement_strings,
)
from Cozter.agent_tools.builtin.edit_file import EditFileTool
from Cozter.agent_tools.builtin.glob import GlobTool
from Cozter.agent_tools.builtin.grep import GrepTool
from Cozter.agent_tools.builtin.multi_edit import MultiEditTool


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


if __name__ == "__main__":
    unittest.main()
