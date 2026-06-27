import asyncio
import os
import tempfile
import unittest

from Cozter.agent_tools.base import (
    apply_string_replacement,
    coerce_int_arg,
    validate_replacement_strings,
)
from Cozter.agent_tools.builtin.edit_file import EditFileTool
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
