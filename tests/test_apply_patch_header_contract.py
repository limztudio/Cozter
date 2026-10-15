"""Unified patches require complete, count-validated canonical hunk headers."""

from pathlib import Path
import tempfile
import unittest

from Cozter.agent_tools.builtin.apply_patch import (
    ApplyPatchTool, _parse_hunk_header, _PatchError,
)


class ApplyPatchHeaderContractTests(unittest.TestCase):
    def test_canonical_implicit_and_explicit_counts_and_section_context(self):
        cases = {
            "@@ -1 +1 @@": (1, 1, 1),
            "@@ -2,3 +8,4 @@": (2, 3, 4),
            "@@ -0,0 +1,2 @@": (0, 0, 2),
            "@@ -1,2 +0,0 @@": (1, 2, 0),
            "@@ -12 +14 @@ a function signature": (12, 1, 1),
        }
        for header, expected in cases.items():
            with self.subTest(header=header):
                self.assertEqual(_parse_hunk_header(header), expected)

    def test_bare_and_malformed_headers_are_rejected_before_any_file_changes(self):
        headers = ("@@", "@@ context", "@@ -x +1 @@", "@@ -1 +1",
                   "@@ -1 +1 @@junk", "@@ -0,1 +1,1 @@", "@@ -1,1 +0,1 @@")
        with tempfile.TemporaryDirectory() as workspace:
            first, second = Path(workspace, "first.txt"), Path(workspace, "second.txt")
            for header in headers:
                with self.subTest(header=header):
                    first.write_text("first old\n", encoding="utf-8")
                    second.write_text("second old\n", encoding="utf-8")
                    patch = (
                        "--- a/first.txt\n+++ b/first.txt\n@@ -1 +1 @@\n-first old\n+first new\n"
                        f"--- a/second.txt\n+++ b/second.txt\n{header}\n-second old\n+second new\n"
                    )
                    output = ApplyPatchTool()._run_sync(workspace, {"patch": patch})
                    self.assertIn("could not parse patch", output)
                    self.assertEqual(first.read_text(encoding="utf-8"), "first old\n")
                    self.assertEqual(second.read_text(encoding="utf-8"), "second old\n")

    def test_declared_counts_still_reject_incomplete_or_excess_hunk_bodies(self):
        for body in ("-old\n", "-old\n+new\n+extra\n"):
            with tempfile.TemporaryDirectory() as workspace:
                target = Path(workspace, "file.txt")
                target.write_text("old\n", encoding="utf-8")
                output = ApplyPatchTool()._run_sync(workspace, {
                    "patch": "--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n" + body,
                })
                self.assertIn("could not parse patch", output)
                self.assertEqual(target.read_text(encoding="utf-8"), "old\n")

    def test_invalid_header_parser_has_no_countless_result(self):
        with self.assertRaises(_PatchError):
            _parse_hunk_header("@@")


if __name__ == "__main__":
    unittest.main()
