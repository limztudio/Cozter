"""Selective discovery avoids filesystem work without changing containment."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from Cozter.agent_tools import base


class DiscoveryOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "workspace"
        self.outside = self.root / "outside"
        self.workspace.mkdir()
        self.outside.mkdir()
        paths = (
            "root.js", "src/main.js", "src/deep/entry.js", "src/deep/entry.ts",
            "src/readme.txt", "node_modules/dep/index.js",
            "node_modules/dep/node_modules/nested.js",
        )
        for relative in paths:
            path = self.workspace / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")
        for number in range(100):
            (self.workspace / "src" / f"unmatched-{number}.txt").touch()

    def _scan(self, pattern: str, approval: str = "auto", root: Path | None = None):
        with base.tool_permission_scope(approval):
            return list(base.iter_workspace_files(
                str(self.workspace), str(root or self.workspace), pattern,
            ))

    def test_bounded_selective_glob_does_not_resolve_unmatched_files(self) -> None:
        original = os.path.realpath
        with mock.patch.object(os.path, "realpath", wraps=original) as resolve:
            found = self._scan("**/*.js")
        self.assertEqual({display for _, display, _ in found}, {
            "root.js", "src/main.js", "src/deep/entry.js",
        })
        resolved = [os.fspath(call.args[0]) for call in resolve.call_args_list]
        self.assertFalse(any(path.endswith(".txt") for path in resolved))
        # Workspace/root once, then one candidate/root check per match.
        self.assertLessEqual(len(resolved), 2 + 2 * len(found))

    def test_full_scan_skips_unused_resolved_target_checks(self) -> None:
        original = os.path.realpath
        with mock.patch.object(os.path, "realpath", wraps=original) as resolve:
            found = self._scan("**/*.js", "full")
        self.assertEqual({display for _, display, _ in found}, {
            "root.js", "src/main.js", "src/deep/entry.js",
        })
        self.assertEqual(resolve.call_count, 2)
        self.assertTrue(all(os.fspath(call.args[0]) == str(self.workspace)
                            for call in resolve.call_args_list))

    def test_pattern_selection_and_explicit_generated_directories_are_unchanged(self) -> None:
        cases = (
            ("**/*.js", {"root.js", "src/main.js", "src/deep/entry.js"}),
            ("src/*.js", {"src/main.js"}),
            ("src/**/entry.[jt]s", {"src/deep/entry.js", "src/deep/entry.ts"}),
            ("node_modules/**/*.js", {
                "node_modules/dep/index.js", "node_modules/dep/node_modules/nested.js",
            }),
        )
        for pattern, expected in cases:
            for approval in ("auto", "full"):
                with self.subTest(pattern=pattern, approval=approval):
                    found = self._scan(pattern, approval)
                    self.assertEqual({display for _, display, _ in found}, expected)
                    self.assertEqual({relative for _, _, relative in found}, expected)

    def test_matching_external_target_is_rejected_only_in_bounded_modes(self) -> None:
        # Simulate a file link for Windows hosts without symlink privileges.
        # All other realpath operations still use the native filesystem.
        link = self.workspace / "escape.js"
        link.touch()
        target = self.outside / "target.js"
        target.write_text("outside\n", encoding="utf-8")
        original = os.path.realpath

        def resolve(path, *args, **kwargs):
            if os.fspath(path) == str(link):
                return str(target)
            return original(path, *args, **kwargs)

        with mock.patch.object(os.path, "realpath", side_effect=resolve):
            full = self._scan("**/*.js", "full")
            bounded = self._scan("**/*.js", "auto")
        self.assertIn((str(link), "escape.js", "escape.js"), full)
        self.assertNotIn("escape.js", {display for _, display, _ in bounded})
        self.assertEqual({display for _, display, _ in bounded}, {
            "root.js", "src/main.js", "src/deep/entry.js",
        })

    def test_real_symlinks_preserve_internal_selection_and_external_labels(self) -> None:
        internal = self.workspace / "inside-link.js"
        external = self.workspace / "outside-link.js"
        target = self.outside / "target.js"
        target.write_text("outside\n", encoding="utf-8")
        try:
            internal.symlink_to(self.workspace / "root.js")
            external.symlink_to(target)
        except (NotImplementedError, OSError) as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        bounded = self._scan("**/*.js", "auto")
        full = self._scan("**/*.js", "full")
        self.assertIn((str(internal), "inside-link.js", "inside-link.js"), bounded)
        self.assertNotIn("outside-link.js", {display for _, display, _ in bounded})
        self.assertIn((str(external), "outside-link.js", "outside-link.js"), full)
        self.assertEqual({display for _, display, _ in full}
                         - {display for _, display, _ in bounded}, {"outside-link.js"})
        outside_found = self._scan("**/*.js", "full", root=self.outside)
        self.assertEqual(outside_found, [(
            str(target), str(target).replace(os.sep, "/"), "target.js",
        )])


if __name__ == "__main__":
    unittest.main()
