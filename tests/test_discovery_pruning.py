"""Selective glob traversal preserves file selection and permission checks."""

import os
from pathlib import Path
import posixpath
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from Cozter.agent_tools import base
from Cozter.agent_tools.builtin.glob import GlobTool
from Cozter.agent_tools.builtin.grep import GrepTool
from Cozter.tests.helpers import path_matches_glob


class DiscoveryPruningTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        for relative in (
            "root.js", ".hidden.js", "src/main.js", "src/deep/entry.js",
            "src/deep/more/entry.ts", "src/deep/more/readme.txt",
            "Upper/main.js", "test/main.js", "test/deep/entry.js",
            "abXYZcd/main.txt", "abXYZcd/deep/extra.txt",
            ".venv/pkg/hidden.js", "node_modules/dep/index.js",
            "node_modules/dep/node_modules/nested.js",
        ):
            path = self.workspace / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("needle\n", encoding="utf-8")
        for number in range(10):
            path = self.workspace / f"irrelevant-{number}" / "deep" / "more"
            path.mkdir(parents=True)
            (path / "unmatched.txt").write_text("needle\n", encoding="utf-8")

    def _scan(self, pattern: str, approval: str = "auto"):
        original_walk = os.walk
        visited: list[str] = []

        def walk(root):
            for entry in original_walk(root):
                visited.append(os.path.relpath(entry[0], root).replace(os.sep, "/"))
                yield entry

        with (
            base.tool_permission_scope(approval),
            mock.patch.object(base.os, "walk", side_effect=walk),
        ):
            found = list(base.iter_workspace_files(
                str(self.workspace), str(self.workspace), pattern,
            ))
        return found, visited

    def _unpruned_selection(self, pattern: str):
        """Reference the previous traversal, with the same matching rules."""
        results = []
        skipped = base._discovery_skip_dirs(pattern)
        for directory, children, files in os.walk(self.workspace):
            children[:] = [name for name in children if name not in skipped]
            for name in files:
                path = os.path.join(directory, name)
                relative = os.path.relpath(path, self.workspace).replace(os.sep, "/")
                if path_matches_glob(relative, pattern):
                    results.append((path, relative, relative))
        return results

    def test_root_only_pattern_does_not_visit_any_child_directory(self) -> None:
        found, visited = self._scan("*.js")
        self.assertEqual(visited, ["."])
        self.assertEqual({item[1] for item in found}, {"root.js", ".hidden.js"})

    def test_nonrecursive_wildcard_pattern_stops_at_its_directory_depth(self) -> None:
        found, visited = self._scan("*/main.js")
        self.assertEqual({item[1] for item in found}, {
            "src/main.js", "Upper/main.js", "test/main.js",
        })
        self.assertTrue(all("/" not in directory for directory in visited))
        self.assertNotIn("src/deep", visited)

    def test_fixed_prefix_prunes_siblings_before_recursive_globstar(self) -> None:
        found, visited = self._scan("src/**/*.js")
        self.assertEqual(set(visited), {
            ".", "src", "src/deep", "src/deep/more",
        })
        self.assertEqual({item[1] for item in found}, {
            "src/main.js", "src/deep/entry.js",
        })

    def test_segment_semantics_and_selection_order_match_unpruned_walk(self) -> None:
        patterns = (
            "", ".", "*", "**", "**/", "**/*.js", "./src//*.js",
            r"src\**\*.js", "src/**", "src/**/**/entry.[jt]s",
            "s[rt]c/*.js", "SRC/*.js", "Upper/*.js", "upper/*.js",
            "ab**cd/*.txt", ".venv/**/*.js",
            "node_modules/**/*.js", "*/deep/*.js", "**/deep/*",
        )
        for pattern in patterns:
            expected = self._unpruned_selection(pattern)
            for approval in ("auto", "full"):
                with self.subTest(pattern=pattern, approval=approval):
                    found, _visited = self._scan(pattern, approval)
                    self.assertEqual(found, expected)

    def test_explicit_generated_prefix_survives_separate_search_root(self) -> None:
        root = self.workspace / "node_modules"
        found = list(base.iter_workspace_files(
            str(self.workspace), str(root), "**/*.js",
            discovery_pattern="node_modules/**/*.js",
        ))
        self.assertEqual({item[1] for item in found}, {
            "node_modules/dep/index.js", "node_modules/dep/node_modules/nested.js",
        })

    def test_glob_limit_and_grep_file_selection_preserve_order(self) -> None:
        pattern = "src/**/*.js"
        expected = self._unpruned_selection(pattern)
        self.assertEqual(GlobTool._find_matches(
            str(self.workspace), pattern, 1,
        ), [item[1] for item in expected[:1]])
        self.assertEqual(GrepTool._scan(
            str(self.workspace), str(self.workspace), pattern,
            re.compile("needle"), 200,
        ), [f"{item[1]}:1: needle" for item in expected])

    def test_pruned_matching_links_still_check_bounded_target_containment(self) -> None:
        # Model file links without depending on Windows symlink privileges.
        internal = self.workspace / "src" / "inside-link.js"
        external = self.workspace / "src" / "outside-link.js"
        internal.touch()
        external.touch()
        outside = self.root / "outside.js"
        outside.write_text("needle\n", encoding="utf-8")
        original = os.path.realpath

        def resolve(path, *args, **kwargs):
            if os.fspath(path) == str(internal):
                return str(self.workspace / "root.js")
            if os.fspath(path) == str(external):
                return str(outside)
            return original(path, *args, **kwargs)

        with mock.patch.object(base.os.path, "realpath", side_effect=resolve):
            bounded, bounded_visited = self._scan("src/*.js", "auto")
            full, full_visited = self._scan("src/*.js", "full")
        self.assertEqual(bounded_visited, full_visited)
        self.assertEqual(set(bounded_visited), {".", "src"})
        self.assertIn((str(internal), "src/inside-link.js", "src/inside-link.js"), bounded)
        self.assertNotIn("src/outside-link.js", {item[1] for item in bounded})
        self.assertEqual({item[1] for item in full} - {item[1] for item in bounded}, {
            "src/outside-link.js",
        })

    def test_literal_posix_backslashes_keep_previous_normalized_selection(self) -> None:
        # Model POSIX names on Windows without creating forbidden filenames.
        entries = {
            "/workspace": ([r"src\deep", "root.js", "ignored"], [r"root\entry.js", "\\", r"\."]),
            r"/workspace/src\deep": (["more"], ["entry.js", r"extra\entry.js"]),
            r"/workspace/src\deep/more": ([], ["nested.js"]),
            "/workspace/root.js": (["\\", r"\."], ["\\", r"\."]),
            "/workspace/root.js/\\": ([], ["\\"]),
            r"/workspace/root.js/\.": ([], [r"\."]),
            "/workspace/ignored": ([], ["entry.js"]),
        }

        def walk(root):
            children, files = entries[root]
            children = children.copy()
            yield root, children, files
            for child in children:
                yield from walk(posixpath.join(root, child))

        filesystem = SimpleNamespace(
            sep="/", walk=walk,
            path=SimpleNamespace(
                realpath=lambda path: path, join=posixpath.join,
                relpath=posixpath.relpath, commonpath=posixpath.commonpath,
                normcase=posixpath.normcase,
            ),
        )
        for pattern in ("src/deep/*.js", "src/deep/**/*.js", "root/*.js", "*.js", "**/*.js", "."):
            expected = []
            for directory, _children, files in walk("/workspace"):
                for name in files:
                    path = posixpath.join(directory, name)
                    relative = posixpath.relpath(path, "/workspace")
                    if path_matches_glob(relative, pattern):
                        expected.append((path, relative, relative))
            with (
                self.subTest(pattern=pattern),
                mock.patch.object(base, "os", filesystem),
                base.tool_permission_scope("full"),
            ):
                self.assertEqual(list(base.iter_workspace_files(
                    "/workspace", "/workspace", pattern,
                )), expected)


if __name__ == "__main__":
    unittest.main()
