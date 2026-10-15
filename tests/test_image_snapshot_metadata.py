"""Image snapshots retain full discovery while avoiding repeated path/stat I/O."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from Cozter import agent


def _image(root: Path, relative: str, payload: bytes = b"fixture") -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return os.path.realpath(path)


def _legacy_paths(root: str) -> list[str]:
    paths = []
    root_real = os.path.realpath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not agent._skip_automatic_image_directory(name)]
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() not in agent.IMAGE_EXTENSIONS:
                continue
            path = os.path.realpath(os.path.join(dirpath, filename))
            if not agent.is_path_within(path, root_real) or not os.path.isfile(path):
                continue
            parts = os.path.relpath(path, root_real).split(os.sep)[:-1]
            if not any(agent._skip_automatic_image_directory(part) for part in parts):
                paths.append(path)
    return paths


def _legacy_snapshot(root: str) -> dict[str, tuple[int, int]]:
    root = os.path.realpath(root)
    if not os.path.isdir(root):
        return {}
    result = {}
    for path in _legacy_paths(root):
        try:
            metadata = os.stat(path)
        except OSError:
            continue
        result[path] = metadata.st_mtime_ns, metadata.st_size
    return result


class ImageSnapshotMetadataTests(unittest.TestCase):
    def test_full_snapshot_and_path_list_match_legacy_and_refresh_each_scan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(30):
                _image(root, f"output/{index % 4}/image-{index}.PNG", bytes(index))
            _image(root, "한글/🙂.webp")
            _image(root, ".cozter/generated_images/result.jpg")
            _image(root, "directory.png/nested.gif")
            for skipped in ("build", "Testing", "__cmake/build", "cmake-build-debug", "node_modules"):
                _image(root, f"{skipped}/hidden.png")
            (root / "source.py").write_text("ordinary source", encoding="utf-8")
            self.assertEqual(agent._iter_image_files(temporary), _legacy_paths(temporary))
            before = agent._snapshot_attachment_images(temporary)
            self.assertEqual(before, _legacy_snapshot(temporary))
            modified = _image(root, "output/0/image-0.PNG", b"changed bytes")
            created = _image(root, "output/fresh.bmp")
            deleted = root / "output/1/image-1.PNG"
            deleted_real = os.path.realpath(deleted)
            deleted.unlink()
            after = agent._snapshot_attachment_images(temporary)
            self.assertEqual(after, _legacy_snapshot(temporary))
            self.assertNotEqual(before[modified], after[modified])
            self.assertIn(created, after)
            self.assertNotIn(deleted_real, after)
            self.assertEqual(agent._snapshot_attachment_images(str(root / "missing")), {})
            self.assertEqual(agent._snapshot_attachment_images(str(root / "source.py")), {})

    def test_each_snapshot_canonicalizes_root_once_and_stats_each_image_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {_image(root, f"output/fixture-{index}.png") for index in range(20)}
            expected = _legacy_snapshot(temporary)
            with (
                mock.patch.object(agent.os.path, "realpath", wraps=os.path.realpath) as realpath,
                mock.patch.object(agent.os, "stat", wraps=os.stat) as metadata,
            ):
                actual = agent._snapshot_attachment_images(temporary)
            self.assertEqual(actual, expected)
            self.assertEqual(realpath.call_count, len(paths) + 1)
            for path in paths:
                self.assertEqual(sum(os.fspath(call.args[0]) == path for call in metadata.call_args_list), 1)

    def test_failed_metadata_and_nonregular_candidates_do_not_hide_other_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = _image(root, "valid.png")
            denied = _image(root, "denied.png")
            nonregular = _image(root, "nonregular.png")
            directory_metadata = os.stat(root)
            original_stat = os.stat

            def metadata(path, *args, **kwargs):
                if os.fspath(path) == denied:
                    raise PermissionError("fixture denied")
                if os.fspath(path) == nonregular:
                    return directory_metadata
                return original_stat(path, *args, **kwargs)

            with mock.patch.object(agent.os, "stat", metadata):
                self.assertEqual(set(agent._snapshot_attachment_images(temporary)), {valid})
                self.assertEqual(agent._iter_image_files(temporary), [valid])

    def test_file_aliases_keep_legacy_containment_skipped_target_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            target = _image(workspace, "output/target.png")
            hidden = _image(workspace, "build/hidden.png")
            outside = _image(root, "outside/private.png")
            try:
                os.symlink(target, workspace / "alias.png")
                os.symlink(hidden, workspace / "hidden-alias.png")
                os.symlink(outside, workspace / "outside-alias.png")
                os.symlink(root / "missing.png", workspace / "broken.png")
            except (NotImplementedError, OSError) as error:
                self.skipTest(f"file symlinks unavailable: {error}")
            expected_paths = _legacy_paths(str(workspace))
            self.assertEqual(expected_paths.count(target), 2)
            self.assertEqual(agent._iter_image_files(str(workspace)), expected_paths)
            self.assertEqual(agent._snapshot_attachment_images(str(workspace)), _legacy_snapshot(str(workspace)))
            self.assertEqual(set(agent._snapshot_attachment_images(str(workspace))), {target})

    def test_canonical_alias_boundaries_are_checked_without_requiring_symlink_privilege(self):
        # Run these boundary branches even on Windows hosts which cannot create
        # file symlinks. Each mocked resolution models the same canonical path
        # returned by realpath for an actual link; metadata reads remain real.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            target = _image(workspace, "output/target.png")
            hidden = _image(workspace, "build/hidden.png")
            outside = _image(root, "outside/private.png")
            replacements = {
                _image(workspace, "alias.png"): target,
                _image(workspace, "hidden-alias.png"): hidden,
                _image(workspace, "outside-alias.png"): outside,
                _image(workspace, "broken.png"): str(workspace / "missing.png"),
            }
            original_realpath = os.path.realpath

            def resolve(path, *args, **kwargs):
                resolved = original_realpath(path, *args, **kwargs)
                return replacements.get(resolved, resolved)

            with mock.patch.object(agent.os.path, "realpath", resolve):
                paths = agent._iter_image_files(str(workspace))
                self.assertEqual(paths, _legacy_paths(str(workspace)))
                self.assertEqual(paths.count(target), 2)
                self.assertEqual(set(agent._snapshot_attachment_images(str(workspace))), {target})
                self.assertEqual(agent._snapshot_attachment_images(str(workspace)), _legacy_snapshot(str(workspace)))

    @unittest.skipUnless(sys.platform == "win32", "Windows junction behavior")
    def test_windows_root_junction_and_nested_junction_containment_match_legacy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            target = _image(workspace, "output/target.png")
            _image(workspace, "build/hidden.png")
            _image(root, "outside/private.png")
            links = [
                (root / "root-alias", workspace),
                (workspace / "outside-alias", root / "outside"),
                (workspace / "hidden-alias", workspace / "build"),
            ]
            for link, destination in links:
                created = subprocess.run(
                    ["cmd.exe", "/c", "mklink", "/J", str(link), str(destination)],
                    capture_output=True, text=True, check=False,
                )
                if created.returncode:
                    self.skipTest("temporary directory junctions unavailable")
            for selected_root in (workspace, root / "root-alias"):
                with self.subTest(root=selected_root.name):
                    expected = _legacy_snapshot(str(selected_root))
                    self.assertEqual(agent._snapshot_attachment_images(str(selected_root)), expected)
                    self.assertEqual(set(expected), {target})
                    self.assertEqual(agent._iter_image_files(str(selected_root)), _legacy_paths(str(selected_root)))

    def test_path_resolution_error_keeps_prior_results_and_warns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = _image(root, "valid.png")
            _image(root, "failed.png")
            _image(root, "later.png")
            original_realpath = os.path.realpath

            def resolve(path, *args, **kwargs):
                if os.path.basename(path) == "failed.png":
                    raise OSError("fixture path unavailable")
                return original_realpath(path, *args, **kwargs)

            with (
                mock.patch.object(agent.os.path, "realpath", resolve),
                mock.patch.object(agent.os, "walk", return_value=iter([
                    (temporary, [], ["valid.png", "failed.png", "later.png"]),
                ])),
                self.assertLogs(agent.logger, "WARNING"),
            ):
                self.assertEqual(set(agent._snapshot_attachment_images(temporary)), {valid})
