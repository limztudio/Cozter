"""Tests for agent.py marker parsing and attachment path guards.

The attachment resolver is security-sensitive: an agent must never be able
to attach a file outside the workspace (path traversal). These tests also
cover the [[await]] and [[attach:]] markers the orchestrator relies on.
"""

import os
import tempfile
import unittest
from unittest import mock

from Cozter import agent


class AwaitMarkerTests(unittest.TestCase):
    def test_await_marker_detected_and_stripped(self) -> None:
        cleaned, awaiting = agent.extract_await("all done [[await]]")
        self.assertTrue(awaiting)
        self.assertNotIn("[[await]]", cleaned)
        self.assertIn("all done", cleaned)

    def test_no_await_marker(self) -> None:
        cleaned, awaiting = agent.extract_await("just text")
        self.assertFalse(awaiting)
        self.assertEqual(cleaned, "just text")


class AttachmentGuardTests(unittest.TestCase):
    @staticmethod
    def _write_png(path: str) -> None:
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\nplaceholder")

    def test_workspace_file_is_attachable(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            path = os.path.join(ws, "foo.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("hi")
            real = os.path.realpath(path)
            self.assertEqual(agent.attachment_source_path("foo.txt", ws), real)
            self.assertEqual(
                agent.prepare_attachment_path("foo.txt", ws), real,
            )

    def test_absolute_path_outside_workspace_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as ws, \
                tempfile.TemporaryDirectory() as outside:
            evil = os.path.join(outside, "evil.txt")
            with open(evil, "w", encoding="utf-8") as f:
                f.write("secret")
            # Exists, but outside the workspace and not an image in a
            # trusted generated-image root -> must be refused.
            self.assertIsNone(agent.attachment_source_path(evil, ws))
            self.assertIsNone(agent.prepare_attachment_path(evil, ws))

    def test_nonexistent_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            self.assertIsNone(agent.attachment_source_path("nope.txt", ws))

    def test_extract_attachment_sources_keeps_only_valid(self) -> None:
        with tempfile.TemporaryDirectory() as ws, \
                tempfile.TemporaryDirectory() as outside:
            good = os.path.join(ws, "good.txt")
            with open(good, "w", encoding="utf-8") as f:
                f.write("ok")
            evil = os.path.join(outside, "evil.txt")
            with open(evil, "w", encoding="utf-8") as f:
                f.write("secret")

            text = f"see [[attach: good.txt]] and [[attach: {evil}]] done"
            cleaned, paths = agent.extract_attachment_sources(text, ws)

            self.assertIn(os.path.realpath(good), paths)
            self.assertNotIn(os.path.realpath(evil), paths)
            self.assertNotIn("[[attach:", cleaned)

    def test_auto_detection_never_scans_shared_external_roots(self) -> None:
        """External images require an explicit marker to avoid cross-chat leaks."""
        with tempfile.TemporaryDirectory() as ws, \
                tempfile.TemporaryDirectory() as external:
            before = agent._snapshot_attachment_images(ws)
            external_image = os.path.join(external, "other-turn.png")
            self._write_png(external_image)
            workspace_image = os.path.join(ws, "this-turn.png")
            self._write_png(workspace_image)

            with mock.patch.dict(
                os.environ, {"COZTER_ATTACHMENT_ROOTS": external}, clear=False,
            ):
                detected = agent._collect_new_attachment_images(before, ws)
                explicit = agent.prepare_attachment_path(external_image, ws)

            self.assertEqual(detected, [os.path.realpath(workspace_image)])
            self.assertIsNotNone(explicit)
            assert explicit is not None
            self.assertTrue(explicit.startswith(os.path.realpath(ws) + os.sep))
            self.assertNotEqual(explicit, os.path.realpath(external_image))

    def test_auto_detection_ignores_workspace_symlink_to_trusted_root(self) -> None:
        """A workspace link must not make a trusted shared root auto-scanned."""
        with tempfile.TemporaryDirectory() as ws, \
                tempfile.TemporaryDirectory() as external:
            external_image = os.path.join(external, "other-turn.png")
            self._write_png(external_image)
            workspace_link = os.path.join(ws, "linked-image.png")
            try:
                os.symlink(external_image, workspace_link)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"file symlinks unavailable: {exc}")

            with mock.patch.dict(
                os.environ, {"COZTER_ATTACHMENT_ROOTS": external}, clear=False,
            ):
                before = agent._snapshot_attachment_images(ws)
                with open(external_image, "ab") as f:
                    f.write(b"updated after snapshot")
                detected = agent._collect_new_attachment_images(before, ws)

            self.assertEqual(detected, [])

    def test_external_image_copy_rejects_symlinked_destination(self) -> None:
        """A workspace state symlink must not redirect an image copy outside."""
        with tempfile.TemporaryDirectory() as ws, \
                tempfile.TemporaryDirectory() as external, \
                tempfile.TemporaryDirectory() as outside:
            source = os.path.join(external, "artifact.png")
            self._write_png(source)
            state_dir = os.path.join(ws, ".cozter")
            os.makedirs(state_dir)
            try:
                os.symlink(
                    outside,
                    os.path.join(state_dir, "generated_images"),
                    target_is_directory=True,
                )
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"directory symlinks unavailable: {exc}")

            with mock.patch.dict(
                os.environ, {"COZTER_ATTACHMENT_ROOTS": external}, clear=False,
            ):
                with self.assertLogs(agent.logger, level="WARNING"):
                    copied = agent.prepare_attachment_path(source, ws)

            self.assertIsNone(copied)
            self.assertEqual(os.listdir(outside), [])

    def test_external_image_copy_keeps_a_concurrent_destination(self) -> None:
        """A generated-image race must choose a new name, never overwrite."""
        with tempfile.TemporaryDirectory() as ws, \
                tempfile.TemporaryDirectory() as external:
            source = os.path.join(external, "artifact.png")
            self._write_png(source)
            original_copy = agent.copy_file_atomically
            first_destination = ""
            call_count = 0

            def publish_with_competitor(src: str, destination: str) -> bool:
                nonlocal first_destination, call_count
                call_count += 1
                if call_count == 1:
                    first_destination = destination
                    with open(destination, "wb") as f:
                        f.write(b"concurrent image")
                return original_copy(src, destination)

            with (
                mock.patch.dict(
                    os.environ, {"COZTER_ATTACHMENT_ROOTS": external}, clear=False,
                ),
                mock.patch.object(
                    agent, "copy_file_atomically", side_effect=publish_with_competitor,
                ),
            ):
                copied = agent.prepare_attachment_path(source, ws)

            self.assertIsNotNone(copied)
            assert copied is not None
            self.assertTrue(copied.endswith("-2.png"))
            with open(first_destination, "rb") as f:
                self.assertEqual(f.read(), b"concurrent image")


if __name__ == "__main__":
    unittest.main()
