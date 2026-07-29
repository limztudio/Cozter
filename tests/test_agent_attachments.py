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


if __name__ == "__main__":
    unittest.main()
