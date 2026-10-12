"""Focused regressions for the 2026-10-13 core source audit."""

import os
import struct
import tempfile
import unittest

from Cozter import utils


class ImageHeaderAuditTests(unittest.TestCase):
    @staticmethod
    def _jpeg_header(
        marker: int = 0xC0, *, fill: bytes = b"", width: int = 47, height: int = 31,
    ) -> bytes:
        # One-component SOF: length, precision, height, width, components.
        frame = struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
        return b"\xff\xd8" + fill + bytes((0xFF, marker)) + frame + b"\xff\xd9"

    def _probe(self, content: bytes) -> tuple[int, int, str] | None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "image.jpg")
            with open(path, "wb") as stream:
                stream.write(content)
            return utils.probe_image_dimensions(path)

    def test_jpeg_marker_fill_bytes_preserve_frame_dimensions(self) -> None:
        for fill in (b"", b"\xff", b"\xff\xff"):
            with self.subTest(fill=fill):
                self.assertEqual(
                    self._probe(self._jpeg_header(fill=fill)), (47, 31, "JPEG"),
                )

    def test_jpeg_all_start_of_frame_variants_report_dimensions(self) -> None:
        # C4 is a Huffman table, C8 an extension, and CC an arithmetic table.
        for marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                       0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            with self.subTest(marker=marker):
                self.assertEqual(
                    self._probe(self._jpeg_header(marker)), (47, 31, "JPEG"),
                )

    def test_jpeg_segment_payload_is_not_scanned_as_a_frame(self) -> None:
        fake_frame = self._jpeg_header(width=99, height=88)[2:-2]
        app_segment = b"\xff\xe0" + struct.pack(">H", len(fake_frame) + 2) + fake_frame
        header = b"\xff\xd8" + app_segment + self._jpeg_header(0xC2, fill=b"\xff")[2:]
        self.assertEqual(self._probe(header), (47, 31, "JPEG"))

    def test_jpeg_short_frame_segment_has_no_dimensions(self) -> None:
        header = b"\xff\xd8\xff\xc0\x00\x02" + b"\x01" * 20
        self.assertIsNone(self._probe(header))

    def test_jpeg_frame_bytes_after_scan_or_end_are_not_header_dimensions(self) -> None:
        for boundary in (b"\xff\xd9", b"\xff\xda\x00\x02"):
            with self.subTest(boundary=boundary):
                header = b"\xff\xd8" + boundary + self._jpeg_header()[2:]
                self.assertIsNone(self._probe(header))

    def test_jpeg_truncated_frame_and_fill_bytes_return_none(self) -> None:
        for header in (self._jpeg_header()[:11], b"\xff\xd8" + b"\xff" * 20):
            with self.subTest(header=header):
                self.assertIsNone(self._probe(header))
