"""Linear vision marker scanning retains the former regex's path semantics."""

import os
import random
import re
import tempfile
import unittest

from Cozter.backends_agent import base


# The old regex, kept here as the expected answer.
_LEGACY_ATTACHMENT_SAVED_RE = re.compile(
    r"\[[^\]\n]*attachment saved to:\s*([^\]\n]+?)\s*\]", re.IGNORECASE,
)


def legacy_paths(prompt: str) -> list[str]:
    return [path for match in _LEGACY_ATTACHMENT_SAVED_RE.finditer(prompt)
            if (path := match.group(1).strip())]


class VisionMarkerScanningTests(unittest.TestCase):
    def test_marker_edge_cases_match_legacy_captures(self):
        cases = [
            "[Photo attachment saved to: photo.png]",
            "[Photo ATTACHMENT SAVED TO: photo.png]",
            "[Photo attachment ſaved to: photo.png]",
            "[[nested [attachment saved to: photo.png]",
            "[x\n[attachment saved to: photo.png]",
            "[attachment saved to:\n\r\tphoto.png\n \t]",
            "[attachment saved to:x\nphoto.png]",
            "[attachment saved to:\n]",
            "[attachment saved to:\n\t\n]",
            "[attachment saved to:x attachment saved to:]",
            "[attachment saved to:x attachment saved to: ]",
            "[attachment saved to:x attachment saved to:\n]",
            "[attachment saved to:x attachment saved to:\r\n]",
            "[attachment saved to:x]\n[attachment saved to:y]",
            "[attachment saved to:x] [unclosed attachment saved to:y",
            "[attachment saved to:\u2003photo.png\u2028]",
            "[attachment saved to:a\u2028b]",
            "[attachment saved to:x\n[attachment saved to:y]",
            "[attachment saved to:x\n[attachment saved to:y\n]",
            "[attachment saved to:x\n[attachment saved to:\n]",
            "[attachment saved to:x\n[attachment saved to:]",
        ]
        for prompt in cases:
            with self.subTest(prompt=prompt):
                self.assertEqual(list(base._attachment_saved_relpaths(prompt)), legacy_paths(prompt))

    def test_deterministic_malformed_and_nested_fixtures_match_legacy(self):
        randomizer = random.Random(913)
        pieces = [
            "[", "]", "\n", "\r", "\t", " ", "\u2003", "\u2028",
            "a", "photo.png", "attachment saved to:", "ATTACHMENT SAVED TO:",
            "attachment ſaved to:", "attachment saved to: ", "[Photo ",
        ]
        for _ in range(12_000):
            prompt = "".join(randomizer.choices(pieces, k=randomizer.randrange(1, 28)))
            self.assertEqual(list(base._attachment_saved_relpaths(prompt)), legacy_paths(prompt), prompt)

    def test_large_unclosed_and_overlapping_prefixes_are_skipped(self):
        for prompt in (
            "[attachment saved to:" * 20_000,
            "[" * 100_000 + "attachment saved to:x\ny]",
            "[" * 100_000 + "\nattachment saved to:x]",
        ):
            with self.subTest(length=len(prompt)):
                self.assertEqual(list(base._attachment_saved_relpaths(prompt)), [])

    def test_large_valid_tail_and_repeated_keywords_keep_last_capture(self):
        path = "x" + " " * 100_000 + "y.png"
        self.assertEqual(list(base._attachment_saved_relpaths(
            "[attachment saved to:" + path + "]",
        )), [path])
        self.assertEqual(list(base._attachment_saved_relpaths(
            "[attachment saved to:x " * 20_000 + "photo.png]",
        )), ["x photo.png"])

    def test_public_resolver_preserves_containment_dedup_and_image_limit(self):
        with tempfile.TemporaryDirectory() as workspace:
            paths = []
            for index in range(5):
                path = os.path.join(workspace, f"photo{index}.png")
                with open(path, "wb") as handle:
                    handle.write(b"fixture")
                paths.append(os.path.realpath(path))
            markers = [
                "[nested [Photo ATTACHMENT SAVED TO:\nphoto0.png \n]",
                "[attachment saved to:photo0.png]",
                "[attachment saved to:../outside.png]",
                "[attachment saved to:\n]",
            ] + [f"[attachment saved to:photo{index}.png]" for index in range(1, 5)]
            self.assertEqual(base.attachment_image_paths("".join(markers), workspace), paths[:4])


if __name__ == "__main__":
    unittest.main()
