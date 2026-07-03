"""Tests for the /sessions selection helper (number / name / substring)."""

import unittest

from Cozter.backends_bot.base import BotPlatform


class SessionPickerTests(unittest.TestCase):
    SESSIONS = [
        {"id": "aaa", "name": "Alpha work", "message_count": 3},
        {"id": "bbb", "name": "Beta notes", "message_count": 0},
        {"id": "ccc", "name": "Session 2026-07-03", "message_count": 10},
    ]

    def _pick(self, choice: str):
        return BotPlatform._pick_session(choice, self.SESSIONS)

    def test_pick_by_number(self) -> None:
        picked = self._pick("2")
        assert picked is not None
        self.assertEqual(picked["id"], "bbb")

    def test_number_out_of_range(self) -> None:
        self.assertIsNone(self._pick("9"))
        self.assertIsNone(self._pick("0"))

    def test_pick_by_exact_name_case_insensitive(self) -> None:
        picked = self._pick("beta notes")
        assert picked is not None
        self.assertEqual(picked["id"], "bbb")

    def test_pick_by_substring(self) -> None:
        picked = self._pick("alpha")
        assert picked is not None
        self.assertEqual(picked["id"], "aaa")

    def test_no_match_or_empty(self) -> None:
        self.assertIsNone(self._pick("zzz"))
        self.assertIsNone(self._pick(""))


if __name__ == "__main__":
    unittest.main()
