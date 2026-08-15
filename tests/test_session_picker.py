"""Tests for the /sessions selection helper (number / name / substring)."""

import unittest
from typing import ClassVar

from Cozter import router
from Cozter.backends_bot.base import BotPlatform


class SessionPickerTests(unittest.TestCase):
    SESSIONS: ClassVar[list[dict[str, int | str]]] = [
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
        self.assertIsNone(self._pick("9" * 5_000))

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

    def test_option_picker_rejects_overlong_decimal_input(self) -> None:
        self.assertIsNone(BotPlatform._pick_option(
            "9" * 5_000, ["first", "second"],
        ))

    def test_router_accepts_safe_punctuated_ids_and_bounds_state(self) -> None:
        session_id = "restored-session_1"
        block = router._build_session_block({
            "id": session_id,
            "name": "N" * 5_000,
            "summary": "S" * 5_000,
            "long_term": ["L" * 5_000] * 5,
        })

        self.assertIn(f"id: {session_id}", block)
        self.assertLessEqual(len(block), router.ROUTER_PER_SESSION_CHARS)
        self.assertEqual(
            router._parse_router_output(session_id, {session_id}), session_id,
        )


if __name__ == "__main__":
    unittest.main()
