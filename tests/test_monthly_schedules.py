"""Calendar and input contracts for weekly and monthly schedules."""

import unittest
from datetime import datetime

from Cozter import schedules


class ScheduleDayTests(unittest.TestCase):
    def test_numbers_are_month_dates_and_names_are_weekdays(self) -> None:
        self.assertEqual(schedules.parse_days("1, 7, 15, 31"), [1, 7, 15, 31])
        self.assertEqual(
            schedules.parse_days("Monday, WED, Sunday"), ["mon", "wed", "sun"],
        )
        self.assertEqual(
            schedules.parse_days("15, Monday, 01, mon, 15"), [15, "mon", 1],
        )

    def test_invalid_input_rejects_entire_text(self) -> None:
        for value in (
            None, True, 1, [1], "", "0", "32", "-1", "+1", "1.0",
            "mon,32", "all,15", "money", "tuesdays", "1,unknown", "9" * 5_000,
        ):
            with self.subTest(value=repr(value)[:80]):
                self.assertEqual(schedules.parse_days(value), [])

    def test_persisted_dates_normalize_without_rewriting_weekdays(self) -> None:
        self.assertEqual(
            schedules.normalize_days(["Monday", "mon", 1, "01", "31", "WED"]),
            ["mon", 1, 31, "wed"],
        )
        self.assertEqual(schedules.normalize_days(["mon", "wed"]), ["mon", "wed"])

    def test_persisted_invalid_items_do_not_hide_valid_dates(self) -> None:
        self.assertEqual(
            schedules.normalize_days([
                True, False, 0, -1, 32, 1.0, None, {}, [], "1.0", "all",
                "unknown", "9" * 5_000, "fri", 15,
            ]),
            ["fri", 15],
        )
        for value in (None, "mon", 1, ("mon",)):
            with self.subTest(value=value):
                self.assertEqual(schedules.normalize_days(value), [])

    def test_labels_explain_recurrence_and_preserve_input_order(self) -> None:
        for value, label in (
            (list(schedules.DAY_ABBREV), "Daily"),
            (["wed", "mon", "wed"], "Weekly: wed, mon"),
            ([15, "1", 15], "Monthly: 15, 1"),
            ([15, "mon", "wed", 1], "Weekly: mon, wed; Monthly: 15, 1"),
            ([True, 32, "unknown"], "?"),
            ("mon", "?"),
        ):
            with self.subTest(value=value):
                self.assertEqual(schedules.format_days(value), label)


class MonthlyCalendarTests(unittest.TestCase):
    def slot(self, days: object, now: datetime, time: str = "09:00") -> datetime | None:
        return schedules.most_recent_slot({"days": days, "time": time}, now)

    def test_before_time_uses_previous_month(self) -> None:
        self.assertEqual(
            self.slot([15], datetime(2026, 10, 15, 8, 59)), datetime(2026, 9, 15, 9),
        )

    def test_multiple_dates_pick_latest_elapsed_slot(self) -> None:
        self.assertEqual(
            self.slot([31, 1, 15], datetime(2026, 10, 15, 8)), datetime(2026, 10, 1, 9),
        )

    def test_missing_month_date_skips_instead_of_clamping(self) -> None:
        cases = (
            (datetime(2026, 3, 1, 10), datetime(2026, 1, 31, 9)),
            (datetime(2026, 5, 1, 10), datetime(2026, 3, 31, 9)),
            (datetime(2026, 4, 30, 23), datetime(2026, 3, 31, 9)),
        )
        for now, expected in cases:
            with self.subTest(now=now):
                self.assertEqual(self.slot([31], now), expected)

    def test_february_29_and_30_use_latest_valid_month(self) -> None:
        self.assertEqual(
            self.slot([29], datetime(2024, 3, 1)), datetime(2024, 2, 29, 9),
        )
        self.assertEqual(
            self.slot([29], datetime(2025, 3, 1)), datetime(2025, 1, 29, 9),
        )
        self.assertEqual(
            self.slot([30], datetime(2024, 3, 1)), datetime(2024, 1, 30, 9),
        )

    def test_year_boundary_keeps_previous_december_occurrence(self) -> None:
        self.assertEqual(
            self.slot([31], datetime(2026, 1, 1)), datetime(2025, 12, 31, 9),
        )

    def test_weekday_stays_weekly_including_before_today_time(self) -> None:
        self.assertEqual(
            self.slot(["mon"], datetime(2026, 10, 19, 8)), datetime(2026, 10, 12, 9),
        )
        self.assertEqual(
            self.slot(["Monday"], datetime(2026, 10, 19, 9)), datetime(2026, 10, 19, 9),
        )

    def test_mixed_days_are_union_without_duplicate_occurrence(self) -> None:
        self.assertEqual(
            self.slot(["mon", 15], datetime(2026, 10, 16, 9)), datetime(2026, 10, 15, 9),
        )
        self.assertEqual(
            self.slot([15, "mon"], datetime(2026, 10, 20, 9)), datetime(2026, 10, 19, 9),
        )
        self.assertEqual(
            self.slot([19, "mon", "19", "Monday"], datetime(2026, 10, 19, 9)),
            datetime(2026, 10, 19, 9),
        )

    def test_decimal_strings_are_monthly_in_persisted_records(self) -> None:
        self.assertEqual(
            self.slot(["7"], datetime(2026, 10, 16)), datetime(2026, 10, 7, 9),
        )

    def test_invalid_fields_and_items_are_handled_defensively(self) -> None:
        now = datetime(2026, 10, 16)
        for days in (None, "15", [True, 0, 32, 1.0], []):
            with self.subTest(days=days):
                self.assertIsNone(self.slot(days, now))
        self.assertEqual(
            self.slot([True, 32, "bad", 15], now), datetime(2026, 10, 15, 9),
        )
        self.assertIsNone(self.slot([15], now, "24:00"))

    def test_datetime_min_does_not_underflow(self) -> None:
        self.assertIsNone(self.slot(["sun", 31], datetime.min))
        self.assertIsNone(self.slot([1], datetime.min))
        self.assertEqual(self.slot(["mon", 1], datetime.min, "00:00"), datetime.min)
        self.assertEqual(
            self.slot([31], datetime(1, 2, 1)), datetime(1, 1, 31, 9),
        )

    def test_datetime_max_does_not_overflow(self) -> None:
        self.assertEqual(
            self.slot([31], datetime.max), datetime(9999, 12, 31, 9),
        )
        self.assertEqual(
            self.slot(["fri"], datetime.max), datetime(9999, 12, 31, 9),
        )


if __name__ == "__main__":
    unittest.main()
