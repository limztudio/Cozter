"""Yearly date syntax, formatting, and calendar matching contracts."""

import unittest
from datetime import datetime

from Cozter import schedules


class YearlyDayTests(unittest.TestCase):
    def test_yearly_dates_accept_one_or_two_digits_and_normalize(self) -> None:
        self.assertEqual(
            schedules.parse_days("1/2, 01/02, 12/25, 2/29, 02/29"),
            ["01/02", "12/25", "02/29"],
        )

    def test_invalid_yearly_syntax_rejects_entire_text(self) -> None:
        for value in (
            "0/1", "13/1", "1/0", "1/32", "2/30", "2/31", "4/31", "11/31",
            "1/", "/1", "1//2", "1/2/3", "2026/1/2", "001/2", "1/002",
            "1 /2", "1/ 2", "+1/2", "-1/2", "1.0/2", "Jan/2", "12-25",
            "mon,02/30", "12/25,unknown", "all,12/25", "9" * 5_000 + "/1",
        ):
            with self.subTest(value=value[:80]):
                self.assertEqual(schedules.parse_days(value), [])

    def test_normalization_filters_invalid_items_without_losing_valid_rules(self) -> None:
        self.assertEqual(
            schedules.normalize_days([
                " 2/29 ", "02/29", "2/30", True, 15, "15", "Friday", "fri",
                "12/25", "04/31", None, 1.5,
            ]),
            ["02/29", 15, "fri", "12/25"],
        )

    def test_yearly_labels_normalize_dates_and_keep_input_order(self) -> None:
        self.assertEqual(
            schedules.format_days(["12/25", "2/29", "12/25"]),
            "Yearly: 12/25, 02/29",
        )

    def test_mixed_labels_group_each_recurrence_kind(self) -> None:
        self.assertEqual(
            schedules.format_days(["12/25", 15, "wed", "2/29", 1, "mon"]),
            "Weekly: wed, mon; Monthly: 15, 1; Yearly: 12/25, 02/29",
        )

    def test_yearly_rule_prevents_daily_only_label(self) -> None:
        days: list[str | int] = [*schedules.DAY_ABBREV, "12/25"]
        self.assertEqual(
            schedules.format_days(days),
            "Weekly: mon, tue, wed, thu, fri, sat, sun; Yearly: 12/25",
        )
        self.assertEqual(schedules.format_days(list(schedules.DAY_ABBREV)), "Daily")

    def test_impossible_yearly_rules_have_invalid_label(self) -> None:
        self.assertEqual(schedules.format_days(["02/30", "04/31"]), "?")


class YearlyCalendarTests(unittest.TestCase):
    def slot(self, days: object, now: datetime, time: str = "09:00") -> datetime | None:
        return schedules.most_recent_slot({"days": days, "time": time}, now)

    def test_yearly_date_is_due_at_exact_time(self) -> None:
        self.assertEqual(
            self.slot(["12/25"], datetime(2026, 12, 25, 9)), datetime(2026, 12, 25, 9),
        )

    def test_before_time_uses_previous_year(self) -> None:
        self.assertEqual(
            self.slot(["12/25"], datetime(2026, 12, 25, 8, 59)), datetime(2025, 12, 25, 9),
        )

    def test_before_date_uses_previous_year_and_after_date_keeps_this_year(self) -> None:
        self.assertEqual(
            self.slot(["12/25"], datetime(2026, 1, 1)), datetime(2025, 12, 25, 9),
        )
        self.assertEqual(
            self.slot(["1/2"], datetime(2026, 12, 31)), datetime(2026, 1, 2, 9),
        )

    def test_multiple_yearly_dates_select_latest_elapsed_occurrence(self) -> None:
        self.assertEqual(
            self.slot(["12/25", "1/2", "10/16"], datetime(2026, 10, 16, 8)),
            datetime(2026, 1, 2, 9),
        )
        self.assertEqual(
            self.slot(["12/25", "1/2", "10/16"], datetime(2026, 10, 16, 9)),
            datetime(2026, 10, 16, 9),
        )

    def test_yearly_february_29_uses_previous_leap_year(self) -> None:
        for now, expected in (
            (datetime(2024, 2, 29, 8), datetime(2020, 2, 29, 9)),
            (datetime(2024, 2, 29, 9), datetime(2024, 2, 29, 9)),
            (datetime(2025, 3, 1), datetime(2024, 2, 29, 9)),
        ):
            with self.subTest(now=now):
                self.assertEqual(self.slot(["02/29"], now), expected)

    def test_february_29_skips_nonleap_century_but_accepts_400_year_boundary(self) -> None:
        for now, expected in (
            (datetime(1903, 3, 1), datetime(1896, 2, 29, 9)),
            (datetime(2000, 3, 1), datetime(2000, 2, 29, 9)),
            (datetime(2104, 2, 29, 8), datetime(2096, 2, 29, 9)),
            (datetime(2400, 3, 1), datetime(2400, 2, 29, 9)),
        ):
            with self.subTest(now=now):
                self.assertEqual(self.slot(["02/29"], now), expected)

    def test_yearly_and_monthly_february_rules_have_distinct_meanings(self) -> None:
        now = datetime(2025, 3, 1)
        self.assertEqual(self.slot([29], now), datetime(2025, 1, 29, 9))
        self.assertEqual(self.slot(["02/29"], now), datetime(2024, 2, 29, 9))

    def test_mixed_weekly_monthly_and_yearly_rules_choose_latest(self) -> None:
        days: list[str | int] = ["mon", 15, "12/25"]
        for now, expected in (
            (datetime(2026, 12, 16), datetime(2026, 12, 15, 9)),
            (datetime(2026, 12, 23), datetime(2026, 12, 21, 9)),
            (datetime(2026, 12, 26), datetime(2026, 12, 25, 9)),
        ):
            with self.subTest(now=now):
                self.assertEqual(self.slot(days, now), expected)

    def test_all_three_overlapping_rules_yield_one_latest_slot(self) -> None:
        self.assertEqual(
            self.slot(["fri", 25, "12/25", "12/25"], datetime(2026, 12, 25, 9)),
            datetime(2026, 12, 25, 9),
        )

    def test_invalid_yearly_items_do_not_hide_valid_rules(self) -> None:
        self.assertEqual(
            self.slot(["02/30", "12/25"], datetime(2026, 12, 26)),
            datetime(2026, 12, 25, 9),
        )
        self.assertIsNone(self.slot(["02/30", "04/31"], datetime(2026, 12, 26)))

    def test_datetime_min_and_first_leap_year_are_safe(self) -> None:
        self.assertIsNone(self.slot(["12/25", "02/29"], datetime.min))
        self.assertIsNone(self.slot(["01/01"], datetime.min))
        self.assertEqual(self.slot(["1/1"], datetime.min, "00:00"), datetime.min)
        self.assertIsNone(self.slot(["02/29"], datetime(3, 12, 31)))
        self.assertEqual(
            self.slot(["02/29"], datetime(4, 3, 1)), datetime(4, 2, 29, 9),
        )

    def test_datetime_max_and_last_leap_year_are_safe(self) -> None:
        self.assertEqual(
            self.slot(["12/31"], datetime.max), datetime(9999, 12, 31, 9),
        )
        self.assertEqual(
            self.slot(["02/29"], datetime.max), datetime(9996, 2, 29, 9),
        )


if __name__ == "__main__":
    unittest.main()
