"""Tests for agent._build_contextual_prompt's configurable budget.

The context block (colony + memory + summary + recent messages) is trimmed
to a character budget, dropping the oldest recent messages first. A larger
budget must keep more history; the user's new message is always present.
"""

import unittest

from Cozter import agent


def _messages(n: int) -> list[dict]:
    return [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"message number {i} " + "x" * 100,
        }
        for i in range(n)
    ]


class ContextBudgetTests(unittest.TestCase):
    def test_larger_budget_keeps_more_history(self) -> None:
        data = {"summary": "", "long_term": [], "messages": _messages(50)}

        big = agent._build_contextual_prompt(
            "NEW MESSAGE", dict(data), budget=1_000_000,
        )
        small = agent._build_contextual_prompt(
            "NEW MESSAGE", dict(data), budget=3_000,
        )

        # The new user message survives at any budget.
        self.assertIn("NEW MESSAGE", big)
        self.assertIn("NEW MESSAGE", small)
        # A tight budget drops older messages, so the prompt is shorter...
        self.assertLess(len(small), len(big))
        # ...but the newest message is retained in both.
        self.assertIn("message number 49", big)
        self.assertIn("message number 49", small)
        # ...while the oldest is dropped under the tight budget.
        self.assertIn("message number 0", big)
        self.assertNotIn("message number 0 ", small)

    def test_no_context_returns_prompt_unchanged(self) -> None:
        out = agent._build_contextual_prompt("hi", None)
        self.assertEqual(out, "hi")


if __name__ == "__main__":
    unittest.main()
