"""Tests for agent._build_contextual_prompt's configurable budget.

The context block (colony + memory + summary + recent messages) is trimmed
to a character budget, dropping the oldest recent messages first. A larger
budget must keep more history; the user's new message is always present.
"""

import tempfile
import unittest

from Cozter import agent, workspace


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


class PromptPolicyTests(unittest.TestCase):
    def test_explicit_session_turn_is_autonomous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace.set_interaction_style(tmp, "collaborative")

            self.assertTrue(
                agent._is_collaborative_turn(
                    tmp, explicit_session=False,
                ),
            )
            self.assertFalse(
                agent._is_collaborative_turn(
                    tmp, explicit_session=True,
                ),
            )


class FormatUsageTests(unittest.TestCase):
    def test_none_and_empty_return_none(self) -> None:
        self.assertIsNone(agent.format_usage(None))
        self.assertIsNone(agent.format_usage({}))
        self.assertIsNone(agent.format_usage("not a dict"))

    def test_tokens_and_cost(self) -> None:
        footer = agent.format_usage({
            "input_tokens": 12470, "output_tokens": 28,
            "total_cost_usd": 0.0123,
        })
        assert footer is not None
        self.assertIn("12.5k in", footer)
        self.assertIn("28 out", footer)
        self.assertIn("$0.0123", footer)

    def test_tokens_only_no_cost(self) -> None:
        footer = agent.format_usage(
            {"input_tokens": 500, "output_tokens": 10},
        )
        assert footer is not None
        self.assertIn("500 in", footer)
        self.assertNotIn("$", footer)


if __name__ == "__main__":
    unittest.main()
