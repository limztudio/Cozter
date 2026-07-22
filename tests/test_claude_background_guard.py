"""Regression coverage for the Claude Bash background-job guard."""

import io
import json
import unittest
from unittest import mock

from Cozter.backends_agent import claude_background_guard as guard


class BackgroundLaunchMechanismTests(unittest.TestCase):
    def test_rejects_bash_run_in_background(self) -> None:
        self.assertEqual(
            guard.background_launch_mechanism({
                "command": "pytest", "run_in_background": True,
            }),
            "Bash `run_in_background`",
        )

    def test_rejects_untracked_shell_launchers(self) -> None:
        cases = {
            "ninja &": "a shell `&` background operator",
            "nohup pytest": "the `nohup` launcher",
            "command disown": "the `disown` shell builtin",
            "env CI=1 claude --background 'run checks'": (
                "a nested `claude --bg` launch"
            ),
            "bash -lc 'pytest &'": "a shell `&` background operator",
            "sh -c 'nohup pytest'": "the `nohup` launcher",
        }
        for command, expected in cases.items():
            with self.subTest(command=command):
                self.assertEqual(
                    guard.background_launch_mechanism({"command": command}),
                    expected,
                )

    def test_does_not_confuse_normal_shell_syntax_with_backgrounding(self) -> None:
        safe_commands = [
            "pytest && ruff check .",
            "pytest &> test.log",
            "printf '%s' '&'",
            "echo hi # `nohup` and & in a comment",
            "echo $((flags & 1))",
            "cat <<'PY' > generated.py\nvalue = 'a & b'\nPY\n",
        ]
        for command in safe_commands:
            with self.subTest(command=command):
                self.assertIsNone(
                    guard.background_launch_mechanism({"command": command}),
                )


class PreToolUseHookTests(unittest.TestCase):
    def test_bash_background_request_returns_claude_deny_shape(self) -> None:
        decision = guard.pre_tool_use_decision({
            "tool_name": "Bash",
            "tool_input": {"command": "make &"},
        })

        self.assertIsNotNone(decision)
        assert decision is not None
        output = decision["hookSpecificOutput"]
        self.assertIsInstance(output, dict)
        assert isinstance(output, dict)
        self.assertEqual(output["hookEventName"], "PreToolUse")
        self.assertEqual(output["permissionDecision"], "deny")
        self.assertIn("[[background: <task>]]", output["permissionDecisionReason"])

    def test_non_bash_tool_is_left_alone(self) -> None:
        self.assertIsNone(guard.pre_tool_use_decision({
            "tool_name": "Write",
            "tool_input": {"content": "nohup &"},
        }))

    def test_main_writes_only_machine_readable_deny_decision(self) -> None:
        stdout = io.StringIO()
        with (
            mock.patch.object(
                guard.sys, "stdin",
                io.StringIO(json.dumps({
                    "tool_name": "Bash",
                    "tool_input": {"command": "claude --bg 'run checks'"},
                })),
            ),
            mock.patch.object(guard.sys, "stdout", stdout),
        ):
            self.assertEqual(guard.main(), 0)

        output = json.loads(stdout.getvalue())
        self.assertEqual(
            output["hookSpecificOutput"]["permissionDecision"], "deny",
        )


if __name__ == "__main__":
    unittest.main()
