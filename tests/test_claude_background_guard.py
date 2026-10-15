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
            "pytest |& tee test.log",
            "pytest &> test.log",
            "printf '%s' '&'",
            "echo hi # `nohup` and & in a comment",
            "echo $((flags & 1))",
            "cat <<'PY' > generated.py\nvalue = 'a & b'\nPY\n",
            "cat <<'SH' > script.sh\necho ready; nohup task &\nSH\n",
        ]
        for command in safe_commands:
            with self.subTest(command=command):
                self.assertIsNone(
                    guard.background_launch_mechanism({"command": command}),
                )

    def test_rejects_launchers_after_a_newline_or_comment(self) -> None:
        for prefix in (
            "echo ready\n", "echo ready # setup\n",
            "echo ready;\n", "echo ready &&\n",
            "cat <<'EOF'\nready\nEOF\n",
        ):
            for command, expected in (
                ("nohup pytest", "the `nohup` launcher"),
                ("disown", "the `disown` shell builtin"),
                ("claude --bg 'run checks'", "a nested `claude --bg` launch"),
            ):
                with self.subTest(prefix=prefix, command=command):
                    self.assertEqual(
                        guard.background_launch_mechanism({"command": prefix + command}),
                        expected,
                    )

    def test_quoted_and_escaped_newlines_do_not_start_commands(self) -> None:
        for command in (
            "printf '%s' 'ready\nnohup pytest'",
            'printf "%s" "ready\nclaude --bg task"',
            "printf '%s' ready\\\nnohup",
            "echo ready#value\ntrue",
        ):
            with self.subTest(command=command):
                self.assertIsNone(guard.background_launch_mechanism({"command": command}))

    def test_rejects_executable_substitutions_in_quotes_and_expanding_heredocs(self) -> None:
        commands = (
            'echo "$(sleep 60 >/dev/null &)"',
            'echo "`sleep 60 >/dev/null &`"',
            'echo "$(printf "%s" "$(sleep 60 >/dev/null &)")"',
            'echo "$(nohup task)"',
            'echo "`claude --bg task`"',
            'echo $(( $(sleep 60 >/dev/null &) + 1 ))',
            "cat <<EOF\n$(sleep 60 >/dev/null &)\nEOF\n",
            "cat <<EOF\n'$(nohup task)'\nEOF\n",
            'cat <<EOF\n"`claude --bg task`"\nEOF\n',
            "cat <<-EOF\n\t$(\nsleep 60 >/dev/null &\n)\n\tEOF\n",
            "cat <<'LITERAL' <<EXPAND\n$(sleep 60 &)\nLITERAL\n$(nohup task)\nEXPAND\n",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertIsNotNone(guard.background_launch_mechanism({"command": command}))

    def test_literal_expansion_markers_and_heredoc_data_remain_allowed(self) -> None:
        commands = (
            "echo '$(sleep 60 &) `nohup task`'",
            r'echo "\$(sleep 60 &) \`nohup task\`"',
            'echo "$(printf "%s" "literal &")"',
            "cat <<EOF\nliteral & nohup task\nEOF\n",
            "cat <<EOF\n\\$(sleep 60 &) \\`nohup task\\`\nEOF\n",
            "cat <<'EOF'\n$(sleep 60 &) `nohup task`\nEOF\n",
            'cat <<"EOF"\n$(sleep 60 &) `nohup task`\nEOF\n',
            "cat <<E'OF'\n$(sleep 60 &) `nohup task`\nEOF\n",
            "cat <<\\EOF\n$(sleep 60 &) `nohup task`\nEOF\n",
            "cat <<''\n$(sleep 60 &) `nohup task`\n\n",
            'echo ready # "$(nohup task)"\ntrue',
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertIsNone(guard.background_launch_mechanism({"command": command}))


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
