"""Block Claude Bash launches that Cozter cannot track durably.

Claude Code runs this module as a ``PreToolUse`` hook for every Bash tool
call.  A foreground Claude session may otherwise use ordinary shell
backgrounding and finish its visible turn before that child process does.
Those jobs have no provider task id for Cozter to persist, poll, cancel, or
report back from.
"""

import json
import re
import shlex
import sys
from collections import deque


_BACKGROUND_HELP = (
    "Cozter cannot track ordinary shell background jobs. Finish this work in "
    "the current Claude session. For an interactive Cozter foreground turn, "
    "put one self-contained [[background: <task>]] marker in the final reply "
    "instead; Cozter will start, persist, and report the provider task."
)
_SHELL_COMMANDS = frozenset({"bash", "sh", "zsh", "dash", "ksh", "fish"})
_COMMAND_SEPARATOR_CHARS = frozenset(";|&()")
_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")


def _command_basename(value: str) -> str:
    """Return a shell command's basename on either POSIX or Windows."""
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def _starts_shell_comment(text: str, index: int) -> bool:
    """Return whether an unquoted ``#`` starts a shell comment."""
    return index == 0 or text[index - 1].isspace() or text[index - 1] in ";|&()<>"


def _advance_shell_quote_or_escape(
    text: str, index: int, quote: str,
) -> tuple[int, str] | None:
    """Advance over one quoted or escaped shell fragment, when present.

    The lightweight shell scanners below use the same quote and backslash
    rules.  Keeping them here prevents their safety behavior from drifting
    when one scanner is tightened without the other.
    """
    char = text[index]
    if quote != "'":
        if text.startswith("$(", index):
            return _substitution_end(text, index) + 1, quote
        if char == "`":
            return _backtick_end(text, index) + 1, quote
    if quote:
        if char == "\\" and quote == '"':
            return index + 2, quote
        if char == quote:
            quote = ""
        return index + 1, quote
    if char in "\"'":
        return index + 1, char
    if char == "\\":
        return index + 2, quote
    return None


def _heredoc_delimiters(line: str) -> list[tuple[str, bool, bool]]:
    """Find straightforward here-document delimiters on one command line.

    The hook only needs to avoid mistaking literal ``&`` characters in a
    here-document body for a shell background operator.  We intentionally do
    not try to implement all of Bash's grammar here; quoted and unquoted
    delimiters cover the normal ``cat <<'EOF'``/``python <<PY`` forms agents
    use to write source files.
    """
    delimiters: list[tuple[str, bool, bool]] = []
    quote = ""
    index = 0
    while index < len(line):
        advanced = _advance_shell_quote_or_escape(line, index, quote)
        if advanced is not None:
            index, quote = advanced
            continue
        char = line[index]
        if char == "#" and _starts_shell_comment(line, index):
            break
        if not line.startswith("<<", index) or line.startswith("<<<", index):
            index += 1
            continue

        cursor = index + 2
        strip_tabs = cursor < len(line) and line[cursor] == "-"
        if strip_tabs:
            cursor += 1
        while cursor < len(line) and line[cursor].isspace():
            cursor += 1
        if cursor >= len(line):
            break
        # Quoting any part of the delimiter disables body expansion, including
        # mixed forms such as E'OF' and escaped forms such as \EOF.
        delimiter_parts: list[str] = []
        delimiter_quote = ""
        literal = False
        while cursor < len(line):
            char = line[cursor]
            if not delimiter_quote and (char.isspace() or char in ";|&()<>"):
                break
            if char == "\\" and delimiter_quote != "'" and cursor + 1 < len(line):
                literal = True
                cursor += 1
                delimiter_parts.append(line[cursor])
            elif char in "\"'" and (not delimiter_quote or char == delimiter_quote):
                literal = True
                delimiter_quote = "" if delimiter_quote else char
            else:
                delimiter_parts.append(char)
            cursor += 1
        delimiter = "".join(delimiter_parts)
        if delimiter or literal:
            delimiters.append((delimiter, strip_tabs, not literal))
        index = max(cursor, index + 2)
    return delimiters


def _without_heredoc_bodies(command: str) -> str:
    """Keep executable substitutions from expanding heredocs, but discard data."""
    pending: deque[tuple[str, bool, bool]] = deque()
    kept: list[str] = []
    body: list[str] = []
    for line in command.splitlines(keepends=True):
        if pending:
            delimiter, strip_tabs, expand = pending[0]
            body_line = line.rstrip("\r\n")
            if strip_tabs:
                body_line = body_line.lstrip("\t")
            # Preserve line count; bodies must not mimic operators.
            kept.append("\n" if line.endswith("\n") else "")
            if body_line == delimiter:
                if expand:
                    for substitution in _command_substitutions("".join(body), literal_text=True):
                        kept.extend(("\n", substitution, "\n"))
                body.clear()
                pending.popleft()
            elif expand:
                body.append(line)
            continue
        kept.append(line)
        pending.extend(_heredoc_delimiters(line))
    if pending and pending[0][2]:
        for substitution in _command_substitutions("".join(body), literal_text=True):
            kept.extend(("\n", substitution, "\n"))
    return "".join(kept)


def _backtick_end(command: str, index: int) -> int:
    cursor = index + 1
    while cursor < len(command):
        if command[cursor] == "\\":
            cursor += 2
        elif command[cursor] == "`":
            return cursor
        else:
            cursor += 1
    return len(command)


def _substitution_end(command: str, index: int) -> int:
    """Match a command substitution with its own quote and parenthesis scope."""
    cursor = index + 2
    depth = 1
    quote = ""
    parents: list[tuple[int, str]] = []
    while cursor < len(command):
        char = command[cursor]
        if char == "\\" and quote != "'":
            cursor += 2
            continue
        if quote == "'":
            if char == "'":
                quote = ""
        elif command.startswith("$(", cursor):
            parents.append((depth, quote))
            depth, quote = 1, ""
            cursor += 2
            continue
        elif char == "`":
            cursor = _backtick_end(command, cursor) + 1
            continue
        elif char in "\"'" and (not quote or char == quote):
            quote = "" if quote else char
        elif not quote:
            if char == "#" and _starts_shell_comment(command, cursor):
                newline = command.find("\n", cursor)
                cursor = len(command) if newline == -1 else newline + 1
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    if not parents:
                        return cursor
                    depth, quote = parents.pop()
        cursor += 1
    return len(command)


def _command_substitutions(command: str, *, literal_text: bool = False) -> list[str]:
    """Extract executable $() and backticks, including inside double quotes.

    Quotes inside an expanding heredoc body are data, so they do not suppress
    its substitutions. Escaped expansion markers remain literal in both modes.
    """
    substitutions: list[str] = []
    quote = ""
    index = 0
    while index < len(command):
        char = command[index]
        if char == "\\" and (literal_text or quote != "'"):
            index += 2
            continue
        if literal_text or quote != "'":
            if command.startswith("$(", index) and not command.startswith("$((", index):
                end = _substitution_end(command, index)
                substitutions.append(command[index + 2:end])
                index = end + 1
                continue
            if char == "`":
                end = _backtick_end(command, index)
                substitutions.append(command[index + 1:end])
                index = end + 1
                continue
            if not literal_text and not quote and char == "#" and _starts_shell_comment(command, index):
                newline = command.find("\n", index)
                index = len(command) if newline == -1 else newline + 1
                continue
        if not literal_text and char in "\"'" and (not quote or char == quote):
            quote = "" if quote else char
        index += 1
    return substitutions


def _has_background_operator(command: str) -> bool:
    """Detect an unquoted shell ``&`` control operator.

    ``&&`` and ``&>``/``>&`` are not background operators.  Quoted strings,
    comments, and common here-document bodies are ignored so normal source
    generation commands keep working.
    """
    command = _without_heredoc_bodies(command)
    quote = ""
    index = 0
    while index < len(command):
        advanced = _advance_shell_quote_or_escape(command, index, quote)
        if advanced is not None:
            index, quote = advanced
            continue
        char = command[index]
        if char == "#" and _starts_shell_comment(command, index):
            newline = command.find("\n", index + 1)
            index = len(command) if newline == -1 else newline + 1
            continue
        if char != "&":
            index += 1
            continue

        previous = command[index - 1] if index else ""
        following = command[index + 1] if index + 1 < len(command) else ""
        if (
            previous == "&"
            or previous == "|"
            or following == "&"
            or previous == ">"
            or following == ">"
        ):
            index += 1
            continue
        return True
    return False


def _shell_tokens(command: str) -> list[str]:
    """Best-effort shell tokenization without executing the command."""
    # Here-document bodies are data, including '&' and launcher-like text.
    command = _without_heredoc_bodies(command)
    # Preserve shell command boundaries; shlex consumes newlines/comments.
    fragments: list[str] = []
    quote = ""
    index = 0
    while index < len(command):
        advanced = _advance_shell_quote_or_escape(command, index, quote)
        if advanced is not None:
            next_index, quote = advanced
            if command.startswith("$(", index) or command[index] == "`":
                # Each substitution is inspected in its own shell scope.
                # Its inner quotes must not alter the surrounding token.
                fragments.append("__cozter_substitution__")
            else:
                fragments.append(command[index:next_index])
            index = next_index
            continue
        char = command[index]
        if char == "#" and _starts_shell_comment(command, index):
            newline = command.find("\n", index + 1)
            index = len(command) if newline == -1 else newline
            continue
        fragments.append(";" if char == "\n" else char)
        index += 1
    try:
        lexer = shlex.shlex(
            "".join(fragments), posix=True, punctuation_chars="|&;()<>",
        )
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except ValueError:
        # The shell rejects most malformed forms.
        return []


def _command_segments(tokens: list[str]) -> list[list[str]]:
    """Split tokenized shell input into simple command segments."""
    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        # shlex groups adjacent punctuation (e.g. a semicolon followed by
        # our preserved newline); every such control group is a boundary.
        if token and all(char in _COMMAND_SEPARATOR_CHARS for char in token):
            if current:
                segments.append(current)
                current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    return segments


def _skip_command_wrappers(segment: list[str]) -> int:
    """Find the executable after common shell wrappers and assignments."""
    index = 0
    while index < len(segment) and _ASSIGNMENT_RE.fullmatch(segment[index]):
        index += 1
    while index < len(segment):
        name = _command_basename(segment[index])
        if name in {"command", "builtin", "exec"}:
            wrapper_index = index
            index += 1
            while index < len(segment):
                option = segment[index]
                if option == "--":
                    index += 1
                    break
                if name == "command" and option == "-p":
                    index += 1
                    continue
                if name == "exec" and option in {"-c", "-l", "-cl", "-lc"}:
                    index += 1
                    continue
                if name == "exec" and option == "-a":
                    # argv[0] is data; the executable follows its argument.
                    index += 2
                    continue
                if option.startswith("-"):
                    # command -v/-V only inspect names; unknown wrapper flags
                    # must not turn their arguments into executed commands.
                    return wrapper_index
                break
            continue
        if name == "env":
            index += 1
            while index < len(segment):
                token = segment[index]
                if _ASSIGNMENT_RE.fullmatch(token):
                    index += 1
                    continue
                if token.startswith("-"):
                    # ``env -u NAME`` takes an argument.
                    if token in {"-u", "--unset"} and index + 1 < len(segment):
                        index += 2
                    else:
                        index += 1
                    continue
                break
            continue
        break
    return index


def _nested_shell_mechanism(segment: list[str], executable_index: int) -> str | None:
    """Inspect a shell's ``-c`` script argument, if present."""
    for index in range(executable_index + 1, len(segment) - 1):
        option = segment[index]
        if option == "--command" or (
            option.startswith("-")
            and not option.startswith("--")
            and "c" in option[1:]
        ):
            return _background_mechanism(segment[index + 1])
    return None


def _segment_background_mechanism(segment: list[str]) -> str | None:
    """Return a prohibited launcher found in one simple shell command."""
    executable_index = _skip_command_wrappers(segment)
    if executable_index >= len(segment):
        return None
    executable = _command_basename(segment[executable_index])
    if executable == "nohup":
        return "the `nohup` launcher"
    if executable == "disown":
        return "the `disown` shell builtin"
    if executable == "claude":
        for argument in segment[executable_index + 1:]:
            if argument in {"--bg", "--background"} or argument.startswith(
                ("--bg=", "--background="),
            ):
                return "a nested `claude --bg` launch"
    if executable in _SHELL_COMMANDS:
        return _nested_shell_mechanism(segment, executable_index)
    if executable in {"eval", "source", "."}:
        for argument in segment[executable_index + 1:]:
            mechanism = _background_mechanism(argument)
            if mechanism is not None:
                return mechanism
    return None


def _background_mechanism(command: str) -> str | None:
    """Return the first ordinary background-launch mechanism in *command*."""
    if _has_background_operator(command):
        return "a shell `&` background operator"
    for substitution in _command_substitutions(_without_heredoc_bodies(command)):
        mechanism = _background_mechanism(substitution)
        if mechanism is not None:
            return mechanism
    for segment in _command_segments(_shell_tokens(command)):
        mechanism = _segment_background_mechanism(segment)
        if mechanism is not None:
            return mechanism
    return None


def background_launch_mechanism(tool_input: object) -> str | None:
    """Return why a Claude Bash tool input must be rejected, if any."""
    if not isinstance(tool_input, dict):
        return "an invalid Bash request"
    if tool_input.get("run_in_background") is True:
        return "Bash `run_in_background`"
    command = tool_input.get("command")
    if not isinstance(command, str):
        return "a Bash request without a command"
    return _background_mechanism(command)


def pre_tool_use_decision(payload: object) -> dict[str, object] | None:
    """Build Claude Code's JSON deny response for a prohibited Bash call."""
    if not isinstance(payload, dict):
        return _deny("an invalid hook request")
    if payload.get("tool_name") != "Bash":
        return None
    mechanism = background_launch_mechanism(payload.get("tool_input"))
    if mechanism is None:
        return None
    return _deny(mechanism)


def _deny(mechanism: str) -> dict[str, object]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                f"Cozter blocked {mechanism}. {_BACKGROUND_HELP}"
            ),
        },
    }


def main() -> int:
    """Read Claude's hook input and write a deny decision when needed."""
    decision: dict[str, object] | None
    try:
        payload = json.load(sys.stdin)
    except (OSError, json.JSONDecodeError):
        decision = _deny("an unreadable Bash hook request")
    else:
        decision = pre_tool_use_decision(payload)
    if decision is not None:
        print(json.dumps(decision, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
