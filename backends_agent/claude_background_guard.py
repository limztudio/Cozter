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


_BACKGROUND_HELP = (
    "Cozter cannot track ordinary shell background jobs. Finish this work in "
    "the current Claude session. For an interactive Cozter foreground turn, "
    "put one self-contained [[background: <task>]] marker in the final reply "
    "instead; Cozter will start, persist, and report the provider task."
)
_SHELL_COMMANDS = frozenset({"bash", "sh", "zsh", "dash", "ksh", "fish"})
_COMMAND_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "(", ")"})
_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")


def _command_basename(value: str) -> str:
    """Return a shell command's basename on either POSIX or Windows."""
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def _starts_shell_comment(text: str, index: int) -> bool:
    """Return whether an unquoted ``#`` starts a shell comment."""
    return index == 0 or text[index - 1].isspace() or text[index - 1] in ";|&()<>"


def _heredoc_delimiters(line: str) -> list[tuple[str, bool]]:
    """Find straightforward here-document delimiters on one command line.

    The hook only needs to avoid mistaking literal ``&`` characters in a
    here-document body for a shell background operator.  We intentionally do
    not try to implement all of Bash's grammar here; quoted and unquoted
    delimiters cover the normal ``cat <<'EOF'``/``python <<PY`` forms agents
    use to write source files.
    """
    delimiters: list[tuple[str, bool]] = []
    quote = ""
    index = 0
    while index < len(line):
        char = line[index]
        if quote:
            if char == "\\" and quote == '"':
                index += 2
                continue
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in "\"'":
            quote = char
            index += 1
            continue
        if char == "\\":
            index += 2
            continue
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
        if line[cursor] in "\"'":
            delimiter_quote = line[cursor]
            cursor += 1
            end = line.find(delimiter_quote, cursor)
            if end == -1:
                break
            delimiter = line[cursor:end]
            cursor = end + 1
        else:
            start = cursor
            while (
                cursor < len(line)
                and not line[cursor].isspace()
                and line[cursor] not in ";|&()<>'\""
            ):
                cursor += 1
            delimiter = line[start:cursor].replace("\\", "")
        if delimiter:
            delimiters.append((delimiter, strip_tabs))
        index = max(cursor, index + 2)
    return delimiters


def _without_heredoc_bodies(command: str) -> str:
    """Replace here-document bodies with newlines before scanning operators."""
    pending: list[tuple[str, bool]] = []
    kept: list[str] = []
    for line in command.splitlines(keepends=True):
        if pending:
            delimiter, strip_tabs = pending[0]
            body_line = line.rstrip("\r\n")
            if strip_tabs:
                body_line = body_line.lstrip("\t")
            # Preserve line count without allowing literal body text to look
            # like a control operator to the simple scanner below.
            kept.append("\n" if line.endswith("\n") else "")
            if body_line == delimiter:
                pending.pop(0)
            continue
        kept.append(line)
        pending.extend(_heredoc_delimiters(line))
    return "".join(kept)


def _skip_arithmetic_expansion(command: str, index: int) -> int:
    """Skip a ``$((...))`` expression so bitwise ``&`` is not misread."""
    depth = 1
    cursor = index + 3  # after the opening ``$((``
    while cursor < len(command):
        if command.startswith("((", cursor):
            depth += 1
            cursor += 2
            continue
        if command.startswith("))", cursor):
            depth -= 1
            cursor += 2
            if depth == 0:
                return cursor
            continue
        if command[cursor] == "\\":
            cursor += 2
            continue
        cursor += 1
    return len(command)


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
        char = command[index]
        if quote:
            if char == "\\" and quote == '"':
                index += 2
                continue
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in "\"'":
            quote = char
            index += 1
            continue
        if char == "\\":
            index += 2
            continue
        if char == "#" and _starts_shell_comment(command, index):
            newline = command.find("\n", index + 1)
            index = len(command) if newline == -1 else newline + 1
            continue
        if command.startswith("$((", index):
            index = _skip_arithmetic_expansion(command, index)
            continue
        if char != "&":
            index += 1
            continue

        previous = command[index - 1] if index else ""
        following = command[index + 1] if index + 1 < len(command) else ""
        if (
            previous == "&"
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
    try:
        lexer = shlex.shlex(
            command, posix=True, punctuation_chars="|&;()<>",
        )
        lexer.whitespace_split = True
        lexer.commenters = "#"
        return list(lexer)
    except ValueError:
        # The shell will reject many malformed forms itself.  The direct
        # background-operator scan still protects the important case.
        return []


def _command_segments(tokens: list[str]) -> list[list[str]]:
    """Split tokenized shell input into simple command segments."""
    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in _COMMAND_SEPARATORS:
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
            index += 1
            continue
        if name == "env":
            index += 1
            while index < len(segment):
                token = segment[index]
                if _ASSIGNMENT_RE.fullmatch(token):
                    index += 1
                    continue
                if token.startswith("-"):
                    # ``env -u NAME`` consumes an option argument; options
                    # without one simply fall through to the next token.
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
