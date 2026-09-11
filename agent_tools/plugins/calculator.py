"""Plugin: safe arithmetic and math-function evaluation.

Small and mid-size models routinely botch multi-step arithmetic
(percentages, unit conversion, exponent chains). This tool evaluates a
pure math expression with Python's ``ast`` module instead of ``eval``:
only numeric literals, arithmetic operators, parentheses, and an
explicit function/constant whitelist are accepted, so the expression
can never touch the host.
"""

from __future__ import annotations

import ast
import math
import operator as op
from typing import Any, Callable, ClassVar

from ..base import AgentTool, require_nonempty_string_arg

_MAX_EXPRESSION_CHARS = 1_000
_MAX_EXPONENT = 10_000
_MAX_FACTORIAL_ARG = 500


class _CalcError(Exception):
    """A model-facing evaluation refusal."""


def _guarded_pow(base: Any, exponent: Any) -> Any:
    """``**`` with magnitude guards so ``9**9**9`` cannot wedge the host."""
    if isinstance(base, (int, float)) and isinstance(exponent, (int, float)):
        if abs(exponent) > _MAX_EXPONENT:
            raise _CalcError(
                f"exponent magnitude is capped at {_MAX_EXPONENT:,}",
            )
        magnitude = abs(base)
        if magnitude > 1 and exponent * math.log10(magnitude) > 50_000:
            raise _CalcError(
                "result would exceed 50,000 digits; reduce the operands",
            )
    try:
        return base ** exponent
    except OverflowError as exc:
        raise _CalcError(f"result too large: {exc}") from exc


def _guarded_factorial(value: Any) -> int:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int) or isinstance(value, bool):
        raise _CalcError("factorial needs a non-negative integer")
    if not 0 <= value <= _MAX_FACTORIAL_ARG:
        raise _CalcError(
            f"factorial argument is capped at {_MAX_FACTORIAL_ARG}",
        )
    return math.factorial(value)


_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: _guarded_pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}

_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "degrees": math.degrees,
    "radians": math.radians,
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    "fabs": math.fabs,
    "hypot": math.hypot,
    "gcd": math.gcd,
    "factorial": _guarded_factorial,
    "round": round,
    "min": min,
    "max": max,
}

_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "tau": math.tau,
    "e": math.e,
    "phi": (1 + math.sqrt(5)) / 2,
}


def _evaluate(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(
            node.value, (int, float),
        ):
            raise _CalcError(
                f"only numeric literals are allowed, not {node.value!r}",
            )
        return node.value
    if isinstance(node, ast.BinOp):
        operation = _BIN_OPS.get(type(node.op))
        if operation is None:
            raise _CalcError(
                f"operator '{type(node.op).__name__}' is not supported",
            )
        return operation(_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp):
        unary_operation = _UNARY_OPS.get(type(node.op))
        if unary_operation is None:
            raise _CalcError(
                f"unary operator '{type(node.op).__name__}' is not supported",
            )
        return unary_operation(_evaluate(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise _CalcError(
            f"unknown name '{node.id}'; constants: "
            + ", ".join(sorted(_CONSTANTS)),
        )
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise _CalcError("only direct math-function calls are supported")
        name = node.func.id
        if name in _CONSTANTS:
            raise _CalcError(f"'{name}' is a constant, not a function")
        function = _FUNCTIONS.get(name)
        if function is None:
            raise _CalcError(
                f"unknown function '{name}'; supported: "
                + ", ".join(sorted(_FUNCTIONS)),
            )
        if node.keywords:
            raise _CalcError("keyword arguments are not supported")
        arguments = [_evaluate(arg) for arg in node.args]
        return function(*arguments)
    raise _CalcError(
        f"syntax '{type(node).__name__}' is not supported; use plain"
        " arithmetic like (2 + 3) * sqrt(16) / 7",
    )


def _format(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _CalcError("result is not finite (overflow or domain error)")
        if value.is_integer() and abs(value) < 1e16:
            return str(int(value))
        return repr(value)
    return str(value)


class CalculatorTool(AgentTool):
    name = "calculator"
    order = 20  # utility tools group
    description = "Exact math."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "expression": {"type": "string"},
        },
        "required": ["expression"],
    }

    async def run(self, workspace_path: str, args: dict) -> str:
        del workspace_path  # pure computation; no workspace access
        expression, error = require_nonempty_string_arg(
            args, "expression", strip=True,
        )
        if error:
            return error
        assert expression is not None  # non-None once error is None
        if len(expression) > _MAX_EXPRESSION_CHARS:
            return (
                f"Error: expression exceeds {_MAX_EXPRESSION_CHARS}"
                " characters"
            )

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            return f"Error: invalid expression: {exc.msg}"
        except (ValueError, MemoryError, RecursionError) as exc:
            return f"Error: could not parse expression: {exc}"

        try:
            value = _evaluate(tree)
            return _format(value)
        except _CalcError as exc:
            return f"Error: {exc}"
        except ZeroDivisionError:
            return "Error: division by zero"
        except (OverflowError, ValueError) as exc:
            return f"Error: {exc}"
        except RecursionError:
            return "Error: expression is nested too deeply"
        except TypeError as exc:
            return f"Error: invalid operands: {exc}"

    def summarize(self, args: dict) -> str:
        expression = (
            args.get("expression", "") if isinstance(args, dict) else ""
        )
        if not isinstance(expression, str):
            expression = str(expression)
        return f"calc: {expression[:120]}" + (
            "… [clipped]" if len(expression) > 120 else ""
        )


if __name__ == "__main__":
    CalculatorTool.run_as_script()
