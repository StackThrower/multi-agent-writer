"""Safe arithmetic tool for the analyst agent.

LLMs are bad at arithmetic, so numbers go through a tool. Never use `eval()` on
model output: here only a whitelisted subset of the Python AST is evaluated.
"""

from __future__ import annotations

import ast
import math
import operator as op

from langchain.tools import tool

_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}
_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}
_NAMES = {"pi": math.pi, "e": math.e}
_MAX_POWER = 1_000


def safe_eval(expression: str) -> float:
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.Name) and node.id in _NAMES:
            return _NAMES[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POWER:
                raise ValueError("exponent too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
        raise ValueError(f"unsupported expression element: {ast.dump(node)[:60]}")

    return _eval(ast.parse(expression, mode="eval"))


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '(1250 - 980) / 980 * 100'.

    Supports + - * / // % **, parentheses, pi, e and sqrt, log, log10, exp, abs,
    round, min, max. Use it for every calculation instead of doing math in your head.
    """
    try:
        return str(safe_eval(expression))
    except Exception as exc:
        return f"Error: {exc}"
