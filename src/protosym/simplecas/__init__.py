"""Demonstration of putting together a simple CAS."""
from __future__ import annotations

import protosym.simplecas.functions  # noqa

from .expr import (
    Add,
    Expr,
    Function,
    Integer,
    List,
    Mul,
    Pow,
    Symbol,
    a,
    b,
    bin_expand,
    cos,
    diff,
    expressify,
    f,
    g,
    negone,
    one,
    sin,
    x,
    y,
    zero,
)
from .lambdification import lambdify
from .matrix import Matrix

__all__ = [
    "expressify",
    "diff",
    "Expr",
    "Matrix",
    "Function",
    "Integer",
    "List",
    "Symbol",
    "Add",
    "Mul",
    "Pow",
    "one",
    "zero",
    "negone",
    "sin",
    "cos",
    "a",
    "b",
    "f",
    "g",
    "x",
    "y",
    "bin_expand",
    "lambdify",
]
