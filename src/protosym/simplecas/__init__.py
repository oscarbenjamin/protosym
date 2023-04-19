"""Demonstration of putting together a simple CAS."""
from __future__ import annotations

import protosym.simplecas.functions  # noqa
from .expr import a
from .expr import Add
from .expr import b
from .expr import bin_expand
from .expr import cos
from .expr import diff
from .expr import Expr
from .expr import expressify
from .expr import f
from .expr import Function
from .expr import g
from .expr import Integer
from .expr import List
from .expr import Mul
from .expr import negone
from .expr import one
from .expr import Pow
from .expr import sin
from .expr import Symbol
from .expr import x
from .expr import y
from .expr import zero
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
