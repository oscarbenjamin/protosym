"""Conversions to and from SymPy expressions.

These are defined in their own module so that SymPy will not imported if it is
not needed.
"""
from __future__ import annotations

from typing import Any

import sympy
from sympy.core.function import AppliedUndef

from protosym.core.sym import HeadOp, HeadRule, PyFunc1, PyOp1, PyOp2, PyOpN, star
from protosym.simplecas import (
    Add,
    Expr,
    Function,
    Integer,
    List,
    Matrix,
    Mul,
    Pow,
    Symbol,
    a,
    b,
    cos,
    sin,
)

eval_to_sympy = Expr.new_evaluator("to_sympy", sympy.Basic)

sympy_integer = PyFunc1[int, sympy.Basic](sympy.Integer)
sympy_symbol = PyFunc1[str, sympy.Basic](sympy.Symbol)
sympy_function = PyFunc1[str, sympy.Basic](sympy.Function)  # pyright: ignore
sympy_add = PyOpN[sympy.Basic](lambda a: sympy.Add(*a))
sympy_mul = PyOpN[sympy.Basic](lambda a: sympy.Mul(*a))
sympy_pow = PyOp2[sympy.Basic](sympy.Pow)
sympy_sin = PyOp1[sympy.Basic](sympy.sin)  # pyright: ignore
sympy_cos = PyOp1[sympy.Basic](sympy.cos)  # pyright: ignore
sympy_tuple = PyOpN[sympy.Basic](lambda a: sympy.Tuple(*a))  # pyright: ignore
sympy_undef_call = HeadOp[sympy.Basic](
    lambda a, b: sympy.Function(str(a))(*b)  # pyright: ignore
)

eval_to_sympy[Integer[a]] = sympy_integer(a)
eval_to_sympy[Symbol[a]] = sympy_symbol(a)
eval_to_sympy[Function[a]] = sympy_function(a)
eval_to_sympy[Add(star(a))] = sympy_add(a)
eval_to_sympy[Mul(star(a))] = sympy_mul(a)
eval_to_sympy[a**b] = sympy_pow(a, b)
eval_to_sympy[sin(a)] = sympy_sin(a)
eval_to_sympy[cos(a)] = sympy_cos(a)
eval_to_sympy[List(star(a))] = sympy_tuple(a)
eval_to_sympy[HeadRule(a, b)] = sympy_undef_call(a, b)


def to_sympy(expr: Expr) -> Any:
    """Convert ``Expr`` to a SymPy expression."""
    return eval_to_sympy(expr)


def to_sympy_matrix(mat: Matrix) -> Any:
    """Convert a simplecas Matrix to a SymPy Matrix."""
    elements_sympy = to_sympy(mat.elements_graph).args
    mat_sympy = sympy.zeros(mat.nrows, mat.ncols)
    for (i, j), n in mat.entrymap.items():
        mat_sympy[i, j] = elements_sympy[n]
    return mat_sympy


def from_sympy(expr: sympy.Basic) -> Expr:
    """Convert a SymPy expression to ``Expr``."""
    return _from_sympy_cache(expr, {})


def from_sympy_matrix(mat: sympy.Matrix) -> Matrix:
    """Convert a SymPy Matrix to a simplecas Matrix."""
    dok = mat.todok()
    elements_sympy = []
    entrymap = {}
    for n, (key, sympy_expr) in enumerate(dok.items()):
        entrymap[key] = n
        elements_sympy.append(sympy_expr)
    elements = list(from_sympy(sympy.Tuple(*elements_sympy)).args)
    return Matrix._new(mat.rows, mat.cols, elements, entrymap)


def _from_sympy_cache(expr: sympy.Basic, cache: dict[sympy.Basic, Expr]) -> Expr:
    ret = cache.get(expr)
    if ret is not None:
        return ret
    elif expr.args:
        ret = _from_sympy_cache_args(expr, sympy, cache)
    elif isinstance(expr, sympy.Integer):
        ret = Integer(expr.p)  # pyright: ignore
    elif isinstance(expr, sympy.Symbol):
        ret = Symbol(expr.name)  # pyright: ignore
    else:
        raise NotImplementedError("Cannot convert " + type(expr).__name__)
    cache[expr] = ret
    return ret


def _from_sympy_cache_args(expr: Any, sympy: Any, cache: dict[Any, Expr]) -> Expr:
    args = [_from_sympy_cache(arg, cache) for arg in expr.args]
    if expr.is_Add:
        return Add(*args)
    elif expr.is_Mul:
        return Mul(*args)
    elif expr.is_Pow:
        return Pow(*args)
    elif isinstance(expr, sympy.sin):
        return sin(*args)
    elif isinstance(expr, sympy.cos):
        return cos(*args)
    elif isinstance(expr, sympy.Tuple):
        return List(*args)
    elif isinstance(expr, AppliedUndef):
        return Function(expr.name)(*args)  # pyright: ignore
    else:
        raise NotImplementedError("Cannot convert " + type(expr).__name__)
