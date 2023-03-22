"""Conversions to and from SymPy expressions.

These are defined in their own module so that SymPy will not imported if it is
not needed.
"""
from __future__ import annotations

from typing import Any

import sympy

from protosym.core.sym import PyFunc1
from protosym.core.sym import PyOp1
from protosym.core.sym import PyOp2
from protosym.core.sym import PyOpN
from protosym.core.sym import star
from protosym.simplecas import a
from protosym.simplecas import Add
from protosym.simplecas import b
from protosym.simplecas import cos
from protosym.simplecas import Expr
from protosym.simplecas import Function
from protosym.simplecas import Integer
from protosym.simplecas import List
from protosym.simplecas import Matrix
from protosym.simplecas import Mul
from protosym.simplecas import Pow
from protosym.simplecas import sin
from protosym.simplecas import Symbol


eval_to_sympy = Expr.new_evaluator("to_sympy", object)

eval_to_sympy[Integer[a]] = PyFunc1(sympy.Integer)(a)
eval_to_sympy[Symbol[a]] = PyFunc1(sympy.Symbol)(a)
eval_to_sympy[Function[a]] = PyFunc1(sympy.Function)(a)
eval_to_sympy[Add(star(a))] = PyOpN[Any](lambda a: sympy.Add(*a))(a)
eval_to_sympy[Mul(star(a))] = PyOpN[Any](lambda a: sympy.Mul(*a))(a)
eval_to_sympy[a**b] = PyOp2(sympy.Pow)(a, b)
eval_to_sympy[sin(a)] = PyOp1(sympy.sin)(a)
eval_to_sympy[cos(a)] = PyOp1(sympy.cos)(a)
eval_to_sympy[List(star(a))] = PyOpN(
    lambda args: sympy.Tuple(*args)  # type:ignore
)(
    a
)


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


def from_sympy(expr: Any) -> Expr:
    """Convert a SymPy expression to ``Expr``."""
    return _from_sympy_cache(expr, sympy, {})


def from_sympy_matrix(mat: Any) -> Matrix:
    """Convert a SymPy Matrix to a simplecas Matrix."""
    dok = mat.todok()
    elements_sympy = []
    entrymap = {}
    for n, (key, sympy_expr) in enumerate(dok.items()):
        entrymap[key] = n
        elements_sympy.append(sympy_expr)
    elements = list(from_sympy(sympy.Tuple(*elements_sympy)).args)
    return Matrix._new(mat.rows, mat.cols, elements, entrymap)


def _from_sympy_cache(expr: Any, sympy: Any, cache: dict[Any, Expr]) -> Expr:
    ret = cache.get(expr)
    if ret is not None:
        return ret
    elif expr.args:
        ret = _from_sympy_cache_args(expr, sympy, cache)
    elif expr.is_Integer:
        ret = Integer(expr.p)
    elif expr.is_Symbol:
        ret = Symbol(expr.name)
    else:
        raise NotImplementedError("Cannot convert " + type(expr).__name__)
    cache[expr] = ret
    return ret


def _from_sympy_cache_args(expr: Any, sympy: Any, cache: dict[Any, Expr]) -> Expr:
    args = [_from_sympy_cache(arg, sympy, cache) for arg in expr.args]
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
    else:
        raise NotImplementedError("Cannot convert " + type(expr).__name__)
