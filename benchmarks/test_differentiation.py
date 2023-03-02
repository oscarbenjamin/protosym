"""Benchmarks for differentiation using ProtoSym and SimpleCAS.

Differentiation using ProtoSym and SimpleCAS is currently benchmarked against
SymPy's and SymEngine's symbolic differentiation.

"""
from typing import Callable
from typing import TypeVar

import pytest
import symengine
import sympy

from protosym import simplecas


ExprType = TypeVar("ExprType")
Fixture = Callable[..., ExprType]


@pytest.mark.benchmark(group="differentiate nested sine first derivative")
class TestNestedSineFirstDerivative:
    """Differentiate ``sin(sin(sin(sin(sin(sin(x))))))`` w.r.t. ``x`` once."""

    @staticmethod
    def test_simplecas(benchmark: Fixture[simplecas.Expr]) -> None:
        """Differentiate using SimpleCAS."""
        x = simplecas.Symbol("x")
        sin = simplecas.sin
        expr = sin(sin(sin(sin(sin(sin(x))))))
        benchmark(expr.diff, x)

    @staticmethod
    def test_sympy(benchmark: Fixture[sympy.Expr]) -> None:
        """Differentiate using SymPy."""
        x = sympy.Symbol("x")
        sin = sympy.sin
        expr = sin(sin(sin(sin(sin(sin(x))))))
        benchmark(expr.diff, x)

    @staticmethod
    def test_symengine(benchmark: Fixture[symengine.Expr]) -> None:
        """Differentiate using SymEngine."""
        x = symengine.Symbol("x")
        sin = symengine.sin
        expr = sin(sin(sin(sin(sin(sin(x))))))
        benchmark(expr.diff, x)


@pytest.mark.benchmark(group="differentiate nested sine tenth derivative")
class TestNestedSineTenthDerivative:
    """Differentiate ``sin(sin(sin(sin(sin(sin(x))))))`` w.r.t. ``x`` ten times."""

    @staticmethod
    def test_simplecas(benchmark: Fixture[simplecas.Expr]) -> None:
        """Differentiate using SimpleCAS."""
        x = simplecas.Symbol("x")
        sin = simplecas.sin
        expr = sin(sin(sin(sin(sin(sin(x))))))
        benchmark(expr.diff, x, 10)

    @staticmethod
    @pytest.mark.skip(reason="too slow for benchmarking")
    def test_sympy(benchmark: Fixture[sympy.Expr]) -> None:
        """Differentiate using SymPy."""
        x = sympy.Symbol("x")
        sin = sympy.sin
        expr = sin(sin(sin(sin(sin(sin(x))))))
        benchmark(expr.diff, x, 10)

    @staticmethod
    def test_symengine(benchmark: Fixture[symengine.Expr]) -> None:
        """Differentiate using SymEngine."""
        x = symengine.Symbol("x")
        sin = symengine.sin
        expr = sin(sin(sin(sin(sin(sin(x))))))
        benchmark(expr.diff, x, 10)
