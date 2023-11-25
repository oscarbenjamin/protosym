"""Benchmarks for differentiation using ProtoSym and SimpleCAS.

Differentiation using ProtoSym and SimpleCAS is currently benchmarked against
SymPy's and SymEngine's symbolic differentiation.

"""
from typing import Callable, TypeVar

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
        result = benchmark(expr.diff, x)
        assert result.eval_f64({x: 1.0}) == pytest.approx(0.13877489681259086)

    @staticmethod
    def test_sympy(benchmark: Fixture[sympy.Expr]) -> None:
        """Differentiate using SymPy."""
        x = sympy.Symbol("x")
        sin = sympy.sin
        expr = sin(sin(sin(sin(sin(sin(x))))))
        result = benchmark(expr.diff, x)
        assert result.evalf(subs={x: 1.0}) == pytest.approx(0.13877489681259086)

    @staticmethod
    def test_symengine(benchmark: Fixture[symengine.Expr]) -> None:
        """Differentiate using SymEngine."""
        x = symengine.Symbol("x")
        sin = symengine.sin
        expr = sin(sin(sin(sin(sin(sin(x))))))
        result = benchmark(expr.diff, x)
        assert float(result.subs(x, 1.0).evalf()) == pytest.approx(0.13877489681259086)


@pytest.mark.benchmark(group="differentiate nested sine tenth derivative")
class TestNestedSineTenthDerivative:
    """Differentiate ``sin(sin(sin(sin(sin(sin(x))))))`` w.r.t. ``x`` ten times."""

    @staticmethod
    def test_simplecas(benchmark: Fixture[simplecas.Expr]) -> None:
        """Differentiate using SimpleCAS."""
        x = simplecas.Symbol("x")
        sin = simplecas.sin
        expr = sin(sin(sin(sin(sin(sin(x))))))
        result = benchmark(expr.diff, x, 10)
        assert result.eval_f64({x: 1.0}) == pytest.approx(11560.616267596966)

    @staticmethod
    @pytest.mark.skip(reason="too slow for benchmarking")
    def test_sympy(benchmark: Fixture[sympy.Expr]) -> None:
        """Differentiate using SymPy."""
        x = sympy.Symbol("x")
        sin = sympy.sin
        expr = sin(sin(sin(sin(sin(sin(x))))))
        result = benchmark(expr.diff, x, 10)
        assert result.evalf(subs={x: 1.0}) == pytest.approx(11560.616267596966)

    @staticmethod
    def test_symengine(benchmark: Fixture[symengine.Expr]) -> None:
        """Differentiate using SymEngine."""
        x = symengine.Symbol("x")
        sin = symengine.sin
        expr = sin(sin(sin(sin(sin(sin(x))))))
        result = benchmark(expr.diff, x, 10)
        assert float(result.subs(x, 1.0).evalf()) == pytest.approx(11560.616267596966)
