from pytest import approx

from protosym.core.f64 import f64_cos


def test_f64() -> None:
    """Basic test for the f64 module."""
    expr = f64_cos(1)
    assert str(expr) == "cos(1)"
    assert expr.evaluate() == approx(0.5403023058681398)
