from pytest import raises


def test_simplecas() -> None:
    """Test basic operations with simplecas."""
    from protosym.simplecas import x, y, sin, cos, eval_f64, Integer, Expr

    raises(TypeError, lambda: Expr([]))  # type: ignore

    assert str(Integer) == "Integer"
    assert str(x) == "x"
    assert str(y) == "y"
    assert str(sin) == "sin"

    expr = sin(cos(x))
    assert str(expr) == "sin(cos(x))"

    assert eval_f64(expr.rep, {x.rep: 1.0}) == 0.5143952585235492
