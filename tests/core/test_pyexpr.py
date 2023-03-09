import math
from typing import Any
from typing import Callable

from pytest import approx
from pytest import raises

from protosym.core.pyexpr import PyExpr
from protosym.core.tree import TreeExpr


py_str = PyExpr.function(str)
py_cos = PyExpr.function(math.cos)
x = PyExpr.symbol("x")
y = PyExpr.symbol("y")


def test_pyexpr() -> None:
    """Test basic features of PyExpr."""
    assert all(isinstance(obj, PyExpr) for obj in (py_str, py_cos, x, y))

    raises(TypeError, lambda: PyExpr(1))  # type: ignore [arg-type]

    expr = py_cos(x) + y
    expr2 = py_cos(x) + y
    expr3 = py_cos(x) + y + y

    assert expr is expr2
    assert (expr == expr2) is True
    assert (expr != expr2) is False
    assert (expr2 == expr3) is False
    assert (expr2 != expr3) is True

    assert str(expr) == "(cos(x) + y)"
    assert expr.evaluate({x: 1, y: 1}) == approx(1.5403023058681398)

    assert str(expr + x) == "((cos(x) + y) + x)"


def test_pyexpr_operations() -> None:
    """Test special operations supported by PyExpr."""
    from protosym.core.pyexpr import PyCall, PyAdd, PyMul, PySub, PyTrueDiv, PyPow

    assert py_cos(x) == PyExpr(PyCall(py_cos.rep, x.rep))
    assert py_str(x) == PyExpr(PyCall(py_str.rep, x.rep))

    operations: list[tuple[TreeExpr, Callable[[Any, Any], Any]]] = [
        (PyAdd, lambda a, b: a + b),
        (PySub, lambda a, b: a - b),
        (PyMul, lambda a, b: a * b),
        (PyTrueDiv, lambda a, b: a / b),
        (PyPow, lambda a, b: a**b),
    ]

    one = PyExpr.pyobject(1)

    for PyOp, func in operations:
        assert func(x, y) == PyExpr(PyOp(x.rep, y.rep))
        assert func(x, 1) == PyExpr(PyOp(x.rep, one.rep))
        assert func(1, y) == PyExpr(PyOp(one.rep, y.rep))


def test_pyexpr_repr() -> None:
    """Test string representation of PyExpr."""
    examples = [
        (py_cos(x), "cos(x)"),
        (py_str(x), "str(x)"),
        (x + y, "(x + y)"),
        (x - y, "(x - y)"),
        (x * y, "(x*y)"),
        (x / y, "(x/y)"),
        (x**y, "(x**y)"),
        (1 / (x + y), "(1/(x + y))"),
    ]
    for expr, strexpr in examples:
        assert repr(expr) == str(expr) == strexpr


def test_pyexpr_evaluate() -> None:
    """Test evaluation of special operations with PyExpr."""
    assert py_str(123).evaluate() == "123"

    two = PyExpr.pyobject(2)
    three = PyExpr.pyobject(3)

    assert (two + three).evaluate() == 5
    assert (two - three).evaluate() == -1
    assert (two * three).evaluate() == 6
    assert (three / two).evaluate() == 1.5
    assert (three**two).evaluate() == 9

    cube = PyExpr.function(lambda x: x**3)
    assert cube(3).evaluate() == 27
    assert cube(three).evaluate() == 27


def test_pyexpr_as_function_to_code() -> None:
    """Test to_code and compile for PyExpr."""
    x = PyExpr.symbol("x")
    y = PyExpr.symbol("y")

    add = (x + y).as_function(x, y)
    assert str(add) == "SymFunction((x, y), (x + y))"
    raises(TypeError, lambda: add(1))

    assert add(1, 2) == 3
    code, namespace = add.to_code("add")
    assert code == """def add(x, y):\n    x2 = x + y\n    return x2"""
    assert namespace == {}

    py_sqrt = PyExpr.function(math.sqrt)
    f = (x + py_sqrt(y)).as_function(x, y)
    assert f(1, 2) == approx(2.414213562373095)
    code, namespace = f.to_code("f")
    assert (
        code
        == """\
def f(x, y):
    x3 = x1(y)
    x4 = x + x3
    return x4\
"""
    )
    assert namespace == {"x1": math.sqrt}

    fc = f.compile("f")
    assert fc(1, 2) == approx(2.414213562373095)

    f_add1 = (1 + x).as_function(x)
    code, namespace = f_add1.to_code("f")
    assert (
        code
        == """\
def f(x):
    x2 = 1 + x
    return x2\
"""
    )
    assert namespace == {}
