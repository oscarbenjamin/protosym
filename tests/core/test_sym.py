from __future__ import annotations

import math
from typing import Any

from protosym.core.exceptions import BadRuleError
from protosym.core.sym import (
    AtomFunc,
    AtomRule,
    HeadOp,
    HeadRule,
    PyFunc1,
    PyOp1,
    PyOp2,
    PyOpN,
    Sym,
    SymAtomType,
    SymDifferentiator,
    SymEvaluator,
    star,
)
from protosym.core.tree import Tree
from pytest import approx, raises


class Expr(Sym):
    """Simple Sym subclass."""

    # Really repr should use to_str but we test that separately.
    def __repr__(self) -> str:
        """Pretty print with recursion."""
        if not self.args:
            return str(self.rep)
        else:
            argstr = ", ".join(map(repr, self.args))
            return f"{self.head}({argstr})"

    def __call__(self, *args: Expr) -> Expr:
        """Construct a new Expr."""
        args_rep = [arg.rep for arg in args]
        return Expr(self.rep(*args_rep))


def _make_atoms() -> (
    tuple[
        SymAtomType[Expr, int],
        SymAtomType[Expr, str],
        Expr,
        Expr,
        Expr,
        Expr,
        Expr,
        Expr,
        Expr,
    ]
):
    """Set up a Sym subclass and create some atoms etc."""
    Integer = Expr.new_atom("Integer", int)
    Function = Expr.new_atom("Function", str)
    Symbol = Expr.new_atom("Symbol", str)
    cos = Function("cos")
    sin = Function("sin")
    Add = Function("Add")
    one = Integer(1)
    a = Expr.new_wild("a")
    b = Expr.new_wild("b")
    x = Symbol("x")

    return Integer, Function, cos, sin, Add, one, a, b, x


def test_Sym() -> None:
    """Test a few properties of Sym separately."""
    s_one = Sym.new_atom("Integer", int)(1)
    assert str(s_one) == "1"
    assert repr(s_one) == "Sym(Tr(Integer(1)))"


def test_Sym_str() -> None:
    """Test str of Sym-related objects."""
    Integer, Function, cos, sin, Add, one, a, b, x = _make_atoms()

    obj_str = [
        (Integer, "Integer"),
        (Function, "Function"),
        (cos, "cos"),
        (sin, "sin"),
        (Add, "Add"),
        (one, "1"),
        (cos(one), "cos(1)"),
        (Add(one, one), "Add(1, 1)"),
        (a, "a"),
        (b, "b"),
    ]
    for obj, objstr in obj_str:
        assert str(obj) == repr(obj) == objstr


def test_Sym_types() -> None:
    """Test types of Sym-related objects."""
    Integer, Function, cos, sin, Add, one, a, b, x = _make_atoms()

    raises(TypeError, lambda: Expr(1))  # type:ignore
    assert Expr(one.rep) is one
    assert type(Integer) is SymAtomType
    assert type(Function) is SymAtomType
    assert type(cos) is Expr
    assert type(Add) is Expr
    assert type(cos.rep) is Tree
    assert type(a) == type(b) == Expr


def test_Sym_evaluator() -> None:
    """Test creating a str-evaluator for Sym."""
    Integer, Function, cos, sin, Add, one, a, b, x = _make_atoms()

    to_str = Expr.new_evaluator("to_str", str)
    assert type(to_str) == SymEvaluator
    assert repr(to_str) == repr(to_str) == "to_str"

    to_str[Integer[a]] = PyFunc1[int, str](str)(a)
    to_str[AtomRule[a]] = AtomFunc(str)(a)
    to_str[cos(a)] = PyOp1(lambda s: f"cos({s})")(a)
    to_str[Add(star(a))] = PyOpN(" + ".join)(a)
    to_str[HeadRule(a, b)] = HeadOp(lambda f, a: f"{f}({', '.join(a)})")(a, b)

    assert to_str(cos(one)) == "cos(1)"
    assert to_str(Add(one, one, one)) == "1 + 1 + 1"

    # Test the generic rules
    assert to_str(sin) == "sin"
    assert to_str(sin(one)) == "sin(1)"


def test_Sym_evaluator_float() -> None:
    """Test creating a str-evaluator for float."""
    Integer, Function, cos, sin, Add, one, a, b, x = _make_atoms()

    x = Expr.new_atom("Symbol", str)("x")

    eval_f64 = Expr.new_evaluator("eval_f64", float)
    eval_f64[Integer[a]] = PyFunc1[int, float](float)(a)
    eval_f64[cos(a)] = PyOp1(math.cos)(a)
    eval_f64[Add(a, b)] = PyOp2[float](lambda a, b: a + b)(a, b)

    assert eval_f64(cos(one)) == approx(0.5403023058681398)
    assert eval_f64(Add(one, one)) == 2.0
    assert eval_f64(Add(x, one), {x: -1.0}) == 0.0


def test_Sym_evaluator_bad_rules() -> None:
    """Test SymEvaluator bad rule handling."""
    Integer, Function, cos, sin, Add, one, a, b, x = _make_atoms()

    eval_f64 = Expr.new_evaluator("eval_f64", float)

    bad_examples = [
        (Integer[a], PyOp1(math.cos)(a)),
        (cos(a), PyFunc1(math.cos)(a)),
        (AtomRule[a], PyOp1(math.cos)(a)),
        (HeadRule(a, b), PyOp2(math.atan2)(a, b)),
        (Integer[a], PyFunc1(int)),
        (Integer[a], PyFunc1(int)(b)),
        (cos(a), PyOp2(math.atan2)(a, b)),
        (AtomRule[a], PyOp2(math.atan2)(a, b)),
        (HeadRule(a, b), PyOp1(math.cos)(a)),
        (Add(a), PyOpN[float](sum)(a)),
    ]

    def set_bad_rule(k: Any, v: Any) -> None:
        eval_f64[k] = v

    for key, value in bad_examples:
        raises(BadRuleError, lambda: set_bad_rule(key, value))

    def set_bad_op() -> None:
        eval_f64.add_op(cos, lambda x: x)  # type:ignore

    raises(BadRuleError, set_bad_op)


def test_SymDifferentiator() -> None:
    """Test the SymDifferentiator wrapper for Differentiator."""
    Integer, Function, cos, sin, Add, one, a, b, x = _make_atoms()
    zero = Integer(0)
    negone = Integer(-1)
    Mul = Function("Mul")
    Vector = Function("Vector")
    Expr = type(one)
    Symbol = Expr.new_atom("Symbol", str)
    y = Symbol("y")

    diff = SymDifferentiator(Expr, zero=zero, one=one, add=Add, mul=Mul)

    assert diff(x, x) == one
    assert diff(x, x, 2) == zero
    assert diff(x, y) == zero

    diff.add_distributive_rule(Vector)

    assert diff(Vector(x, y), x) == Vector(one, zero)

    diff[sin(a), a] = cos(a)
    diff[cos(a), a] = Mul(negone, sin(a))

    assert diff(sin(x), x) == cos(x)
    assert diff(sin(sin(x)), x) == Mul(cos(sin(x)), cos(x))

    def set_bad1() -> None:
        diff[sin(a), a, b] = cos(a)  # type: ignore

    def set_bad2() -> None:
        diff[sin(a, a), a] = cos(a)

    raises(TypeError, set_bad1)
    raises(TypeError, set_bad2)
