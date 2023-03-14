from __future__ import annotations

import math

from pytest import approx
from pytest import raises

from protosym.core.sym import Sym
from protosym.core.sym import SymAtomType
from protosym.core.sym import SymEvaluator
from protosym.core.tree import TreeAtom


def test_Sym() -> None:
    """Test defining a simple subclass of Sym."""

    class Expr(Sym):
        def __repr__(self) -> str:
            return to_str(self)

        def __call__(self, *args: Expr) -> Expr:
            args_rep = [arg.rep for arg in args]
            return Expr(self.rep(*args_rep))

    Integer = Expr.new_atom("Integer", int)
    Symbol = Expr.new_atom("Symbol", str)
    Function = Expr.new_atom("Function", str)
    cos = Function("cos")
    sin = Function("sin")
    Add = Function("Add")
    one = Integer(1)
    x = Symbol("x")

    assert str(Integer) == repr(Integer) == "Integer"

    raises(TypeError, lambda: Expr(1))  # type:ignore
    assert Expr(one.rep) is one

    assert type(Integer) is SymAtomType
    assert type(Function) is SymAtomType
    assert type(cos) is Expr
    assert type(Add) is Expr
    assert type(cos.rep) is TreeAtom

    to_str = Expr.new_evaluator("to_str", str)
    to_str.add_atom(Integer, str)
    to_str.add_atom_generic(str)
    to_str.add_op1(cos, lambda s: f"cos({s})")
    to_str.add_opn(Add, " + ".join)
    to_str.add_op_generic(lambda f, a: f"{f}({', '.join(a)})")

    assert to_str(cos(one)) == "cos(1)"
    assert to_str(Add(one, one, one)) == "1 + 1 + 1"

    # Test the generic rules
    assert to_str(sin) == "sin"
    assert to_str(sin(one)) == "sin(1)"

    assert type(to_str) == SymEvaluator
    assert repr(to_str) == repr(to_str) == "to_str"

    eval_f64 = Expr.new_evaluator("eval_f64", float)
    eval_f64.add_atom(Integer, float)
    eval_f64.add_op1(cos, math.cos)
    eval_f64.add_op2(Add, lambda a, b: a + b)

    assert eval_f64(cos(one)) == approx(0.5403023058681398)
    assert eval_f64(Add(one, one)) == 2.0
    assert eval_f64(Add(x, one), {x: -1.0}) == 0.0

    s_one = Sym.new_atom("Integer", int)(1)
    assert str(s_one) == "1"
    assert repr(s_one) == "Sym(TreeAtom(Integer(1)))"
