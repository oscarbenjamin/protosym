from __future__ import annotations

import math
from typing import Any

from pytest import approx
from pytest import raises

from protosym.core.sym import AtomFunc
from protosym.core.sym import AtomRule
from protosym.core.sym import HeadOp
from protosym.core.sym import HeadRule
from protosym.core.sym import PyFunc1
from protosym.core.sym import PyOp1
from protosym.core.sym import PyOp2
from protosym.core.sym import PyOpN
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

    a = Expr.new_wild("a")
    b = Expr.new_wild("b")
    assert type(a) == type(b) == Expr

    to_str = Expr.new_evaluator("to_str", str)
    to_str[Integer[a]] = PyFunc1[int, str](str)(a)
    to_str[AtomRule[a]] = AtomFunc(str)(a)
    to_str[cos(a)] = PyOp1(lambda s: f"cos({s})")(a)
    to_str[Add(a)] = PyOpN(" + ".join)(a)
    to_str[HeadRule(a, b)] = HeadOp(lambda f, a: f"{f}({', '.join(a)})")(a, b)

    assert to_str(cos(one)) == "cos(1)"
    assert to_str(Add(one, one, one)) == "1 + 1 + 1"

    # Test the generic rules
    assert to_str(sin) == "sin"
    assert to_str(sin(one)) == "sin(1)"

    assert type(to_str) == SymEvaluator
    assert repr(to_str) == repr(to_str) == "to_str"

    eval_f64 = Expr.new_evaluator("eval_f64", float)
    eval_f64[Integer[a]] = PyFunc1[int, float](float)(a)
    eval_f64[cos(a)] = PyOp1(math.cos)(a)
    eval_f64[Add(a, b)] = PyOp2[float](lambda a, b: a + b)(a, b)

    assert eval_f64(cos(one)) == approx(0.5403023058681398)
    assert eval_f64(Add(one, one)) == 2.0
    assert eval_f64(Add(x, one), {x: -1.0}) == 0.0

    s_one = Sym.new_atom("Integer", int)(1)
    assert str(s_one) == "1"
    assert repr(s_one) == "Sym(TreeAtom(Integer(1)))"

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
    ]

    def set_bad_rule(k: Any, v: Any) -> None:
        eval_f64[k] = v

    for key, value in bad_examples:
        raises(TypeError, lambda: set_bad_rule(key, value))

    def set_bad_op() -> None:
        eval_f64.add_op(cos, lambda x: x)  # type:ignore

    raises(TypeError, set_bad_op)
