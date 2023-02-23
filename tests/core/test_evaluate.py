import math
from typing import Callable

from protosym.core.atom import AtomType
from protosym.core.evaluate import Evaluator
from protosym.core.tree import TreeAtom
from protosym.core.tree import TreeExpr


def test_Evaluator() -> None:
    """Test defining and using a simple Evaluator."""
    Integer = AtomType("Integer", int)
    Function = AtomType("Function", str)
    Symbol = AtomType("Symbol", str)
    one = TreeAtom(Integer(1))
    two = TreeAtom(Integer(2))
    cos = TreeAtom(Function("cos"))
    sin = TreeAtom(Function("sin"))
    Pow = TreeAtom(Function("Pow"))
    Add = TreeAtom(Function("Add"))
    x = TreeAtom(Symbol("x"))

    eval_f64 = Evaluator[float]()
    eval_f64.add_atom(Integer, float)
    eval_f64.add_op1(cos, math.cos)
    eval_f64.add_op1(sin, math.sin)
    eval_f64.add_op2(Pow, pow)
    eval_f64.add_opn(Add, math.fsum)

    test_cases: list[tuple[TreeExpr, dict[TreeExpr, float], float]] = [
        (sin(cos(one)), {}, 0.5143952585235492),
        (sin(cos(x)), {x: 1.0}, 0.5143952585235492),
        (Add(Pow(sin(one), two), Pow(cos(one), two)), {}, 1.0),
    ]

    # Test __call__ for which vals is optional
    for expr, vals, expected in test_cases:
        assert eval_f64(expr, vals) == expected
        if vals == {}:
            assert eval_f64(expr) == expected

    # Test all implementations
    eval_funcs: list[Callable[[TreeExpr, dict[TreeExpr, float]], float]] = [
        eval_f64,
        eval_f64.evaluate,
        eval_f64.eval_recursive,
        eval_f64.eval_forward,
    ]
    for expr, vals, expected in test_cases:
        for func in eval_funcs:
            assert func(expr, vals) == expected
