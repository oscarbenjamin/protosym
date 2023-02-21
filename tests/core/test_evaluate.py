import math

from protosym.core.atom import AtomType
from protosym.core.evaluate import Evaluator
from protosym.core.tree import TreeAtom


def test_Evaluator() -> None:
    """Test defning and using a simple Evaluator."""
    Integer = AtomType("Integer", int)
    Function = AtomType("Function", str)
    Symbol = AtomType("Symbol", str)
    one = TreeAtom(Integer(1))
    cos = TreeAtom(Function("cos"))
    sin = TreeAtom(Function("sin"))
    x = TreeAtom(Symbol("x"))

    eval_f64 = Evaluator[float]()
    eval_f64.add_atom(Integer, float)
    eval_f64.add_operation(cos, math.cos)
    eval_f64.add_operation(sin, math.sin)

    assert eval_f64(sin(cos(one))) == 0.5143952585235492
    assert eval_f64(sin(cos(x)), {x: 1.0}) == 0.5143952585235492
    assert eval_f64(sin(one)) ** 2 + eval_f64(cos(one)) ** 2 == 1.0
