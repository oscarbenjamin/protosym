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
    eval_f64.add_opN(Add, math.fsum)

    assert eval_f64(sin(cos(one))) == 0.5143952585235492
    assert eval_f64(sin(cos(x)), {x: 1.0}) == 0.5143952585235492
    assert eval_f64(Add(Pow(sin(one), two), Pow(cos(one), two))) == 1.0
