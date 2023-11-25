from protosym.core.atom import AtomType
from protosym.core.differentiate import DiffProperties, diff_forward
from protosym.core.tree import SubsFunc, Tr


def test_core_differentiate() -> None:
    """Test elementary differentiation routines."""
    Integer = AtomType("Integer", int)
    Function = AtomType("Function", str)
    Symbol = AtomType("Symbol", str)

    zero = Tr(Integer(0))
    one = Tr(Integer(1))
    negone = Tr(Integer(-1))

    sin = Tr(Function("sin"))
    cos = Tr(Function("cos"))
    Vector = Tr(Function("Vector"))
    Add = Tr(Function("Add"))
    Mul = Tr(Function("Mul"))
    Pow = Tr(Function("Pow"))

    x = Tr(Symbol("x"))
    y = Tr(Symbol("y"))

    prop = DiffProperties(zero=zero, one=one, add=Add, mul=Mul)

    assert diff_forward(zero, x, prop) == zero
    assert diff_forward(x, x, prop) == one
    assert diff_forward(x, y, prop) == zero
    assert diff_forward(Add(x, y), x, prop) == one
    assert diff_forward(Add(y, y), x, prop) == zero
    assert diff_forward(Mul(x, y), y, prop) == Mul(x, one)
    assert diff_forward(Mul(x, y), x, prop) == Mul(one, y)
    assert diff_forward(Mul(x, y), x, prop) == Mul(one, y)

    prop.add_distributive(Vector)

    assert diff_forward(Vector(x, y), x, prop) == Vector(one, zero)

    prop.add_diff_rule(Pow, 0, SubsFunc(Mul(y, Pow(x, Add(y, negone))), [x, y]))
    prop.add_diff_rule(sin, 0, SubsFunc(cos(x), [x]))
    prop.add_diff_rule(cos, 0, SubsFunc(Mul(negone, sin(x)), [x]))

    assert diff_forward(Pow(x, y), x, prop) == Mul(y, Pow(x, Add(y, negone)))
    assert diff_forward(sin(x), x, prop) == cos(x)
    assert diff_forward(sin(sin(x)), x, prop) == Mul(cos(sin(x)), cos(x))
    assert diff_forward(sin(sin(y)), x, prop) == zero
    assert diff_forward(Add(sin(x), cos(x)), x, prop) == Add(
        cos(x), Mul(negone, sin(x))
    )
