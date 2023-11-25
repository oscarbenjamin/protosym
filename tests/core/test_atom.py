from protosym.core.atom import Atom, AtomType


def test_AtomType() -> None:
    """Test creation and equality of AtomTypes."""
    Integer = AtomType("Integer", int)
    Symbol = AtomType("Symbol", str)

    assert isinstance(Integer, AtomType)
    assert isinstance(Symbol, AtomType)
    assert type(Integer) is AtomType
    assert type(Symbol) is AtomType
    assert str(Integer) == repr(Integer) == "Integer"
    assert str(Symbol) == repr(Symbol) == "Symbol"
    assert (Integer == Integer) is True
    assert (Symbol == Symbol) is True
    assert (Integer == Symbol) is False  # type:ignore[comparison-overlap]
    assert (Symbol == Integer) is False  # type:ignore[comparison-overlap]


def test_Atom() -> None:
    """Test creation and equality of Atoms."""
    Integer = AtomType("Integer", int)
    Symbol = AtomType("Symbol", str)

    one = Integer(1)
    zero = Integer(0)
    x = Symbol("x")

    assert type(one) is Atom
    assert type(zero) is Atom
    assert type(x) is Atom

    assert str(one) == "1"
    assert str(zero) == "0"
    assert str(x) == "x"

    assert repr(one) == "Integer(1)"
    assert repr(zero) == "Integer(0)"
    assert repr(x) == "Symbol('x')"

    assert (one == one) is True
    assert (zero == zero) is True
    assert (x == x) is True
    assert (one == zero) is False
    assert (zero == one) is False
    assert (x == one) is False  # type:ignore[comparison-overlap]

    assert (one != one) is False
    assert (zero != zero) is False
    assert (x != x) is False
    assert (one != zero) is True
    assert (zero != one) is True
    assert (x != one) is True  # type:ignore[comparison-overlap]

    # Atoms should be globally unique
    one2 = Integer(1)
    # assert one is one2
    assert (one == one2) is True
