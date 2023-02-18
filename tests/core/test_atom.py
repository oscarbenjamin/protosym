from protosym.core.atom import Atom
from protosym.core.atom import AtomType
from protosym.core.atom import AtomTypeInt
from protosym.core.atom import AtomTypeStr


def test_AtomType() -> None:
    """Test creation and equality of AtomTypes."""
    Integer = AtomTypeInt("Integer", int)
    Symbol = AtomTypeStr("Symbol", str)

    assert isinstance(Integer, AtomType)
    assert isinstance(Symbol, AtomType)
    assert isinstance(Integer, AtomTypeInt)
    assert not isinstance(Integer, AtomTypeStr)
    assert isinstance(Symbol, AtomTypeStr)
    assert not isinstance(Symbol, AtomTypeInt)

    assert type(Integer) is AtomTypeInt
    assert type(Symbol) is AtomTypeStr
    assert str(Integer) == repr(Integer) == "Integer"
    assert str(Symbol) == repr(Symbol) == "Symbol"
    assert (Integer == Integer) is True
    assert (Symbol == Symbol) is True
    assert (Integer == Symbol) is False  # type:ignore[comparison-overlap]
    assert (Symbol == Integer) is False  # type:ignore[comparison-overlap]


def test_Atom() -> None:
    """Test creation and equality of Atoms."""
    Integer = AtomTypeInt("Integer", int)
    Symbol = AtomTypeStr("Symbol", str)

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
    assert one is one2
    assert (one == one2) is True
