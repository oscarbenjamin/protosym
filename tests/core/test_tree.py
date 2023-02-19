from protosym.core.atom import Atom
from protosym.core.atom import AtomTypeInt
from protosym.core.atom import AtomTypeStr
from protosym.core.tree import topological_sort
from protosym.core.tree import TreeAtom
from protosym.core.tree import TreeExpr
from protosym.core.tree import TreeNode


def test_TreeExpr_basic() -> None:
    """Test basic construction and equality of TreeExpr."""
    Integer = AtomTypeInt("Integer", int)
    Function = AtomTypeStr("Function", str)
    one_atom = Integer(1)
    f_atom = Function("f")

    one_tree = TreeAtom(one_atom)
    f_tree = TreeAtom(f_atom)
    f_one = f_tree(one_tree)

    assert str(one_tree) == "1"
    assert str(f_tree) == "f"
    assert str(f_one) == "f(1)"

    assert repr(one_tree) == "TreeAtom(Integer(1))"
    assert repr(f_tree) == "TreeAtom(Function('f'))"
    assert repr(f_one) == "TreeNode(TreeAtom(Function('f')), TreeAtom(Integer(1)))"

    assert not isinstance(f_one, Atom)
    assert not isinstance(f_tree, Atom)
    assert isinstance(f_tree, TreeAtom)
    assert isinstance(f_tree, TreeExpr)
    assert isinstance(one_tree, TreeAtom)
    assert isinstance(one_tree, TreeExpr)
    assert isinstance(f_one, TreeNode)
    assert isinstance(f_one, TreeExpr)

    assert (one_tree == one_tree) is True
    assert (f_tree == f_tree) is True
    assert (f_one == f_one) is True
    assert (one_tree == f_tree) is False  # type: ignore[comparison-overlap]
    assert (one_tree == one_atom) is False  # type: ignore[comparison-overlap]

    assert (one_tree != one_tree) is False
    assert (f_tree != f_tree) is False
    assert (f_one != f_one) is False
    assert (one_tree != f_tree) is True  # type: ignore[comparison-overlap]
    assert (one_tree != one_atom) is True  # type: ignore[comparison-overlap]

    assert TreeAtom(one_atom) is TreeAtom(one_atom)
    assert f_tree(one_tree) is f_tree(one_tree)


def test_topological_sort() -> None:
    """Simple test for the topological_sort function."""
    Function = AtomTypeStr("Function", str)
    Symbol = AtomTypeStr("Symbol", str)
    f = TreeAtom(Function("f"))
    x = TreeAtom(Symbol("x"))
    y = TreeAtom(Symbol("y"))

    expr = f(f(x, y), f(f(x), f(x, y)))
    subexpressions = [
        f,
        x,
        y,
        f(x, y),
        f(x),
        f(f(x), f(x, y)),
        f(f(x, y), f(f(x), f(x, y))),
    ]
    assert topological_sort(expr) == subexpressions
