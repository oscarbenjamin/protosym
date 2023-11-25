from protosym.core.atom import Atom, AtomType
from protosym.core.tree import (
    ForwardGraph,
    SubsFunc,
    Tr,
    Tree,
    forward_graph,
    funcs_symbols,
    topological_sort,
    topological_split,
)
from pytest import raises


def test_Tree_basic() -> None:
    """Test basic construction and equality of Tree."""
    Integer = AtomType("Integer", int)
    Function = AtomType("Function", str)
    one_atom = Integer(1)
    f_atom = Function("f")

    one_tree = Tr(one_atom)
    f_tree = Tr(f_atom)
    f_one = f_tree(one_tree)

    assert str(one_tree) == "1"
    assert str(f_tree) == "f"
    assert str(f_one) == "f(1)"

    assert repr(one_tree) == "Tr(Integer(1))"
    assert repr(f_tree) == "Tr(Function('f'))"
    assert repr(f_one) == "Tree(Tr(Function('f')), Tr(Integer(1)))"

    assert not isinstance(f_one, Atom)
    assert not isinstance(f_tree, Atom)
    assert isinstance(f_tree, Tree)
    assert isinstance(f_tree, Tree)
    assert isinstance(one_tree, Tree)
    assert isinstance(one_tree, Tree)
    assert isinstance(f_one, Tree)
    assert isinstance(f_one, Tree)

    assert (one_tree == one_tree) is True
    assert (f_tree == f_tree) is True
    assert (f_one == f_one) is True
    assert (one_tree == f_tree) is False
    assert (one_tree == one_atom) is False  # type: ignore[comparison-overlap]

    assert (one_tree != one_tree) is False
    assert (f_tree != f_tree) is False
    assert (f_one != f_one) is False
    assert (one_tree != f_tree) is True
    assert (one_tree != one_atom) is True  # type: ignore[comparison-overlap]

    # assert Tr(one_atom) is Tr(one_atom)
    # assert f_tree(one_tree) is f_tree(one_tree)

    raises(TypeError, lambda: Tree(1))  # type: ignore
    raises(TypeError, lambda: Tr(1))  # type: ignore


def test_funcs_symbols() -> None:
    """Test the funcs_symbols convenience function."""
    [f, g], [x, y] = funcs_symbols(["f", "g"], ["x", "y"])
    assert all(isinstance(e, Tree) for e in [f, g, x, y])


def test_topological_sort_split() -> None:
    """Simple tests for topological_sort and topological_split."""
    Function = AtomType("Function", str)
    Symbol = AtomType("Symbol", str)
    f = Tr(Function("f"))
    x = Tr(Symbol("x"))
    y = Tr(Symbol("y"))

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
    # Passing heads=True will include f in the list.
    assert topological_sort(expr) == subexpressions[1:]
    assert topological_sort(expr, heads=False) == subexpressions[1:]
    assert topological_sort(expr, heads=True) == subexpressions

    # Test generating a topological sort that excludes f(x, y).
    # This also excludes children like y that do not appear elsewhere.
    expected_exclude = [
        x,
        f(x),
        f(f(x), f(x, y)),
        f(f(x, y), f(f(x), f(x, y))),
    ]
    assert topological_sort(expr, exclude={f(x, y)}) == expected_exclude

    expected_split = (
        [x, y],
        {f},
        [f(x, y), f(x), f(f(x), f(x, y)), f(f(x, y), f(f(x), f(x, y)))],
    )
    assert topological_split(expr) == expected_split


def test_forward_graph() -> None:
    """Basic test for the forward_graph function."""
    [f, g], [x, y] = funcs_symbols(["f", "g"], ["x", "y"])

    expr = f(y, f(x, g(y)))

    expected = ForwardGraph(
        [y, x],
        {g, f},
        [(g, [0]), (f, [1, 2]), (f, [0, 3])],
    )

    assert forward_graph(expr) == expected


def test_subsfunc() -> None:
    """Test basic functionality of SubsFunc."""
    [f, g], [x, y, z, t] = funcs_symbols(["f", "g"], ["x", "y", "z", "t"])
    expr = f(f(x, y), g(y))
    subs = SubsFunc(expr, [x, y])
    assert subs(z, t) == f(f(z, t), g(t))
    assert subs.nargs == 2
    assert subs.atoms == [f, g]
    assert subs.operations == [[2, 0, 1], [3, 1], [2, 4, 5]]

    subs = SubsFunc(expr, [f(x, y)])
    assert subs(z) == f(z, g(y))
    assert subs.nargs == 1
    assert subs.atoms == [f, g(y)]
    assert subs.operations == [[1, 0, 2]]

    subs = SubsFunc(expr, [expr])
    assert subs(t) == t
    assert subs.nargs == 1
    assert subs.atoms == []
    assert subs.operations == []

    subs = SubsFunc(expr, [t])
    assert subs(z) == expr
    assert subs.nargs == 1
    assert subs.atoms == [expr]
    assert subs.operations == []

    raises(TypeError, lambda: subs(z, t))
