"""protosym.core.tree module.

This module defines classes for representing expressions in top-down tree form.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import cast
from typing import Generic as _Generic
from typing import Hashable as _Hashable
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import TypeVar as _TypeVar
from weakref import WeakValueDictionary as _WeakDict

from protosym.core.atom import AtomType


_T = _TypeVar("_T", bound=_Hashable)  # , covariant=True)


if _TYPE_CHECKING:
    from protosym.core.atom import Atom


_all_tree_atoms: _WeakDict[Any, Any] = _WeakDict()
_all_tree_nodes: _WeakDict[Any, Any] = _WeakDict()


class TreeExpr:
    """Base class for tree expressions.

    Every :class:`TreeExpr` is actually either an instance of :class:`TreeAtom`
    for atomic expressions or :class:`TreeNode` for compound expressions.

    >>> from protosym.core.atom import AtomType
    >>> from protosym.core.tree import TreeAtom, TreeExpr, TreeNode
    >>> Function = AtomType('Function', str)
    >>> Integer = AtomType('Integer', int)
    >>> f = TreeAtom(Function('f'))
    >>> one = TreeAtom(Integer(1))
    >>> f
    TreeAtom(Function('f'))
    >>> one
    TreeAtom(Integer(1))
    >>> expr = f(one)
    >>> expr
    TreeNode(TreeAtom(Function('f')), TreeAtom(Integer(1)))
    >>> type(one)
    <class 'protosym.core.tree.TreeAtom'>
    >>> type(expr)
    <class 'protosym.core.tree.TreeNode'>
    >>> [isinstance(e, TreeExpr) for e in [f, one, f(one)]]
    [True, True, True]

    The verbose ``repr`` output seen above is deliberate so that the difference
    can clearly be seen between :class:`Atom`, :class:`TreeAtom` and higher
    level expression types. Prettier output can be found with ``str`` or
    ``print``.

    >>> print(f)
    f
    >>> print(one)
    1
    >>> print(f(one))
    f(1)

    Compound expressions are represented as :class:`TreeNode` instances and
    have a list of ``children`` that are themselves :class:`TreeExpr` whereas
    :class:`TreeAtom` has no children and holds an internal ``value`` that is
    of type :class:`Atom`.

    >>> expr
    TreeNode(TreeAtom(Function('f')), TreeAtom(Integer(1)))
    >>> expr.children
    (TreeAtom(Function('f')), TreeAtom(Integer(1)))
    >>> expr.children[0]
    TreeAtom(Function('f'))
    >>> expr.children[0].children
    ()
    >>> expr.children[0].value
    Function('f')
    >>> type(expr.children[0].value)
    <class 'protosym.core.atom.Atom'>

    Every :class:`TreeExpr` is either constructed from an :class:`Atom` using
    :class:`TreeAtom` or it is constructed as a compound expressions defined in
    terms of preexisting :class:`TreeExpr`. There are two equivalent ways to
    construct a :class:`TreeExpr`:

    >>> f(one) == TreeNode(f, one)
    True

    Every :class:`TreeExpr` is callable so this syntax can always be used even
    if it does not seem to make sense:

    >>> print(one(f))
    1(f)
    >>> one(f)
    TreeNode(TreeAtom(Integer(1)), TreeAtom(Function('f')))

    See Also
    ========

    TreeAtom: subclass of :class:`TreeExpr` representing atomic trees.
    TreeNode: subclass of :class:`TreeExpr` representing compound trees.
    """

    __slots__ = (
        "__weakref__",
        "children",
    )

    children: tuple[TreeExpr, ...]
    """Docstring for children"""

    def __call__(*expressions: TreeExpr) -> TreeNode:
        """Compound expressions are made by calling TreeExpr instances."""
        return TreeNode(*expressions)


class TreeAtom(TreeExpr, _Generic[_T]):
    """Class for atomic tree expressions.

    A :class:`TreeAtom` wraps an :class:`Atom` as its internal ``value``
    attribute and then provides an empty :attr:`TreeExpr.children` attribute to
    match the interface expected of :class:`TreeExpr`.

    >>> from protosym.core.atom import AtomType
    >>> from protosym.core.tree import TreeAtom
    >>> Integer = AtomType('Integer', int)
    >>> one = TreeAtom(Integer(1))
    >>> one
    TreeAtom(Integer(1))
    >>> one.value
    Integer(1)
    >>> one.children
    ()

    See Also
    ========

    protosym.core.atom.Atom: Lower level representation of atomic expressions.
    TreeExpr: superclass of :class:`TreeAtom` representing all tree-form
        expressions.
    TreeNode: alternate subclass of :class:`TreeExpr` representing compound
        expressions.
    """

    __slots__ = ("value",)

    value: Atom[_T]

    def __new__(cls, value: Atom[_T]) -> TreeAtom[_T]:
        """Return a prevously created TreeAtom or a new one."""
        previous = _all_tree_atoms.get(value, None)
        if previous is not None:
            return cast("TreeAtom[_T]", previous)

        obj = object.__new__(cls)
        obj.children = ()
        obj.value = value

        obj = cast("TreeAtom[_T]", _all_tree_atoms.setdefault(value, obj))

        return obj

    def __repr__(self) -> str:
        """Show the verbose representaton."""
        return f"TreeAtom({self.value!r})"

    def __str__(self) -> str:
        """Show the pretty form."""
        return str(self.value)


class TreeNode(TreeExpr):
    """Class for compound tree expressions.

    A :class:`TreeNode` represents a non-atomic :class:`TreeExpr` by holding a
    tuple of child :class:`TreeExpr` objects. Before we can construct a
    :class:`TreeNode` we first need to construct :class:`TreeAtom` expressions
    to represent the children.

    >>> from protosym.core.atom import AtomType
    >>> from protosym.core.tree import TreeNode
    >>> Integer = AtomType('Integer', int)
    >>> Function = AtomType('Function', str)
    >>> one = TreeAtom(Integer(1))
    >>> cos = TreeAtom(Function('cos'))
    >>> cos(one)
    TreeNode(TreeAtom(Function('cos')), TreeAtom(Integer(1)))
    >>> cos(cos(one))
    TreeNode(TreeAtom(Function('cos')), TreeNode(TreeAtom(Function('cos')),\
 TreeAtom(Integer(1))))
    >>> print(cos(cos(one)))
    cos(cos(1))

    See Also
    ========

    TreeExpr: superclass of :class:`TreeNode` representing all tree-form
        expressions.
    TreeAtom: alternate subclass of :class:`TreeExpr` representing atomic
        expressions.
    """

    __slots__ = ()

    def __new__(cls, *children: TreeExpr) -> TreeNode:
        """Return a prevously created TreeAtom or a new one."""
        previous = _all_tree_nodes.get(children, None)
        if previous is not None:
            return previous  # type:ignore

        obj = object.__new__(cls)
        obj.children = children

        obj = _all_tree_nodes.setdefault(children, obj)

        return obj

    def __repr__(self) -> str:
        """Show the verbose representaton."""
        argstr = ", ".join(map(repr, self.children))
        return f"TreeNode({argstr})"

    def __str__(self) -> str:
        """Show the pretty form."""
        head = self.children[0]
        args = self.children[1:]
        argstr = ", ".join(map(str, args))
        return f"{head}({argstr})"


def funcs_symbols(
    function_names: list[str], symbol_names: list[str]
) -> tuple[list[TreeAtom[str]], list[TreeAtom[str]]]:
    """Convenience function to make some functions and symbols."""
    Function = AtomType("Function", str)  # noqa
    Symbol = AtomType("Symbol", str)  # noqa
    functions = [TreeAtom(Function(name)) for name in function_names]
    symbols = [TreeAtom(Symbol(name)) for name in symbol_names]
    return functions, symbols


def topological_sort(expression: TreeExpr, *, heads: bool = False) -> list[TreeExpr]:
    """List of subexpressions of a :class:`TreeExpr` sorted topologically.

    Create some functions and symbols and use them to make an expression:

    >>> from protosym.core.tree import funcs_symbols
    >>> [f], [x, y] = funcs_symbols(['f'], ['x', 'y'])
    >>> expr = f(f(x, y), f(f(x)))
    >>> print(expr)
    f(f(x, y), f(f(x)))

    Now :func:`topological_sort` gives a list of all subexpressions of
    *expression* in topological order. The ordering means that no expression
    appears before any of its children:

    >>> for e in topological_sort(expr):
    ...     print(e)
    x
    y
    f(x, y)
    f(x)
    f(f(x))
    f(f(x, y), f(f(x)))

    The ``heads`` parameter defaults to ``False`` meaning that heads are not
    explicitly included in the list. Pass ``heads=True`` to include them.

    >>> topological_sort(f(x, y)) == [x, y, f(x, y)]
    True
    >>> topological_sort(f(x, y), heads=True) == [f, x, y, f(x, y)]
    True

    See Also
    ========

    TreeExpr: The expression class that this function operates on.
    topological_split: Splits the sort into atoms, heads and nodes.
    """
    #
    # We use a stack here rather than recursion so that there is no limit on
    # the recursion depth. Otherwise though this is really just the same as
    # walking an expression tree recursively but using a set to avoid
    # re-walking any repeating subexpressions. At the end though we get a
    # structure that contains each subexpression exactly once with no
    # repetition. That allows any calling routines to avoid needing to check or
    # optimise for repeating subexpressions. This routine could be made more
    # efficient but in usage it does not seem to be a major bottleneck compared
    # to the ones processing its output.
    #
    def get_children(expr: TreeExpr) -> list[TreeExpr]:
        if heads:
            children = expr.children
        else:
            children = expr.children[1:]
        return list(children)[::-1]

    seen = set()
    expressions = []
    stack = [(expression, get_children(expression))]

    while stack:
        top, children = stack[-1]
        while children:
            child = children.pop()
            if child not in seen:
                seen.add(child)
                stack.append((child, get_children(child)))
                break
        else:
            stack.pop()
            expressions.append(top)

    return expressions


def topological_split(
    expr: TreeExpr,
) -> tuple[list[TreeExpr], set[TreeExpr], list[TreeExpr]]:
    """Topological sort split into atoms, heads and compound expressions.

    First build an expression:

    >>> from protosym.core.tree import funcs_symbols
    >>> [f, g], [x, y] = funcs_symbols(['f', 'g'], ['x', 'y'])
    >>> expr = f(g(x, y), y)
    >>> print(expr)
    f(g(x, y), y)

    Now compute the topological split:

    >>> atoms, heads, nodes = topological_split(expr)
    >>> atoms == [x, y]
    True
    >>> heads == {f, g}
    True
    >>> nodes == [g(x, y), f(g(x, y), y)]
    True

    The nodes will be sorted topologically with each expression appearing
    after all of its children.

    See Also
    ========

    TreeExpr: The expression class that this operates on.
    topological_sort: Topological sort as a list of all subexpressions.
    """
    subexpressions = topological_sort(expr)

    atoms: list[TreeExpr] = []
    heads: set[TreeExpr] = set()
    nodes: list[TreeExpr] = []

    for subexpr in subexpressions:
        children = subexpr.children
        if not children:
            atoms.append(subexpr)
        else:
            heads.add(children[0])
            nodes.append(subexpr)

    return atoms, heads, nodes


def forward_graph(expr: TreeExpr) -> ForwardGraph:
    """Build a ``ForwardGraph`` from a ``TreeExpr``.

    >>> from protosym.core.tree import funcs_symbols
    >>> [f, g], [x, y] = funcs_symbols(['f', 'g'], ['x', 'y'])
    >>> expr = f(g(x, y), y)
    >>> print(expr)
    f(g(x, y), y)

    Now build the forward graph:

    >>> from protosym.core.tree import forward_graph
    >>> graph = forward_graph(expr)
    >>> graph.atoms == [x, y]
    True
    >>> graph.heads == {f, g}
    True
    >>> graph.operations == [(g, [0, 1]), (f, [2, 1])]
    True

    The forward graph can be used to rebuild the expression through *forward
    evaluation*:

    >>> stack = list(graph.atoms)
    >>> for head, indices in graph.operations:
    ...     args = [stack[i] for i in indices]
    ...     stack.append(head(*args))

    Now ``stack`` is the topological sort of ``expr`` and ``stack[-1]`` is
    ``expr``.

    >>> from protosym.core.tree import topological_sort
    >>> stack == [x, y, g(x, y), f(g(x, y), y)]
    True
    >>> stack == topological_sort(expr)
    True
    >>> stack[-1] == expr
    True

    See Also
    ========

    topological_sort
    ForwardGraph: The class of the object returned.
    """
    atoms, heads, nodes = topological_split(expr)

    num_atoms = len(atoms)

    operations: list[tuple[TreeExpr, list[int]]] = []
    indices: dict[TreeExpr, int] = dict(zip(atoms, range(num_atoms)))

    for index, subexpr in enumerate(nodes, num_atoms):
        head = subexpr.children[0]
        args = subexpr.children[1:]
        arg_indices = [indices[e] for e in args]
        operations.append((head, arg_indices))
        indices[subexpr] = index

    return ForwardGraph(atoms, heads, operations)


@dataclass
class ForwardGraph:
    """Representation of an expression as a forward graph.

    See Also
    ========

    TreeExpr: Representation of an expression as a tree.
    """

    atoms: list[TreeExpr]
    heads: set[TreeExpr]
    operations: list[tuple[TreeExpr, list[int]]]
