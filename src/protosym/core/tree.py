"""protosym.core.tree module.

This module defines classes for representing expressions in top-down tree form.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING as _TYPE_CHECKING
from weakref import WeakValueDictionary as _WeakDict

from protosym.core.atom import Atom
from protosym.core.atom import AtomType


if _TYPE_CHECKING:
    from protosym.core.atom import AnyAtom


__all__ = [
    "Tree",
    "Tr",
    "ForwardGraph",
    "topological_sort",
    "topological_split",
    "forward_graph",
]


_all_trees: _WeakDict[AnyAtom | tuple[Tree, ...], Tree] = _WeakDict()


class Tree:
    """Base class for tree expressions.

    Every :class:`Tree` is either an atomic :class:`Tree` or a compound
    :class:`Tree` having children that are themselves of type :class:`Tree`.

    >>> from protosym.core.atom import AtomType
    >>> from protosym.core.tree import Tr
    >>> Function = AtomType('Function', str)
    >>> Integer = AtomType('Integer', int)
    >>> f = Tr(Function('f'))
    >>> one = Tr(Integer(1))
    >>> f
    Tr(Function('f'))
    >>> one
    Tr(Integer(1))
    >>> expr = f(one)
    >>> expr
    Tree(Tr(Function('f')), Tr(Integer(1)))
    >>> type(one)
    <class 'protosym.core.tree.Tree'>
    >>> type(expr)
    <class 'protosym.core.tree.Tree'>
    >>> [isinstance(e, Tree) for e in [f, one, f(one)]]
    [True, True, True]

    The verbose ``repr`` output seen above is deliberate so that the difference
    can clearly be seen between :class:`Atom`, :class:`Tree` and higher
    level expression types. Prettier output can be found with ``str`` or
    ``print``.

    >>> print(f)
    f
    >>> print(one)
    1
    >>> print(f(one))
    f(1)

    Compound expressions are represented as :class:`Tree` instances and have a
    list of ``children`` that are themselves :class:`Tree` whereas an atomic
    :class:`Tree` has no children and holds an internal ``value`` that is of
    type :class:`Atom`.

    >>> expr
    Tree(Tr(Function('f')), Tr(Integer(1)))
    >>> expr.children
    (Tr(Function('f')), Tr(Integer(1)))
    >>> expr.children[0]
    Tr(Function('f'))
    >>> expr.children[0].children
    ()
    >>> expr.children[0].value
    Function('f')
    >>> type(expr.children[0].value)
    <class 'protosym.core.atom.Atom'>

    Every :class:`Tree` is either constructed from an :class:`Atom` using
    :class:`Tree` or it is constructed as a compound expressions defined in
    terms of preexisting :class:`Tree`. There are two equivalent ways to
    construct a non-atomic :class:`Tree`:

    >>> f(one) == Tree(f, one)
    True

    Every :class:`Tree` is callable so this syntax can always be used even if
    it does not seem to make sense:

    >>> print(one(f))
    1(f)
    >>> one(f)
    Tree(Tr(Integer(1)), Tr(Function('f')))
    """

    __slots__ = (
        "__weakref__",
        "value",
        "children",
    )

    children: tuple[Tree, ...]
    """Docstring for children"""

    value: AnyAtom
    """Docstring for value"""

    def __new__(cls, *children: Tree) -> Tree:
        """Return a prevously created Tree or a new one."""
        previous = _all_trees.get(children, None)
        if previous is not None:
            return previous

        if not all(isinstance(child, Tree) for child in children):
            raise TypeError("All arguments should be Tree.")

        obj = object.__new__(cls)
        obj.children = children

        obj = _all_trees.setdefault(children, obj)

        return obj

    @classmethod
    def atom(cls, value: AnyAtom) -> Tree:
        """Create a Tree representing an atomic expression."""
        if not isinstance(value, Atom):
            raise TypeError("The value should be an Atom.")

        previous = _all_trees.get(value, None)
        if previous is not None:
            return previous

        obj = super().__new__(cls)
        obj.value = value
        obj.children = ()

        obj = _all_trees.setdefault(value, obj)

        return obj

    def __call__(*expressions: Tree) -> Tree:
        """Compound expressions are made by calling Tree instances."""
        return Tree(*expressions)

    def __repr__(self) -> str:
        """Show the verbose representaton."""
        if self.children:
            argstr = ", ".join(map(repr, self.children))
            return f"Tree({argstr})"
        else:
            return f"Tr({self.value!r})"

    def __str__(self) -> str:
        """Show the pretty form."""
        if self.children:
            head = self.children[0]
            args = self.children[1:]
            argstr = ", ".join(map(str, args))
            return f"{head}({argstr})"
        else:
            return str(self.value)


# Convenient shorthand for creating atoms
Tr = Tree.atom


def funcs_symbols(
    function_names: list[str], symbol_names: list[str]
) -> tuple[list[Tree], list[Tree]]:
    """Convenience function to make some functions and symbols.

    >>> from protosym.core.tree import funcs_symbols
    >>> [f, g], [x, y] = funcs_symbols(['f', 'g'], ['x', 'y'])
    >>> expr = f(g(x, y))
    >>> print(expr)
    f(g(x, y))

    This is mainly just here to reduce boilerplate in other docstrings.
    """
    Function = AtomType("Function", str)  # noqa
    Symbol = AtomType("Symbol", str)  # noqa
    functions = [Tr(Function(name)) for name in function_names]
    symbols = [Tr(Symbol(name)) for name in symbol_names]
    return functions, symbols


def topological_sort(expression: Tree, *, heads: bool = False) -> list[Tree]:
    """List of subexpressions of a :class:`Tree` sorted topologically.

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

    Tree: The expression class that this function operates on.
    topological_split: Splits the sort into atoms, heads and nodes.
    """
    #
    # We use a stack here rather than recursion so that there is no limit on
    # the recursion depth. Otherwise though this is really just the same as
    # walking an expression tree recursively but using a set to avoid
    # re-walking any repeating subexpressions. At the end we get a structure
    # that contains each subexpression exactly once with no repetition. That
    # allows any calling routines to avoid needing to check or optimise for
    # repeating subexpressions. This routine could be made more efficient but
    # in usage it does not seem to be a major bottleneck compared to the ones
    # processing its output.
    #
    def get_children(expr: Tree) -> list[Tree]:
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
    expr: Tree,
) -> tuple[list[Tree], set[Tree], list[Tree]]:
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

    Tree: The expression class that this operates on.
    topological_sort: Topological sort as a list of all subexpressions.
    """
    subexpressions = topological_sort(expr)

    atoms: list[Tree] = []
    heads: set[Tree] = set()
    nodes: list[Tree] = []

    for subexpr in subexpressions:
        children = subexpr.children
        if not children:
            atoms.append(subexpr)
        else:
            heads.add(children[0])
            nodes.append(subexpr)

    return atoms, heads, nodes


def forward_graph(expr: Tree) -> ForwardGraph:
    """Build a ``ForwardGraph`` from a ``Tree``.

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

    operations: list[tuple[Tree, list[int]]] = []
    indices: dict[Tree, int] = dict(zip(atoms, range(num_atoms)))

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

    Tree: Representation of an expression as a tree.
    """

    atoms: list[Tree]
    heads: set[Tree]
    operations: list[tuple[Tree, list[int]]]
