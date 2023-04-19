"""Core routines for differentiating at Tree level.

This module implements the basic forward differentiation algorithm. Higher
level code in the sym module wraps this to make a nicer pattern-matching style
interface for specifying differentiation rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING as _TYPE_CHECKING

from protosym.core.tree import forward_graph


__all__ = [
    "DiffProperties",
    "diff_forward",
]


if _TYPE_CHECKING:
    from protosym.core.tree import Tree, SubsFunc

    _DiffRules = dict[tuple[Tree, int], SubsFunc]


@dataclass(frozen=True)
class DiffProperties:
    """Collection of properties needed for differentiation."""

    zero: Tree
    one: Tree
    add: Tree
    mul: Tree
    distributive: set[Tree] = field(default_factory=set)
    diff_rules: _DiffRules = field(default_factory=dict)

    def add_distributive(self, head: Tree) -> None:
        """Add a distributive rule :math:`f(x, y)' = f(x', y')`."""
        self.distributive.add(head)

    def add_diff_rule(self, head: Tree, argnum: int, func: SubsFunc) -> None:
        """Add an elementary rule like :math:`sin(x)' = cos(x)`."""
        self.diff_rules[head, argnum] = func


def diff_forward(
    expression: Tree,
    sym: Tree,
    prop: DiffProperties,
) -> Tree:
    """Derivative of expression wrt sym.

    Uses forward accumulation algorithm.
    """
    one = prop.one
    zero = prop.zero
    add = prop.add
    mul = prop.mul
    distributive = prop.distributive
    diff_rules = prop.diff_rules

    graph = forward_graph(expression)

    stack = list(graph.atoms)
    diff_stack = [one if expr == sym else zero for expr in stack]

    for func, indices in graph.operations:
        args = [stack[i] for i in indices]
        diff_args = [diff_stack[i] for i in indices]
        expr = func(*args)

        if func in distributive:
            diff_terms = [func(*diff_args)]
        elif set(diff_args) == {zero}:
            # XXX: This could return the wrong thing if the expression has an
            # unrecognised head and does not represent a number.
            diff_terms = []
        elif func == add:
            diff_terms = [da for da in diff_args if da != zero]
        elif func == mul:
            diff_terms = product_rule_forward(args, diff_args, zero, mul)
        else:
            diff_terms = chain_rule_forward(
                func, args, diff_args, zero, one, mul, diff_rules
            )

        if not diff_terms:
            derivative = zero
        elif len(diff_terms) == 1:
            derivative = diff_terms[0]
        else:
            derivative = add(*diff_terms)

        stack.append(expr)
        diff_stack.append(derivative)

    # At this point stack is a topological sort of expr and diff_stack is the
    # list of derivatives of every subexpression in expr. At the top of the
    # stack is expr and its derivative is at the top of diff_stack.
    return diff_stack[-1]


def product_rule_forward(
    args: list[Tree],
    diff_args: list[Tree],
    zero: Tree,
    mul: Tree,
) -> list[Tree]:
    """Product rule in forward accumulation."""
    terms: list[Tree] = []
    for n, diff_arg in enumerate(diff_args):
        if diff_arg != zero:
            # XXX: Maybe check if diff_arg == 1 here?
            term = mul(*args[:n], diff_arg, *args[n + 1 :])
            terms.append(term)
    return terms


def chain_rule_forward(
    func: Tree,
    args: list[Tree],
    diff_args: list[Tree],
    zero: Tree,
    one: Tree,
    mul: Tree,
    diff_rules: _DiffRules,
) -> list[Tree]:
    """Chain rule in forward accumulation."""
    terms: list[Tree] = []
    for n, diff_arg in enumerate(diff_args):
        if diff_arg != zero:
            pdiff = diff_rules[(func, n)]
            diff_term = pdiff(*args)
            if diff_arg != one:
                diff_term = mul(diff_term, diff_arg)
            terms.append(diff_term)
    return terms


try:
    import rust_protosym
except ImportError:
    rust_protosym = None

if rust_protosym is not None:  # pragma: no cover
    from rust_protosym import DiffProperties, diff_forward  # type:ignore # noqa
