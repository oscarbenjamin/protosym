"""Core routines for differentiating at Tree level.

This module implements the basic forward differentiation algorithm. Higher
level code in the sym module wraps this to make a nicer pattern-matching style
interface for specifying differentiation rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from functools import reduce
from typing import TYPE_CHECKING as _TYPE_CHECKING

from protosym.core.tree import forward_graph
from protosym.core.tree import Tr


__all__ = [
    "DiffProperties",
    "diff_forward",
]


if _TYPE_CHECKING:
    from typing import Callable, Sequence
    from protosym.core.atom import AtomType
    from protosym.core.tree import Tree, SubsFunc

    _DiffRules = dict[tuple[Tree, int], SubsFunc]


@dataclass(frozen=True)
class RingOps:
    """Collection of ring operations."""

    Integer: AtomType[int]
    iadd: Callable[[int, int], int]
    imul: Callable[[int, int], int]
    add: Tree
    mul: Tree
    pow: Tree

    def split_integers(self, args: Sequence[Tree]) -> tuple[list[int], list[Tree]]:
        integers: list[int] = []
        new_args: list[Tree] = []
        for arg in args:
            if not arg.children and (atom := arg.value).atom_type == self.Integer:
                integers.append(atom.value)  # type: ignore
            else:
                new_args.append(arg)
        return integers, new_args

    def flatten_add(self, args: list[Tree]) -> Tree:
        integers: list[int] = []
        new_args: list[Tree] = []

        # Associativity of add
        # (x + y) + z -> x + y + z
        for arg in args:
            if arg.children and arg.children[0] == self.add:
                new_args.extend(arg.children[1:])
            else:
                new_args.append(arg)

        # Collect integer part
        # x + 1 + 2 -> 3 + x
        new_args2: list[Tree] = []
        for arg in new_args:
            if not arg.children and (atom := arg.value).atom_type == self.Integer:
                integers.append(atom.value)  # type:ignore
            else:
                new_args2.append(arg)

        # Process all muls, extracting their coefficients
        # 2*x + 3*x -> 5*x
        totals = {}
        for arg in new_args2:
            if arg.children and arg.children[0] == self.mul:
                intfacs, factors = self.split_integers(arg.children[1:])
                if len(factors) == 1:
                    [fac] = factors
                else:
                    fac = self.mul(*factors)
                integer = reduce(self.imul, intfacs, 1)
                if fac not in totals:
                    totals[fac] = 0
                totals[fac] += integer
            else:
                if arg not in totals:
                    totals[arg] = 0
                totals[arg] += 1

        new_args3: list[Tree] = []
        for fac, c in totals.items():
            if c == 0:
                continue
            elif c == 1:
                new_args3.append(fac)
            elif fac.children and fac.children[0] == self.mul:
                new_args3.append(self.mul(Tr(self.Integer(c)), *fac.children[1:]))
            else:
                new_args3.append(self.mul(Tr(self.Integer(c)), fac))

        int_value = reduce(self.iadd, integers, 0)

        if int_value:
            new_args3.insert(0, Tr(self.Integer(int_value)))

        if not new_args3:
            expr = Tr(self.Integer(0))
        elif len(new_args3) == 1:
            [expr] = new_args3
        else:
            expr = self.add(*new_args3)

        return expr

    def flatten_mul(self, args: list[Tree]) -> Tree:
        integers: list[int] = []
        new_args: list[Tree] = []

        for arg in args:
            if arg.children and arg.children[0] == self.mul:
                new_args.extend(arg.children[1:])
            else:
                new_args.append(arg)

        new_args2: list[Tree] = []
        for arg in new_args:
            if not arg.children and (atom := arg.value).atom_type == self.Integer:
                integers.append(atom.value)  # type:ignore
            else:
                new_args2.append(arg)

        powers = {}
        for arg in new_args2:
            if arg.children and arg.children[0] == self.pow:
                base, s_exp = arg.children[1:]
                exp: int
                if s_exp.value.atom_type == self.Integer:
                    exp = s_exp.value.value  # type: ignore
                else:
                    base, exp = arg, 1
            else:
                base, exp = arg, 1

            if base not in powers:
                powers[base] = 0
            powers[base] += exp

        new_args3: list[Tree] = []
        for base, exp in powers.items():
            if exp == 0:
                continue
            elif exp == 1:
                new_args3.append(base)
            else:
                new_args3.append(self.pow(base, Tr(self.Integer(exp))))

        int_value = reduce(self.imul, integers, 1)

        if int_value == 0:
            return Tr(self.Integer(int_value))
        elif int_value != 1:
            new_args3.insert(0, Tr(self.Integer(int_value)))

        if not new_args3:
            expr = Tr(self.Integer(1))
        elif len(new_args3) == 1:
            [expr] = new_args3
        else:
            expr = self.mul(*new_args3)

        return expr

    def flatten_pow(self, args: list[Tree]) -> Tree:
        base, exponent = args

        # (x**y)**a -> x**(a*y) for integer a
        if base.children and base.children[0] == self.pow:
            base_base, base_exp = base.children[1:]
            if not exponent.children and exponent.value.atom_type == self.Integer:
                exponent = self.flatten_mul([base_exp, exponent])
                base = base_base

        if exponent == Tr(self.Integer(0)):
            expr = Tr(self.Integer(1))
        elif exponent == Tr(self.Integer(1)):
            expr = base
        else:
            expr = self.pow(base, exponent)
        return expr

    def flatten(self, expr: Tree) -> Tree:
        """Apply the standard ring simplification rules.

        Identity (addition): :math:`x + 0 = x`
        Identity (multiplication): :math:`x * 1 = x`
        Associativity (addition): :math:`(x + y) + z = x + (y + z)`
        Associativity (multiplication): :math:`(x * y) * z = x * (y * z)`
        Commutativity (addition): :math:`x + y = y + x`
        Commutativity (multiplication): :math:`x * y = y * x`
        Add to Mul: :math:`2*x + 3*x = 5*x`
        Mul to Pow: :math:`x^2 * x^3 = x^5`
        """
        graph = forward_graph(expr)
        stack = list(graph.atoms)
        for func, indices in graph.operations:

            args = [stack[i] for i in indices]

            if func == self.add:
                expr = self.flatten_add(args)
            elif func == self.mul:
                expr = self.flatten_mul(args)
            elif func == self.pow and len(args) == 2:
                expr = self.flatten_pow(args)
            else:
                expr = func(*args)

            stack.append(expr)

        return stack[-1]

    def flatten(self, expr: Tree) -> Tree:
        """Apply common ring simplification rules.

        Identity (addition): :math:`x + 0 = x`
        Identity (multiplication): :math:`x * 1 = x`
        Associativity (addition): :math:`(x + y) + z = x + (y + z)`
        Associativity (multiplication): :math:`(x * y) * z = x * (y * z)`
        Commutativity (addition): :math:`x + y = y + x`
        Commutativity (multiplication): :math:`x * y = y * x`
        Add to Mul: :math:`2*x + 3*x = 5*x`
        Mul to Pow: :math:`x^2 * x^3 = x^5`
        ...
        """
        graph = forward_graph(expr)
        stack = list(graph.atoms)
        for func, indices in graph.operations:

            args = [stack[i] for i in indices]

            if func == self.add:
                expr = self.flatten_add(args)
            elif func == self.mul:
                expr = self.flatten_mul(args)
            elif func == self.pow and len(args) == 2:
                expr = self.flatten_pow(args)
            else:
                expr = func(*args)

            stack.append(expr)

        return stack[-1]


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
