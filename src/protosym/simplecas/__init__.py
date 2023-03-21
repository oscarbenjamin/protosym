"""Demonstration of putting together a simple CAS."""
from __future__ import annotations

import math
import struct
from functools import reduce
from functools import wraps
from typing import Any
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Union

from protosym.core.evaluate import Transformer
from protosym.core.exceptions import ProtoSymError
from protosym.core.sym import AtomFunc
from protosym.core.sym import AtomRule
from protosym.core.sym import HeadOp
from protosym.core.sym import HeadRule
from protosym.core.sym import PyFunc1
from protosym.core.sym import PyOp1
from protosym.core.sym import PyOp2
from protosym.core.sym import PyOpN
from protosym.core.sym import star
from protosym.core.sym import Sym
from protosym.core.tree import forward_graph
from protosym.core.tree import topological_sort
from protosym.core.tree import TreeAtom


if _TYPE_CHECKING:
    from protosym.core.sym import SymEvaluator
    from protosym.core.tree import TreeExpr


class ExpressifyError(TypeError):
    """Raised when an object cannot be expressified."""

    pass


def expressify(obj: Any) -> Expr:
    """Convert a native Python object to an ``Expr``.

    >>> from protosym.simplecas import expressify
    >>> one = expressify(1)
    >>> one
    1
    >>> type(one)
    <class 'protosym.simplecas.Expr'>

    This is the internal representation of the ``one`` object returned by
    :func:`expressify`:

    >>> one.rep
    TreeAtom(Integer(1))

    It is harmless to call :func:`expressify` more than once because it will
    just return the same object.

    >>> expressify(one) is one
    True

    Notes
    =====

    Currently :func:`expressify` only supports converting ``int`` to
    :class:`Expr`. Otherwise any :class:`Expr` is returned as is. It will be
    extended as further possible types are added to :mod:`protosym.simplecas`.
    """
    #
    # This is supposed to be analogous to sympify but needs improvement.
    #
    # - It should be extensible.
    # - There should also be different kinds of expressify for different
    #   contexts e.g. if we know that we are expecting a bool rather than a
    #   number.
    #
    if isinstance(obj, Expr):
        return obj
    elif isinstance(obj, int):
        return Integer(obj)
    else:
        raise ExpressifyError


if _TYPE_CHECKING:
    Expressifiable = Union["Expr", int]
    ExprBinOp = Callable[["Expr", "Expr"], "Expr"]
    ExpressifyBinOp = Callable[["Expr", Expressifiable], "Expr"]


def expressify_other(method: ExprBinOp) -> ExpressifyBinOp:
    """Decorator to call ``expressify`` on operands in ``__add__`` etc."""

    @wraps(method)
    def expressify_method(self: Expr, other: Expressifiable) -> Expr:
        if not isinstance(other, Expr):
            try:
                other = expressify(other)
            except ExpressifyError:
                return NotImplemented
        return method(self, other)

    return expressify_method


class Expr(Sym):
    """User-facing class for representing expressions.

    To create an :class:`Expr` first import the basic types from
    :mod:`protosym.simplecas` and then use them to build up some expressions.

    >>> from protosym.simplecas import Symbol, cos, sin
    >>> x = Symbol('x')
    >>> y = Symbol('y')
    >>> expr = sin(x)
    >>> expr
    sin(x)
    >>> expr.diff(x)
    cos(x)

    Unlike some other symbolic manipulation systems expressions here are inert
    and will not change their form implicitly.

    >>> x + x
    (x + x)
    >>> y + x
    (y + x)
    >>> x + y
    (x + y)

    This has some surprising consequences such as the preservation of the order
    of binary operations.

    >>> x + x + x
    ((x + x) + x)

    Here the :class:`Expr` was created by evaluating the Python code ``x + x +
    x`` which is in fact evaluated as ``((x + x) + x)``. Since :class:`Expr`
    preserves this form that is what the resulting expression will be.

    On the other hand if brackets are included explicitly then they will be
    preserved.

    >>> (x + (x + x))
    (x + (x + x))

    An :class:`Expr` directly displays its internal form which can be surprising e.g.:

    >>> x - y
    (x + (-1*y))

    This is in fact the form that many algebra systems use to represent ``x -
    y`` although often the resulting expression would be displayed differently.

    The :class:`Expr` class has many methods some of which are listed below.

    See Also
    ========

    diff
    eval_f64
    eval_latex
    from_sympy
    count_ops_graph
    count_ops_tree
    from_sympy
    to_sympy
    """

    def __repr__(self) -> str:
        """Pretty string representation of the expression."""
        return self.eval_repr()

    def __str__(self) -> str:
        """Pretty string representation of the expression."""
        return self.eval_repr()

    def _repr_latex_(self) -> str:
        """Latex hook for IPython."""
        return f"${self.eval_latex()}$"

    def _sympy_(self) -> Any:
        """Hook for SymPy's ``sympify`` function."""
        return self.to_sympy()

    def __call__(self, *args: Expressifiable) -> Expr:
        """Call this Expr as a function."""
        args_expr = [expressify(arg) for arg in args]
        args_rep = [arg.rep for arg in args_expr]
        return Expr(self.rep(*args_rep))

    def __pos__(self) -> Expr:
        """+Expr -> Expr."""
        return self

    def __neg__(self) -> Expr:
        """+Expr -> Expr."""
        return Mul(negone, self)

    @expressify_other
    def __add__(self, other: Expr) -> Expr:
        """Expr + Expr -> Expr."""
        return Add(self, other)

    @expressify_other
    def __radd__(self, other: Expr) -> Expr:
        """Expr + Expr -> Expr."""
        return Add(other, self)

    @expressify_other
    def __sub__(self, other: Expr) -> Expr:
        """Expr - Expr -> Expr."""
        return Add(self, Mul(negone, other))

    @expressify_other
    def __rsub__(self, other: Expr) -> Expr:
        """Expr - Expr -> Expr."""
        return Add(other, Mul(negone, self))

    @expressify_other
    def __mul__(self, other: Expr) -> Expr:
        """Expr * Expr -> Expr."""
        return Mul(self, other)

    @expressify_other
    def __rmul__(self, other: Expr) -> Expr:
        """Expr * Expr -> Expr."""
        return Mul(other, self)

    @expressify_other
    def __truediv__(self, other: Expr) -> Expr:
        """Expr / Expr -> Expr."""
        return Mul(self, Pow(other, negone))

    @expressify_other
    def __rtruediv__(self, other: Expr) -> Expr:
        """Expr / Expr -> Expr."""
        return Mul(other, Pow(self, negone))

    @expressify_other
    def __pow__(self, other: Expr) -> Expr:
        """Expr ** Expr -> Expr."""
        return Pow(self, other)

    @expressify_other
    def __rpow__(self, other: Expr) -> Expr:
        """Expr ** Expr -> Expr."""
        return Pow(other, self)

    def eval_repr(self) -> str:
        """Pretty string e.g. "cos(x) + 1"."""
        return eval_repr(self)

    def eval_latex(self) -> str:
        r"""Return a LaTeX representaton of the expression.

        >>> from protosym.simplecas import Symbol, sin
        >>> x = Symbol('x')
        >>> expr = sin(x**2)
        >>> expr
        sin(x**2)
        >>> print(expr.eval_latex())
        \sin(x^{2})
        """
        return eval_latex(self)

    def to_sympy(self) -> Any:
        """Convert to a SymPy expression.

        >>> from protosym.simplecas import sin, Symbol
        >>> x = Symbol('x')
        >>> expr_protosym = sin(x)
        >>> expr_protosym
        sin(x)
        >>> type(expr_protosym)
        <class 'protosym.simplecas.Expr'>

        >>> expr_sympy = expr_protosym.to_sympy()
        >>> expr_sympy
        sin(x)
        >>> type(expr_sympy)
        sin

        See Also
        ========

        from_sympy
        """
        return to_sympy(self)

    @classmethod
    def from_sympy(cls, expr: Any) -> Expr:
        """Create a simplecas ``Expr`` from a SymPy expression.

        >>> from sympy import sin, Symbol
        >>> x = Symbol('x')
        >>> expr_sympy = sin(x)
        >>> expr_sympy
        sin(x)
        >>> type(expr_sympy)
        sin

        >>> from protosym.simplecas import Expr
        >>> expr_protosym = Expr.from_sympy(expr_sympy)
        >>> expr_protosym
        sin(x)
        >>> type(expr_protosym)
        <class 'protosym.simplecas.Expr'>

        See Also
        ========

        to_sympy
        """
        return from_sympy(expr)

    def eval_f64(self, values: Optional[dict[Expr, float]] = None) -> float:
        """Evaluate the expression as a 64-bit ``float``.

        >>> from protosym.simplecas import Symbol, sin
        >>> expr1 = sin(sin(1))
        >>> expr1
        sin(sin(1))
        >>> expr1.eval_f64()
        0.7456241416655579

        If the expression contains symbols to be substituted then they can be
        provided as a dictionary of values:

        >>> x = Symbol('x')
        >>> expr2 = sin(sin(x))
        >>> expr2
        sin(sin(x))
        >>> expr2.eval_f64({x: 1.0})
        0.7456241416655579

        Python floats are based on IEEE 754 64-bit binary floating point which
        gives 53 bits of precision and a range of magnitudes approximately from
        :math:`10^{-300}` to :math:`10^{300}`.
        """
        return eval_f64(self, values)

    def count_ops_tree(self) -> int:
        """Count operations in ``Expr`` following tree representation.

        See :meth:`count_ops_graph` for an explanation.
        """
        return count_ops_tree(self)

    def count_ops_graph(self) -> int:
        """Count operations in ``Expr`` following tree representation.

        The number of operations in the *graph* representation of an expression
        is equal to the number of distinct subexpressions it has. By contrast
        the number of operations in the *tree* representation is equal to the
        number of subexpressions **counted with their multiplicity**. In other
        words the tree can have many repeating subexpressions whereas the graph
        will only contain one copy of each unique subexpression.

        We will make a function that can create large expressions and then
        count their operations:

        >>> from protosym.simplecas import Symbol
        >>> x = Symbol('x')
        >>> def make_expression(n):
        ...     e = x
        ...     for _ in range(n):
        ...         e = e**2 + e
        ...     return e

        For example this is what ``make_expression`` returns for small ``n``:

        >>> make_expression(1)
        (x**2 + x)
        >>> make_expression(2)
        ((x**2 + x)**2 + (x**2 + x))
        >>> make_expression(3)
        (((x**2 + x)**2 + (x**2 + x))**2 + ((x**2 + x)**2 + (x**2 + x)))

        We can see that these expressions have many repeating subexpressions.
        It is because of this that their *graph* representation will be a lot
        smaller than their *tree* representation.

        >>> for n in [1, 2, 5, 10, 20, 50, 100]:
        ...     expr = make_expression(n)
        ...     print(expr.count_ops_graph(), expr.count_ops_tree())
        4 5
        6 13
        12 125
        22 4093
        42 4194301
        102 4503599627370493
        202 5070602400912917605986812821501

        What we can see is the for this (very extreme) class of expressions as
        we increase ``n`` the size of the graph representation grows *linearly*
        as :math:`2n + 2` whereas the size of the tree representation grows
        *exponentially* as :math:`2^{n+2} - 3`. For this class of expressions the
        graph representation will be much more efficient than the tree
        representation both in terms of memory and also computing time.

        See Also
        ========

        count_ops_tree
        """
        return len(topological_sort(self.rep))

    def diff(self, sym: Expr, ntimes: int = 1) -> Expr:
        """Differentiate ``expr`` wrt ``sym`` (``ntimes`` times).

        >>> from protosym.simplecas import x, sin
        >>> sin(x).diff(x)
        cos(x)

        Currently no simplification is done which can lead to some strange
        looking output:

        >>> sin(x).diff(x, 4)
        (-1*(-1*sin(x)))

        Large expressions can be generated and differentiated efficiently:

        >>> expr = sin(sin(sin(sin(sin(x))))).diff(x, 10)
        >>> expr.count_ops_graph()
        1552
        >>> expr.count_ops_tree()
        893621974

        Notes
        =====

        Currently the differentiation algorithm is based on *forward
        accumulation* which is a common technique in the authomatic
        differentiation literature.

        References
        ==========

        https://en.wikipedia.org/wiki/Automatic_differentiation
        """
        deriv_rep = self.rep
        sym_rep = sym.rep
        for _ in range(ntimes):
            deriv_rep = _diff_forward(deriv_rep, sym_rep)
        return Expr(deriv_rep)

    def bin_expand(self) -> Expr:
        """Expand associative operators to binary operations.

        >>> from protosym.simplecas import Add
        >>> expr = Add(1, 2, 3, 4)
        >>> expr
        (1 + 2 + 3 + 4)
        >>> expr.bin_expand()
        (((1 + 2) + 3) + 4)
        """
        return Expr(bin_expand(self.rep))

    def to_llvm_ir(self, symargs: list[Expr]) -> str:
        """Return LLVM IR code evaluating this expression.

        >>> from protosym.simplecas import sin, x
        >>> expr = sin(x) + sin(sin(x))
        >>> print(expr.to_llvm_ir([x]))
        ; ModuleID = "mod1"
        target triple = "unknown-unknown-unknown"
        target datalayout = ""
        <BLANKLINE>
        declare double    @llvm.pow.f64(double %Val1, double %Val2)
        declare double    @llvm.sin.f64(double %Val)
        declare double    @llvm.cos.f64(double %Val)
        <BLANKLINE>
        define double @"jit_func1"(double %"x")
        {
        %".0" = call double @llvm.sin.f64(double %"x")
        %".1" = call double @llvm.sin.f64(double %".0")
        %".2" = fadd double %".0", %".1"
        ret double %".2"
        }

        """
        return _to_llvm_f64([arg.rep for arg in symargs], self.rep)


# Avoid importing SymPy if possible.
_eval_to_sympy: SymEvaluator[Expr, Any] | None = None


def _get_eval_to_sympy() -> SymEvaluator[Expr, Any]:
    """Return an evaluator for converting to SymPy."""
    global _eval_to_sympy
    if _eval_to_sympy is not None:
        return _eval_to_sympy

    import sympy

    eval_to_sympy = Expr.new_evaluator("to_sympy", object)
    eval_to_sympy[Integer[a]] = PyFunc1(sympy.Integer)(a)
    eval_to_sympy[Symbol[a]] = PyFunc1(sympy.Symbol)(a)
    eval_to_sympy[Function[a]] = PyFunc1(sympy.Function)(a)
    eval_to_sympy[Add(star(a))] = PyOpN[Any](lambda a: sympy.Add(*a))(a)
    eval_to_sympy[Mul(star(a))] = PyOpN[Any](lambda a: sympy.Mul(*a))(a)
    eval_to_sympy[a**b] = PyOp2(sympy.Pow)(a, b)
    eval_to_sympy[sin(a)] = PyOp1(sympy.sin)(a)
    eval_to_sympy[cos(a)] = PyOp1(sympy.cos)(a)
    eval_to_sympy[List(star(a))] = PyOpN(
        lambda args: sympy.Tuple(*args)  # type:ignore
    )(a)

    # Store in the global to avoid recreating the Evaluator
    _eval_to_sympy = eval_to_sympy

    return eval_to_sympy


def to_sympy(expr: Expr) -> Any:
    """Convert ``Expr`` to a SymPy expression."""
    eval_to_sympy = _get_eval_to_sympy()
    return eval_to_sympy(expr)


def from_sympy(expr: Any) -> Expr:
    """Convert a SymPy expression to ``Expr``."""
    import sympy

    return _from_sympy_cache(expr, sympy, {})


def _from_sympy_cache(expr: Any, sympy: Any, cache: dict[Any, Expr]) -> Expr:
    ret = cache.get(expr)
    if ret is not None:
        return ret
    elif expr.args:
        ret = _from_sympy_cache_args(expr, sympy, cache)
    elif expr.is_Integer:
        ret = Integer(expr.p)
    elif expr.is_Symbol:
        ret = Symbol(expr.name)
    else:
        raise NotImplementedError("Cannot convert " + type(expr).__name__)
    cache[expr] = ret
    return ret


def _from_sympy_cache_args(expr: Any, sympy: Any, cache: dict[Any, Expr]) -> Expr:
    args = [_from_sympy_cache(arg, sympy, cache) for arg in expr.args]
    if expr.is_Add:
        return Add(*args)
    elif expr.is_Mul:
        return Mul(*args)
    elif expr.is_Pow:
        return Pow(*args)
    elif isinstance(expr, sympy.sin):
        return sin(*args)
    elif isinstance(expr, sympy.cos):
        return cos(*args)
    elif isinstance(expr, sympy.Tuple):
        return List(*args)
    else:
        raise NotImplementedError("Cannot convert " + type(expr).__name__)


Integer = Expr.new_atom("Integer", int)
Symbol = Expr.new_atom("Symbol", str)
Function = Expr.new_atom("Function", str)

zero = Integer(0)
one = Integer(1)
negone = Integer(-1)

x = Symbol("x")
y = Symbol("y")

f = Function("f")
g = Function("g")

Pow = Function("pow")
sin = Function("sin")
cos = Function("cos")
Add = Function("Add")
Mul = Function("Mul")

a = Expr.new_wild("a")
b = Expr.new_wild("b")

# ------------------------------------------------------------------------- #
#                                                                           #
#     eval_f64: 64 bit floating point evaluation.                           #
#                                                                           #
# ------------------------------------------------------------------------- #

f64_from_int = PyFunc1[int, float](float)
f64_add = PyOpN[float](math.fsum)
f64_mul = PyOpN[float](math.prod)
f64_pow = PyOp2[float](math.pow)
f64_sin = PyOp1[float](math.sin)
f64_cos = PyOp1[float](math.cos)

eval_f64 = Expr.new_evaluator("eval_f64", float)
eval_f64[Integer[a]] = f64_from_int(a)
eval_f64[Add(star(a))] = f64_add(a)
eval_f64[Mul(star(a))] = f64_mul(a)
eval_f64[a**b] = f64_pow(a, b)
eval_f64[sin(a)] = f64_sin(a)
eval_f64[cos(a)] = f64_cos(a)

# ------------------------------------------------------------------------- #
#                                                                           #
#     eval_repr: Pretty string representation                               #
#                                                                           #
# ------------------------------------------------------------------------- #

repr_atom = AtomFunc[str](str)
repr_call = HeadOp[str](lambda head, args: f'{head}({", ".join(args)})')
str_from_int = PyFunc1[int, str](str)
str_from_str = PyFunc1[str, str](str)
repr_add = PyOpN[str](lambda args: f'({" + ".join(args)})')
repr_mul = PyOpN[str](lambda args: f'({"*".join(args)})')
repr_pow = PyOp2[str](lambda b, e: f"{b}**{e}")


eval_repr = Expr.new_evaluator("eval_repr", str)
eval_repr[HeadRule(a, b)] = repr_call(a, b)
eval_repr[AtomRule[a]] = repr_atom(a)
eval_repr[Integer[a]] = str_from_int(a)
eval_repr[Symbol[a]] = str_from_str(a)
eval_repr[Function[a]] = str_from_str(a)
eval_repr[Add(star(a))] = repr_add(a)
eval_repr[Mul(star(a))] = repr_mul(a)
eval_repr[a**b] = repr_pow(a, b)

# ------------------------------------------------------------------------- #
#                                                                           #
#     latex: LaTeX string representation                                    #
#                                                                           #
# ------------------------------------------------------------------------- #

latex_add = PyOpN(lambda args: f'({" + ".join(args)})')
latex_mul = PyOpN(lambda args: "(%s)" % r" \times ".join(args))
latex_pow = PyOp2(lambda b, e: f"{b}^{{{e}}}")
latex_sin = PyOp1(lambda a: rf"\sin({a})")
latex_cos = PyOp1(lambda a: rf"\cos({a})")

eval_latex = Expr.new_evaluator("eval_latex", str)
eval_latex[HeadRule(a, b)] = repr_call(a, b)
eval_latex[Integer[a]] = str_from_int(a)
eval_latex[Symbol[a]] = str_from_str(a)
eval_latex[Function[a]] = str_from_str(a)
eval_latex[Add(star(a))] = latex_add(a)
eval_latex[Mul(star(a))] = latex_mul(a)
eval_latex[a**b] = latex_pow(a, b)
eval_latex[sin(a)] = latex_sin(a)
eval_latex[cos(a)] = latex_cos(a)

# ------------------------------------------------------------------------- #
#                                                                           #
#     binexpand: Expand associative operators to binary operations.         #
#                                                                           #
# ------------------------------------------------------------------------- #

bin_expand = Transformer()
bin_expand.add_opn(Add.rep, lambda args: reduce(Add.rep, args))
bin_expand.add_opn(Mul.rep, lambda args: reduce(Mul.rep, args))

#
# Maybe it should be possible to just pass these as arguments to the Evaluator
# constructor.
#
one_func = AtomFunc[int](lambda a: 1)
sum_plus_one = HeadOp[int](lambda head, counts: 1 + sum(counts))

count_ops_tree = Expr.new_evaluator("count_ops_tree", int)
count_ops_tree[AtomRule[a]] = one_func(a)
count_ops_tree[HeadRule(a, b)] = sum_plus_one(a, b)


#
# We will need to think of a better structure for differentiation. Ideally it
# would be implemented as a generic routine in core but it really needs to know
# about Add, Mul, Pow, Integer etc so for now we implement it here. Probably
# what is needed is something like a differentiation "context" object that
# provides the necessary Add, Mul, Pow, zero, one etc that could be passed into
# the core differentiation routine along with the special case rules like
# sin->cos.
#
# Certainly differentiation is an example that blurs a bit the line between
# having a generic core that is agnostic to the kinds of expressions that we
# want to operate on and then defining everything else outside the core.
#


derivatives: dict[tuple[TreeExpr, int], Callable[..., TreeExpr]] = {
    (sin.rep, 0): cos.rep,
    (cos.rep, 0): lambda e: Mul.rep(negone.rep, sin.rep(e)),
    (Pow.rep, 0): lambda b, e: Mul.rep(e, Pow.rep(b, Add.rep(e, negone.rep))),
}


def _prod_rule_forward(
    args: list[TreeExpr], diff_args: list[TreeExpr]
) -> list[TreeExpr]:
    """Product rule in forward accumulation."""
    terms: list[TreeExpr] = []
    for n, diff_arg in enumerate(diff_args):
        if diff_arg != zero.rep:
            term = Mul.rep(*args[:n], diff_arg, *args[n + 1 :])
            terms.append(term)
    return terms


def _chain_rule_forward(
    func: TreeExpr, args: list[TreeExpr], diff_args: list[TreeExpr]
) -> list[TreeExpr]:
    """Chain rule in forward accumulation."""
    terms: list[TreeExpr] = []
    for n, diff_arg in enumerate(diff_args):
        if diff_arg != zero.rep:
            pdiff = derivatives[(func, n)]
            diff_term = pdiff(*args)
            if diff_arg != one.rep:
                diff_term = Mul.rep(diff_term, diff_arg)
            terms.append(diff_term)
    return terms


def _diff_forward(expression: TreeExpr, sym: TreeExpr) -> TreeExpr:
    """Derivative of expression wrt sym.

    Uses forward accumulation algorithm.
    """
    #
    # Using rep everywhere here shows that we are probably implementing this at
    # the wrong level.
    #

    graph = forward_graph(expression)

    stack = list(graph.atoms)
    diff_stack = [one.rep if expr == sym else zero.rep for expr in stack]

    for func, indices in graph.operations:
        args = [stack[i] for i in indices]
        diff_args = [diff_stack[i] for i in indices]
        expr = func(*args)

        if set(diff_args) == {zero.rep}:
            diff_terms = []
        elif func == Add.rep:
            diff_terms = [da for da in diff_args if da != zero.rep]
        elif func == Mul.rep:
            diff_terms = _prod_rule_forward(args, diff_args)
        elif func == List.rep:
            diff_terms = [List.rep(*diff_args)]
        else:
            diff_terms = _chain_rule_forward(func, args, diff_args)

        if not diff_terms:
            derivative = zero.rep
        elif len(diff_terms) == 1:
            derivative = diff_terms[0]
        else:
            derivative = Add.rep(*diff_terms)

        stack.append(expr)
        diff_stack.append(derivative)

    # At this point stack is a topological sort of expr and diff_stack is the
    # list of derivatives of every subexpression in expr. At the top of the
    # stack is expr and its derivative is at the top of diff_stack.
    return diff_stack[-1]


# -------------------------------------------------
# Matrices
# ------------------------------------------------


List = Function("List")


class Matrix:
    """Matrix of Expr."""

    nrows: int
    ncols: int
    shape: tuple[int, int]
    elements: list[Expr]
    elements_graph: Expr
    entrymap: dict[tuple[int, int], int]

    def __new__(cls, entries: Sequence[Sequence[Expressifiable]]) -> Matrix:
        """New Matrix from a list of lists."""
        if not isinstance(entries, list) or not all(
            isinstance(row, list) for row in entries
        ):
            raise TypeError("Input should be a list of lists.")

        nrows = len(entries)
        ncols = len(entries[0])
        if not all(len(row) == ncols for row in entries):
            raise TypeError("All rows should be the same length.")

        entries_expr = [[expressify(e) for e in row] for row in entries]

        elements: list[Expr] = []
        entrymap = {}
        for i, row in enumerate(entries_expr):
            for j, entry in enumerate(row):
                if entry != zero:
                    entrymap[(i, j)] = len(elements)
                    elements.append(entry)

        return cls._new(nrows, ncols, elements, entrymap)

    @classmethod
    def _new(
        cls,
        nrows: int,
        ncols: int,
        elements: list[Expr],
        entrymap: dict[tuple[int, int], int],
    ) -> Matrix:
        """New matrix from the internal representation."""
        obj = super().__new__(cls)
        obj.nrows = nrows
        obj.ncols = ncols
        obj.shape = (nrows, ncols)
        obj.elements = list(elements)
        obj.elements_graph = List(*elements)
        obj.entrymap = entrymap
        return obj

    def __getitem__(self, ij: tuple[int, int]) -> Expr:
        """Element indexing ``M[i, j]``."""
        if isinstance(ij, tuple) and len(ij) == 2:
            i, j = ij
            if isinstance(i, int) and isinstance(j, int):
                if not (0 <= i < self.nrows and 0 <= j < self.ncols):
                    raise IndexError("Indices out of bounds.")
                if ij in self.entrymap:
                    return self.elements[self.entrymap[ij]]
                else:
                    return zero
        raise TypeError("Matrix indices should be a pair of integers.")

    def tolist(self) -> list[list[Expr]]:
        """Convert to list of lists format."""
        entries = [[zero] * self.ncols for _ in range(self.nrows)]
        for (i, j), n in self.entrymap.items():
            entries[i][j] = self.elements[n]
        return entries

    def to_sympy(self) -> Any:
        """Convert a simplecas Matrix to a SymPy Matrix."""
        import sympy

        elements_sympy = to_sympy(self.elements_graph).args
        mat_sympy = sympy.zeros(self.nrows, self.ncols)
        for (i, j), n in self.entrymap.items():
            mat_sympy[i, j] = elements_sympy[n]
        return mat_sympy

    @classmethod
    def from_sympy(cls, mat: Any) -> Matrix:
        """Convert a SymPy Matrix to a simplecas Matrix."""
        import sympy

        dok = mat.todok()
        elements_sympy = []
        entrymap = {}
        for n, (key, sympy_expr) in enumerate(dok.items()):
            entrymap[key] = n
            elements_sympy.append(sympy_expr)
        elements = list(from_sympy(sympy.Tuple(*elements_sympy)).args)
        return cls._new(mat.rows, mat.cols, elements, entrymap)

    def __repr__(self) -> str:
        """Convert to pretty representation."""
        # Inefficient because does not use a graph...
        # (This computes separate repr for each element)
        return f"Matrix({self.tolist()!r})"

    def __add__(self, other: Matrix) -> Matrix:
        """Matrix addition A + B -> C."""
        if not isinstance(other, Matrix):
            return NotImplemented
        return self.binop(other, Add)

    def binop(self, other: Matrix, func: Expr) -> Matrix:
        """Elementwise binary operaton on two matrices."""
        if self.shape != other.shape:
            raise TypeError("Shape mismatch.")
        new_elements = self.elements.copy()
        new_entrymap = self.entrymap.copy()
        for ij, n_other in other.entrymap.items():
            if ij in new_entrymap:
                self_ij = new_elements[new_entrymap[ij]]
                other_ij = other.elements[n_other]
                result = func(self_ij, other_ij)
                new_elements[new_entrymap[ij]] = result
            else:
                new_entrymap[ij] = len(new_elements)
                new_elements.append(other.elements[n_other])
        return self._new(self.nrows, self.ncols, new_elements, new_entrymap)

    def diff(self, sym: Expr) -> Matrix:
        """Differentiate Matrix wrt ``sym``."""
        if not isinstance(sym, Expr):
            raise TypeError("Differentiation var should be a symbol.")
        # Use the element_graph rather than differentiating each element
        # separately.
        elements_diff = _diff_forward(self.elements_graph.rep, sym.rep)
        new_elements = list(Expr(elements_diff).args)
        return self._new(self.nrows, self.ncols, new_elements, self.entrymap)

    def to_llvm_ir(self, symargs: list[Expr]) -> str:
        """Return LLVM IR code evaluating this Matrix.

        >>> from protosym.simplecas import sin, cos, x, Matrix
        >>> M = Matrix([[cos(x), sin(x)], [-sin(x), cos(x)]])
        >>> print(M.to_llvm_ir([x]))
        ; ModuleID = "mod1"
        target triple = "unknown-unknown-unknown"
        target datalayout = ""
        <BLANKLINE>
        declare double    @llvm.pow.f64(double %Val1, double %Val2)
        declare double    @llvm.sin.f64(double %Val)
        declare double    @llvm.cos.f64(double %Val)
        <BLANKLINE>
        define void @"jit_func1"(double* %"_out", double %"x")
        {
        %".0" = call double @llvm.cos.f64(double %"x")
        %".1" = call double @llvm.sin.f64(double %"x")
        %".2" = fmul double 0xbff0000000000000, %".1"
        %".3" = getelementptr double, double* %"_out", i32 0
        store double %".0", double* %".3"
        %".4" = getelementptr double, double* %"_out", i32 1
        store double %".1", double* %".4"
        %".5" = getelementptr double, double* %"_out", i32 2
        store double %".2", double* %".5"
        %".6" = getelementptr double, double* %"_out", i32 3
        store double %".0", double* %".6"
        ret void
        }

        """
        return _to_llvm_f64_matrix([arg.rep for arg in symargs], self)


# -------------------------------------------------
# lambdification with LLVM
# ------------------------------------------------


class LLVMNotImplementedError(ProtoSymError):
    """Raised when an operation is not supported for LLVM."""

    pass


def _double_to_hex(f: float) -> str:
    return hex(struct.unpack("<Q", struct.pack("<d", f))[0])


_llvm_header = """
; ModuleID = "mod1"
target triple = "unknown-unknown-unknown"
target datalayout = ""

declare double    @llvm.pow.f64(double %Val1, double %Val2)
declare double    @llvm.sin.f64(double %Val)
declare double    @llvm.cos.f64(double %Val)

"""


def _to_llvm_f64(symargs: list[TreeExpr], expression: TreeExpr) -> str:
    """Code for LLVM IR function computing ``expression`` from ``symargs``."""
    expression = bin_expand(expression)

    graph = forward_graph(expression)

    argnames = {s: f'%"{s}"' for s in symargs}  # noqa

    identifiers = []
    for a in graph.atoms:
        if a in symargs:
            identifiers.append(argnames[a])
        elif isinstance(a, TreeAtom) and a.value.atom_type == Integer.atom_type:
            identifiers.append(_double_to_hex(a.value.value))
        else:
            raise LLVMNotImplementedError("No LLVM rule for: " + repr(a))

    args = ", ".join(f"double {argnames[arg]}" for arg in symargs)
    signature = f'define double @"jit_func1"({args})'

    instructions: list[str] = []
    for func, indices in graph.operations:
        n = len(instructions)
        identifier = f'%".{n}"'
        identifiers.append(identifier)
        argids = [identifiers[i] for i in indices]

        if func == Add.rep:
            line = f"{identifier} = fadd double " + ", ".join(argids)
        elif func == Mul.rep:
            line = f"{identifier} = fmul double " + ", ".join(argids)
        elif func == Pow.rep:
            args = f"double {argids[0]}, double {argids[1]}"
            line = f"{identifier} = call double @llvm.pow.f64({args})"
        elif func == sin.rep:
            line = f"{identifier} = call double @llvm.sin.f64(double {argids[0]})"
        elif func == cos.rep:
            line = f"{identifier} = call double @llvm.cos.f64(double {argids[0]})"
        else:
            raise LLVMNotImplementedError("No LLVM rule for: " + repr(func))

        instructions.append(line)

    instructions.append(f"ret double {identifiers[-1]}")

    function_lines = [signature, "{", *instructions, "}"]
    module_code = _llvm_header + "\n".join(function_lines)
    return module_code


def lambdify(args: list[Expr], expression: Expr | Matrix) -> Callable[..., Any]:
    """Turn ``expression`` into an efficient callable function of ``args``.

    >>> from protosym.simplecas import Symbol, sin, lambdify
    >>> x = Symbol('x')
    >>> f = lambdify([x], sin(x))
    >>> f(1)
    0.8414709848078965
    >>> import math; math.sin(1)
    0.8414709848078965
    """
    args_rep = [arg.rep for arg in args]
    if isinstance(expression, Expr):
        return _lambdify_llvm(args_rep, expression.rep)
    elif isinstance(expression, Matrix):
        return _lambdify_llvm_matrix(args_rep, expression)
    else:
        raise TypeError("Expression should be Expr or Matrix.")


_exe_eng = []


def _compile_llvm(module_code: str) -> Any:
    try:
        import llvmlite.binding as llvm
    except ImportError:  # pragma: no cover
        msg = "llvmlite needs to be installed to use lambdify_llvm."
        raise ImportError(msg) from None

    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    llmod = llvm.parse_assembly(module_code)

    pmb = llvm.create_pass_manager_builder()
    pmb.opt_level = 2
    pass_manager = llvm.create_module_pass_manager()
    pmb.populate(pass_manager)

    pass_manager.run(llmod)

    target_machine = llvm.Target.from_default_triple().create_target_machine()
    exe_eng = llvm.create_mcjit_compiler(llmod, target_machine)
    exe_eng.finalize_object()
    _exe_eng.append(exe_eng)

    fptr = exe_eng.get_function_address("jit_func1")
    return fptr


def _lambdify_llvm(args: list[TreeExpr], expression: TreeExpr) -> Callable[..., float]:
    """Lambdify using llvmlite."""
    import ctypes

    module_code = _to_llvm_f64(args, expression)

    fptr = _compile_llvm(module_code)

    rettype = ctypes.c_double
    argtypes = [ctypes.c_double] * len(args)

    cfunc = ctypes.CFUNCTYPE(rettype, *argtypes)(fptr)
    return cfunc


def _to_llvm_f64_matrix(symargs: list[TreeExpr], mat: Matrix) -> str:  # noqa [C901]
    """Code for LLVM IR function computing ``expression`` from ``symargs``."""
    elements_graph = bin_expand(mat.elements_graph.rep)

    graph = forward_graph(elements_graph)

    argnames = {s: f'%"{s}"' for s in symargs}  # noqa

    identifiers = []
    for a in graph.atoms:
        if a in symargs:
            identifiers.append(argnames[a])
        elif isinstance(a, TreeAtom) and a.value.atom_type == Integer.atom_type:
            identifiers.append(_double_to_hex(a.value.value))
        else:
            raise LLVMNotImplementedError("No LLVM rule for: " + repr(a))

    all_args = ['double* %"_out"'] + [f"double {argnames[arg]}" for arg in symargs]
    all_args_str = ", ".join(all_args)
    signature = f'define void @"jit_func1"({all_args_str})'

    instructions: list[str] = []
    for func, indices in graph.operations[:-1]:
        n = len(instructions)
        identifier = f'%".{n}"'
        identifiers.append(identifier)
        argids = [identifiers[i] for i in indices]

        if func == Add.rep:
            line = f"{identifier} = fadd double " + ", ".join(argids)
        elif func == Mul.rep:
            line = f"{identifier} = fmul double " + ", ".join(argids)
        elif func == Pow.rep:
            args = f"double {argids[0]}, double {argids[1]}"
            line = f"{identifier} = call double @llvm.pow.f64({args})"
        elif func == sin.rep:
            line = f"{identifier} = call double @llvm.sin.f64(double {argids[0]})"
        elif func == cos.rep:
            line = f"{identifier} = call double @llvm.cos.f64(double {argids[0]})"
        else:
            raise LLVMNotImplementedError("No LLVM rule for: " + repr(func))

        instructions.append(line)

    # The above loop stops short of the final operation which should be the
    # List at the top of the stack. Now all values are computed and just need
    # to be copied to the relevant locations in the _out array.
    _, indices = graph.operations[-1]

    ncols = mat.ncols
    identifier_count = len(instructions)
    for (i, j), n in sorted(mat.entrymap.items()):
        raw_index = i * ncols + j
        identifier_value = identifiers[indices[n]]
        ptr = f'%".{identifier_count}"'
        identifier_count += 1
        line1 = f'{ptr} = getelementptr double, double* %"_out", i32 {raw_index}'
        line2 = f"store double {identifier_value}, double* {ptr}"
        instructions.append(line1)
        instructions.append(line2)

    instructions.append("ret void")

    function_lines = [signature, "{", *instructions, "}"]
    module_code = _llvm_header + "\n".join(function_lines)
    return module_code


def _lambdify_llvm_matrix(args: list[TreeExpr], mat: Matrix) -> Callable[..., Any]:
    """Lambdify a matrix.

    >>> from protosym.simplecas import lambdify, Matrix
    >>> f = lambdify([], Matrix([[1, 2], [3, 4]]))
    >>> f()
    array([[1., 2.],
           [3., 4.]])
    """
    import ctypes

    module_code = _to_llvm_f64_matrix(args, mat)

    fptr = _compile_llvm(module_code)

    c_float64 = ctypes.POINTER(ctypes.c_double)
    rettype = ctypes.c_double
    argtypes = [c_float64] + [ctypes.c_double] * len(args)

    cfunc = ctypes.CFUNCTYPE(rettype, *argtypes)(fptr)

    import numpy as np

    def npfunc(*args: float) -> Any:
        arr = np.zeros(mat.shape, np.float64)
        arr_p = arr.ctypes.data_as(c_float64)
        cfunc(arr_p, *args)
        return arr

    return npfunc
