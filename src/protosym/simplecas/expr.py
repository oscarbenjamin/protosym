"""The Expr class."""
from __future__ import annotations

from functools import reduce, wraps
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any, Callable, TypeVar, Union

from protosym.core.evaluate import Transformer
from protosym.core.sym import (
    AtomFunc,
    AtomRule,
    HeadOp,
    HeadRule,
    Sym,
    SymDifferentiator,
)
from protosym.core.tree import SubsFunc, topological_sort
from protosym.simplecas.exceptions import ExpressifyError

T_sym = TypeVar("T_sym", bound=Sym)


if _TYPE_CHECKING:
    from protosym.core.tree import Tree

    Expressifiable = Union["Expr", int]
    ExprBinOp = Callable[["Expr", "Expr"], "Expr"]
    ExpressifyBinOp = Callable[["Expr", Expressifiable], "Expr"]


def expressify(obj: Any) -> Expr:
    """Convert a native Python object to an ``Expr``.

    >>> from protosym.simplecas import expressify
    >>> one = expressify(1)
    >>> one
    1
    >>> type(one)
    <class 'protosym.simplecas.expr.Expr'>

    This is the internal representation of the ``one`` object returned by
    :func:`expressify`:

    >>> one.rep
    Tr(Integer(1))

    It is harmless to call :func:`expressify` more than once because it will
    just return the same object.

    >>> expressify(one) is one
    True

    Notes
    -----
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


def expressify_other(method: ExprBinOp) -> ExpressifyBinOp:
    """Call ``expressify`` on operands in ``__add__`` etc."""

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

    An :class:`Expr` directly displays its internal form which can be
    surprising e.g.:

    >>> x - y
    (x + (-1*y))

    This is in fact the form that many algebra systems use to represent ``x -
    y`` although often the resulting expression would be displayed differently.

    The :class:`Expr` class has many methods some of which are listed below.

    See Also
    --------
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
        """Support IPython's LaTeX hook."""
        return f"${self.eval_latex()}$"

    def _sympy_(self) -> Any:
        """Support SymPy's ``sympify`` function."""
        return self.to_sympy()

    def __call__(self, *args: Expressifiable) -> Expr:
        """Call this Expr as a function."""
        args_expr = [expressify(arg) for arg in args]
        args_rep = [arg.rep for arg in args_expr]
        return Expr(self.rep(*args_rep))

    def xreplace(self, reps: dict[Expressifiable, Expressifiable]) -> Expr:
        """Replace subexpressions in an :class:`Expr`.

        >>> from protosym.simplecas import cos, x, y
        >>> e = cos(x) + x
        >>> e.xreplace({x:y})
        (cos(y) + y)
        >>> e.xreplace({cos(x): x, x: y})
        (x + y)
        """
        func = self.as_function(*reps)
        return func(*reps.values())

    def as_function(self, *args: Expressifiable) -> ExprFunction:
        """Make a callable :class:`ExprFunction` out of this expression.

        >>> from protosym.simplecas import cos, x, y
        >>> expr = cos(x) + y
        >>> func = expr.as_function(x, y)
        >>> func(1, 2)
        (cos(1) + 2)
        """
        return ExprFunction(self, args)

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
        <class 'protosym.simplecas.expr.Expr'>

        >>> # xdoctest: +REQUIRES(module:sympy)
        >>> expr_sympy = expr_protosym.to_sympy()
        >>> expr_sympy
        sin(x)
        >>> type(expr_sympy)
        sin

        See Also
        --------
        from_sympy
        """
        from protosym.simplecas.sympy_conversions import to_sympy

        return to_sympy(self)

    @classmethod
    def from_sympy(cls, expr: Any) -> Expr:
        """Create a simplecas ``Expr`` from a SymPy expression.

        >>> # xdoctest: +REQUIRES(module:sympy)
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
        <class 'protosym.simplecas.expr.Expr'>

        See Also
        --------
        to_sympy
        """
        from protosym.simplecas.sympy_conversions import from_sympy

        return from_sympy(expr)

    def eval_f64(self, values: dict[Expr, float] | None = None) -> float:
        """Evaluate the expression as a 64-bit ``float``.

        >>> from protosym.simplecas import Symbol, sin
        >>> expr1 = sin(sin(1))
        >>> expr1
        sin(sin(1))
        >>> expr1.eval_f64()  # doctest: +ELLIPSIS
        0.7456241416655...

        If the expression contains symbols to be substituted then they can be
        provided as a dictionary of values:

        >>> x = Symbol('x')
        >>> expr2 = sin(sin(x))
        >>> expr2
        sin(sin(x))
        >>> expr2.eval_f64({x: 1.0})  # doctest: +ELLIPSIS
        0.7456241416655...

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
        *exponentially* as :math:`2^{n+2} - 3`. For this class of expressions
        the graph representation will be much more efficient than the tree
        representation both in terms of memory and also computing time.

        See Also
        --------
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

        Differentiation rules for new functions can be added as needed:

        >>> from protosym.simplecas import Expr, Function, diff
        >>> a = Expr.new_wild('a')
        >>> tan = Function('tan')
        >>> diff[tan(a), a] = 1 + tan(a)**2
        >>> tan(tan(x)).diff(x)
        ((1 + tan(tan(x))**2)*(1 + tan(x)**2))

        Notes
        -----
        Currently the differentiation algorithm is based on *forward
        accumulation* which is a common technique in the automatic
        differentiation literature.

        See Also
        --------
        diff_forward

        References
        ----------
        https://en.wikipedia.org/wiki/Automatic_differentiation
        """
        deriv = self
        for _ in range(ntimes):
            deriv = diff(deriv, sym)
        return deriv

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
        from protosym.simplecas.lambdification import _to_llvm_f64

        return _to_llvm_f64([arg.rep for arg in symargs], self.rep)


class ExprFunction:
    """Function that rebuilds a symbolic expression.

    See Also
    --------
    Expr.as_function: the usual way to create an :class:`ExprFunction`.
    """

    def __init__(self, expr: Expressifiable, params: tuple[Expressifiable, ...]):
        """Create a new :class:`ExprFunction`."""
        expr_rep = expressify(expr).rep
        params_rep = [expressify(par).rep for par in params]
        self.func = SubsFunc(expr_rep, params_rep)

    def __call__(self, *args: Expressifiable) -> Expr:
        """Call this :class:`ExprFunction` with arguments."""
        args_rep = [expressify(arg).rep for arg in args]
        return Expr(self.call(args_rep))

    def call(self, args: list[Tree]) -> Tree:
        """Call this :class:`ExprFunction` with :class:`Tree` arguments."""
        return self.func(*args)


eval_f64 = Expr.new_evaluator("eval_f64", float)
eval_repr = Expr.new_evaluator("eval_repr", str)
eval_latex = Expr.new_evaluator("eval_latex", str)

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
List = Function("List")

bin_expand = Transformer()
bin_expand.add_opn(Add.rep, lambda args: reduce(Add.rep, args))
bin_expand.add_opn(Mul.rep, lambda args: reduce(Mul.rep, args))

#
# An evaluator to count the size of an expression tree.
#
a = Expr.new_wild("a")
b = Expr.new_wild("b")

one_func = AtomFunc[int](lambda _: 1)
sum_plus_one = HeadOp[int](lambda _, counts: 1 + sum(counts))

count_ops_tree = Expr.new_evaluator("count_ops_tree", int)
count_ops_tree[AtomRule[a]] = one_func(a)
count_ops_tree[HeadRule(a, b)] = sum_plus_one(a, b)

#
# Differentiation.
#

diff = SymDifferentiator(Expr, add=Add, mul=Mul, zero=zero, one=one)

diff[sin(a), a] = cos(a)
diff[cos(a), a] = -sin(a)
diff[a**b, a] = b * a ** (b + (-1))  # what if b=0?
diff.add_distributive_rule(List)
