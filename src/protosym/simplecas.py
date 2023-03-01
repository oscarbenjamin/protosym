"""Demonstration of putting together a simple CAS."""
from __future__ import annotations

import math
from functools import reduce
from functools import wraps
from typing import Any
from typing import Callable
from typing import Generic
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import TypeVar
from typing import Union
from weakref import WeakValueDictionary as _WeakDict

from protosym.core.atom import AtomType
from protosym.core.evaluate import Evaluator
from protosym.core.evaluate import Transformer
from protosym.core.tree import forward_graph
from protosym.core.tree import topological_sort
from protosym.core.tree import TreeAtom
from protosym.core.tree import TreeExpr


T = TypeVar("T")


class ExprAtomType(Generic[T]):
    """Wrapper around AtomType to construct atoms as Expr."""

    name: str
    atom_type: AtomType[T]

    def __init__(self, name: str, typ: Type[T]) -> None:
        """New ExprAtomType."""
        self.name = name
        self.atom_type = AtomType(name, typ)

    def __repr__(self) -> str:
        """The name of the ExprAtomType."""
        return self.name

    def __call__(self, value: T) -> Expr:
        """Create a new Atom as an Expr."""
        atom = self.atom_type(value)
        return Expr(TreeAtom[T](atom))


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


class Expr:
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

    An :class:`Expr` display directly its internal form which can be surprising e.g.:

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

    _all_expressions: _WeakDict[Any, Any] = _WeakDict()

    rep: TreeExpr

    def __new__(cls, tree_expr: TreeExpr) -> Expr:
        """Create an Expr from a TreeExpr."""
        if not isinstance(tree_expr, TreeExpr):
            # Maybe call expressify here?
            raise TypeError("First argument to Expr should be TreeExpr")

        key = tree_expr

        expr = cls._all_expressions.get(tree_expr, None)
        if expr is not None:
            return expr  # type:ignore

        obj = super().__new__(cls)
        obj.rep = tree_expr

        obj = cls._all_expressions.setdefault(key, obj)

        return obj

    def __repr__(self) -> str:
        """Pretty string representation of the expression."""
        return self.eval_repr()

    def _repr_latex_(self) -> str:
        """Latex hook for IPython."""
        return f"${self.eval_latex()}$"

    def _sympy_(self) -> Any:
        """Hook for SymPy's ``sympify`` function."""
        return self.to_sympy()

    @classmethod
    def new_atom(cls, name: str, typ: Type[T]) -> ExprAtomType[T]:
        """Define a new AtomType."""
        return ExprAtomType[T](name, typ)

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
        return eval_repr(self.rep)

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
        return eval_latex(self.rep)

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
        values_rep = {}
        if values is not None:
            values_rep = {e.rep: v for e, v in values.items()}
        return eval_f64(self.rep, values_rep)

    def count_ops_tree(self) -> int:
        """Count operations in ``Expr`` following tree representation.

        See :meth:`count_ops_graph` for an explanation.
        """
        return count_ops_tree(self.rep)

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
        as :math:`2n + 2` where as the size of the tree representation grows
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

        Large expressions can be generated and diferentiated efficiently:

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


# Avoid importing SymPy if possible.
_eval_to_sympy: Evaluator[Any] | None = None


def _get_eval_to_sympy() -> Evaluator[Any]:
    """Return an evaluator for converting to SymPy."""
    global _eval_to_sympy
    if _eval_to_sympy is not None:
        return _eval_to_sympy

    import sympy

    eval_to_sympy = Evaluator[Any]()
    eval_to_sympy.add_atom(Integer.atom_type, sympy.Integer)
    eval_to_sympy.add_atom(Symbol.atom_type, sympy.Symbol)
    eval_to_sympy.add_atom(Function.atom_type, sympy.Function)
    eval_to_sympy.add_op1(sin.rep, sympy.sin)
    eval_to_sympy.add_op1(cos.rep, sympy.cos)
    eval_to_sympy.add_op2(Pow.rep, sympy.Pow)
    eval_to_sympy.add_opn(Add.rep, lambda a: sympy.Add(*a))
    eval_to_sympy.add_opn(Mul.rep, lambda a: sympy.Mul(*a))

    # Store in the global to avoid recreating the Evaluator
    _eval_to_sympy = eval_to_sympy

    return eval_to_sympy


def to_sympy(expr: Expr) -> Any:
    """Convert ``Expr`` to a SymPy expression."""
    eval_to_sympy = _get_eval_to_sympy()
    return eval_to_sympy(expr.rep)


def from_sympy(expr: Any) -> Expr:
    """Convert a SymPy expression to ``Expr``."""
    import sympy

    if expr.is_Integer:
        return Integer(expr.p)
    elif expr.is_Symbol:
        return Symbol(expr.name)
    elif expr.args:
        args = [from_sympy(arg) for arg in expr.args]
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

eval_f64 = Evaluator[float]()
eval_f64.add_atom(Integer.atom_type, float)
eval_f64.add_op1(sin.rep, math.sin)
eval_f64.add_op1(cos.rep, math.cos)
eval_f64.add_op2(Pow.rep, pow)
eval_f64.add_opn(Add.rep, math.fsum)
eval_f64.add_opn(Mul.rep, math.prod)

eval_repr = Evaluator[str]()
eval_repr.add_atom(Integer.atom_type, str)
eval_repr.add_atom(Symbol.atom_type, str)
eval_repr.add_atom(Function.atom_type, str)
eval_repr.add_op_generic(lambda head, args: f'{head}({", ".join(args)})')
eval_repr.add_op1(sin.rep, lambda a: f"sin({a})")
eval_repr.add_op1(cos.rep, lambda a: f"cos({a})")
eval_repr.add_op2(Pow.rep, lambda b, e: f"{b}**{e}")
eval_repr.add_opn(Add.rep, lambda args: f'({" + ".join(args)})')
eval_repr.add_opn(Mul.rep, lambda args: f'({"*".join(args)})')

eval_latex = Evaluator[str]()
eval_latex.add_atom(Integer.atom_type, str)
eval_latex.add_atom(Symbol.atom_type, str)
eval_latex.add_atom(Function.atom_type, str)
eval_latex.add_op_generic(lambda head, args: f'{head}({", ".join(args)})')
eval_latex.add_op1(sin.rep, lambda a: rf"\sin({a})")
eval_latex.add_op1(cos.rep, lambda a: rf"\cos({a})")
eval_latex.add_op2(Pow.rep, lambda b, e: f"{b}^{{{e}}}")
eval_latex.add_opn(Add.rep, lambda args: f'({" + ".join(args)})')
eval_latex.add_opn(Mul.rep, lambda args: "(%s)" % r" \times ".join(args))

bin_expand = Transformer()
bin_expand.add_opn(Add.rep, lambda args: reduce(Add.rep, args))
bin_expand.add_opn(Mul.rep, lambda args: reduce(Mul.rep, args))

#
# Maybe it should be possible to just pass these as arguments to the Evaluator
# constructor.
#
count_ops_tree = Evaluator[int]()
count_ops_tree.add_atom_generic(lambda atom: 1)
count_ops_tree.add_op_generic(lambda head, argcounts: 1 + sum(argcounts))


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
