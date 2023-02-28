"""Demonstration of putting together a simple CAS."""
from __future__ import annotations

import math
from functools import reduce
from functools import wraps
from typing import Any
from typing import Callable
from typing import Generic
from typing import Iterable
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
    """Convert a native object to an ``Expr``."""
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
    """User-facing class for representing expressions."""

    _all_expressions: _WeakDict[Any, Any] = _WeakDict()

    rep: TreeExpr

    def __new__(cls, tree_expr: TreeExpr) -> Expr:
        """Create an Expr from a TreeExpr."""
        if not isinstance(tree_expr, TreeExpr):
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
        """Return a LaTeX representaton of the expression."""
        return eval_latex(self.rep)

    def to_sympy(self) -> Any:
        """Convert to a SymPy expression."""
        return to_sympy(self)

    @classmethod
    def from_sympy(cls, expr: Any) -> Expr:
        """Create an ``Expr`` from a SymPy expression."""
        return from_sympy(expr)

    def eval_f64(self, values: Optional[dict[Expr, float]] = None) -> float:
        """Evaluate the expression as a float."""
        values_rep = {}
        if values is not None:
            values_rep = {e.rep: v for e, v in values.items()}
        return eval_f64(self.rep, values_rep)

    def count_ops_tree(self) -> int:
        """Count operations in ``Expr`` following tree representation."""
        return count_ops_tree(self.rep)

    def count_ops_graph(self) -> int:
        """Count operations in ``Expr`` following tree representation."""
        return len(topological_sort(self.rep))

    def diff(self, sym: Expr, ntimes: int = 1) -> Expr:
        """Differentiate ``expr`` wrt ``sym``.

        >>> from protosym.simplecas import x, sin
        >>> sin(x).diff(x)
        cos(x)
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
        return Expr(_bin_expand(self.rep))


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
eval_repr.add_op1(sin.rep, lambda a: f"sin({a})")
eval_repr.add_op1(cos.rep, lambda a: f"cos({a})")
eval_repr.add_op2(Pow.rep, lambda b, e: f"{b}**{e}")
eval_repr.add_opn(Add.rep, lambda args: f'({" + ".join(args)})')
eval_repr.add_opn(Mul.rep, lambda args: f'({"*".join(args)})')

eval_latex = Evaluator[str]()
eval_latex.add_atom(Integer.atom_type, str)
eval_latex.add_atom(Symbol.atom_type, str)
eval_latex.add_atom(Function.atom_type, str)
eval_latex.add_op1(sin.rep, lambda a: rf"\sin({a})")
eval_latex.add_op1(cos.rep, lambda a: rf"\cos({a})")
eval_latex.add_op2(Pow.rep, lambda b, e: f"{b}^{{{e}}}")
eval_latex.add_opn(Add.rep, lambda args: f'({" + ".join(args)})')
eval_latex.add_opn(Mul.rep, lambda args: "(%s)" % r" \times ".join(args))

_bin_expand = Transformer()
_bin_expand.add_opn(Add.rep, lambda args: reduce(Add.rep, args))
_bin_expand.add_opn(Mul.rep, lambda args: reduce(Mul.rep, args))


def _op1(a: int | str) -> int:
    return 1


def _sum1(*a: int) -> int:
    return 1 + sum(a)


def _sum1n(a: Iterable[int]) -> int:
    return 1 + sum(a)


count_ops_tree = Evaluator[int]()
count_ops_tree.add_atom(Integer.atom_type, _op1)
count_ops_tree.add_atom(Symbol.atom_type, _op1)
count_ops_tree.add_op1(sin.rep, _sum1)
count_ops_tree.add_op1(cos.rep, _sum1)
count_ops_tree.add_op2(Pow.rep, _sum1)
count_ops_tree.add_opn(Add.rep, _sum1n)
count_ops_tree.add_opn(Mul.rep, _sum1n)


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
