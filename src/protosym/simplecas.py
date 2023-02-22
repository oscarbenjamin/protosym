"""Demonstration of putting together a simple CAS."""
from __future__ import annotations

import math
from typing import Any
from typing import Generic
from typing import Optional
from typing import Type
from typing import TypeVar
from weakref import WeakValueDictionary as _WeakDict

from protosym.core.atom import AtomType
from protosym.core.evaluate import Evaluator
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

    @classmethod
    def new_atom(cls, name: str, typ: Type[T]) -> ExprAtomType[T]:
        """Define a new AtomType."""
        return ExprAtomType[T](name, typ)

    def __call__(self, *args: Expr) -> Expr:
        """Call this Expr as a function."""
        args_rep = [arg.rep for arg in args]
        return Expr(self.rep(*args_rep))

    def __pos__(self) -> Expr:
        """+Expr -> Expr."""
        return self

    def __neg__(self) -> Expr:
        """+Expr -> Expr."""
        return Mul(negone, self)

    def __add__(self, other: Expr) -> Expr:
        """Expr + Expr -> Expr."""
        if not isinstance(other, Expr):
            return NotImplemented
        return Add(self, other)

    def __sub__(self, other: Expr) -> Expr:
        """Expr - Expr -> Expr."""
        if not isinstance(other, Expr):
            return NotImplemented
        return Add(self, Mul(negone, other))

    def __mul__(self, other: Expr) -> Expr:
        """Expr * Expr -> Expr."""
        if not isinstance(other, Expr):
            return NotImplemented
        return Mul(self, other)

    def __truediv__(self, other: Expr) -> Expr:
        """Expr / Expr -> Expr."""
        if not isinstance(other, Expr):
            return NotImplemented
        return Mul(self, Pow(other, negone))

    def __pow__(self, other: Expr) -> Expr:
        """Expr ** Expr -> Expr."""
        if not isinstance(other, Expr):
            return NotImplemented
        return Pow(self, other)

    def eval_repr(self) -> str:
        """Pretty string e.g. "cos(x) + 1"."""
        return eval_repr(self.rep)

    def eval_f64(self, values: Optional[dict[Expr, float]] = None) -> float:
        """Evaluate the expression as a float."""
        values_rep = {}
        if values is not None:
            values_rep = {e.rep: v for e, v in values.items()}
        return eval_f64(self.rep, values_rep)


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
eval_f64.add_opN(Add.rep, math.fsum)
eval_f64.add_opN(Mul.rep, math.prod)

eval_repr = Evaluator[str]()
eval_repr.add_atom(Integer.atom_type, str)
eval_repr.add_atom(Symbol.atom_type, str)
eval_repr.add_atom(Function.atom_type, str)
eval_repr.add_op1(sin.rep, lambda a: f"sin({a})")
eval_repr.add_op1(cos.rep, lambda a: f"cos({a})")
eval_repr.add_op2(Pow.rep, lambda b, e: f"{b}**{e}")
eval_repr.add_opN(Add.rep, lambda args: f'({" + ".join(args)})')
eval_repr.add_opN(Mul.rep, lambda args: f'({"*".join(args)})')
