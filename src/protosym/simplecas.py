"""Demonstration of putting together a simple CAS."""
from __future__ import annotations

import math
from typing import Generic
from typing import Type
from typing import TypeVar

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

    def __init__(self, tree_expr: TreeExpr) -> None:
        """Create an Expr from a TreeExpr."""
        if not isinstance(tree_expr, TreeExpr):
            raise TypeError("First argument to Expr should be TreeExpr")
        self.rep = tree_expr

    def __repr__(self) -> str:
        """Pretty string representation of the expression."""
        return str(self.rep)

    @classmethod
    def new_atom(cls, name: str, typ: Type[T]) -> ExprAtomType[T]:
        """Define a new AtomType."""
        return ExprAtomType[T](name, typ)

    def __call__(self, *args: Expr) -> Expr:
        """Call this Expr as a function."""
        args_rep = [arg.rep for arg in args]
        return Expr(self.rep(*args_rep))


Integer = Expr.new_atom("Integer", int)
Symbol = Expr.new_atom("Symbol", str)
Function = Expr.new_atom("Function", str)

zero = Integer(0)
one = Integer(1)

x = Symbol("x")
y = Symbol("y")

sin = Function("sin")
cos = Function("cos")

eval_f64 = Evaluator[float]()
eval_f64.add_atom(Integer.atom_type, float)
eval_f64.add_operation(sin.rep, math.sin)
eval_f64.add_operation(cos.rep, math.cos)

if __name__ == "__main__":  # pragma: no cover
    expr = sin(cos(one))
    print(expr, "=", eval_f64(expr.rep))
