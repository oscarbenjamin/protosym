"""Define the core evaluation code."""
from __future__ import annotations

from typing import Callable
from typing import cast
from typing import Generic
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import TypeVar

from protosym.core.atom import AnyValue as _AnyValue
from protosym.core.atom import AtomType
from protosym.core.tree import TreeAtom
from protosym.core.tree import TreeExpr


if _TYPE_CHECKING:
    from typing import Optional


__all__ = ["Evaluator"]


_T = TypeVar("_T")
_S = TypeVar("_S")


class Evaluator(Generic[_T]):
    """Objects that evaluate expressions."""

    atoms: dict[AtomType[_AnyValue], Callable[[_AnyValue], _T]]
    operations: dict[TreeExpr, Callable[[_T], _T]]

    def __init__(self) -> None:
        """Create an empty evaluator."""
        self.atoms = {}
        self.operations = {}

    def add_atom(self, atom_type: AtomType[_S], func: Callable[[_S], _T]) -> None:
        """Add an evaluation rule for a particular AtomType."""
        atom_type_cast = cast(AtomType[_AnyValue], atom_type)
        func_cast = cast(Callable[[_AnyValue], _T], func)
        self.atoms[atom_type_cast] = func_cast

    def add_operation(self, head: TreeExpr, func: Callable[[_T], _T]) -> None:
        """Add an evaluation rule for a particular head."""
        self.operations[head] = func

    def evaluate(self, expr: TreeExpr, values: dict[TreeExpr, _T]) -> _T:
        """Evaluate the expression using the rules."""
        if expr in values:
            return values[expr]
        elif isinstance(expr, TreeAtom):
            value = expr.value
            atom_func = self.atoms[value.atom_type]
            return atom_func(value.value)
        else:
            head = expr.children[0]
            children = expr.children[1:]
            op_func = self.operations[head]
            childvals = [self.evaluate(c, values) for c in children]
            return op_func(*childvals)

    def __call__(
        self, expr: TreeExpr, values: Optional[dict[TreeExpr, _T]] = None
    ) -> _T:
        """Short-hand for evaluate."""
        if values is None:
            values = {}
        return self.evaluate(expr, values)
