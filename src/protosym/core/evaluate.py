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


__all__ = ["Evaluator"]


_T = TypeVar("_T")
_S = TypeVar("_S")


if _TYPE_CHECKING:
    from typing import Optional, Iterable

    op1 = Callable[[_T], _T]
    op2 = Callable[[_T, _T], _T]
    opN = Callable[[Iterable[_T]], _T]  # noqa


class Evaluator(Generic[_T]):
    """Objects that evaluate expressions."""

    atoms: dict[AtomType[_AnyValue], Callable[[_AnyValue], _T]]
    operations: dict[TreeExpr, tuple[Callable[..., _T], bool]]

    def __init__(self) -> None:
        """Create an empty evaluator."""
        self.atoms = {}
        self.operations = {}

    def add_atom(self, atom_type: AtomType[_S], func: Callable[[_S], _T]) -> None:
        """Add an evaluation rule for a particular AtomType."""
        atom_type_cast = cast(AtomType[_AnyValue], atom_type)
        func_cast = cast(Callable[[_AnyValue], _T], func)
        self.atoms[atom_type_cast] = func_cast

    def add_op1(self, head: TreeExpr, func: op1[_T]) -> None:
        """Add an evaluation rule for a particular head."""
        self.operations[head] = (func, True)

    def add_op2(self, head: TreeExpr, func: op2[_T]) -> None:
        """Add an evaluation rule for a particular head."""
        self.operations[head] = (func, True)

    def add_opN(self, head: TreeExpr, func: opN[_T]) -> None:  # noqa
        """Add an evaluation rule for a particular head."""
        self.operations[head] = (func, False)

    def call(self, head: TreeExpr, argvals: Iterable[_T]) -> _T:
        """Evaluate one function with some values."""
        op_func, star_args = self.operations[head]
        if star_args:
            result = op_func(*argvals)
        else:
            result = op_func(argvals)
        return result

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
            childvals = [self.evaluate(c, values) for c in children]
            return self.call(head, childvals)

    def __call__(
        self, expr: TreeExpr, values: Optional[dict[TreeExpr, _T]] = None
    ) -> _T:
        """Short-hand for evaluate."""
        if values is None:
            values = {}
        return self.evaluate(expr, values)
