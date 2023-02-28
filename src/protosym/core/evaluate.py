"""Define the core evaluation code."""
from __future__ import annotations

from typing import Callable
from typing import cast
from typing import Generic
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import TypeVar

from protosym.core.exceptions import NoEvaluationRuleError
from protosym.core.tree import forward_graph
from protosym.core.tree import TreeAtom
from protosym.core.tree import TreeExpr


if _TYPE_CHECKING:
    from protosym.core.atom import AnyValue as _AnyValue
    from protosym.core.atom import AtomType


__all__ = ["Evaluator"]


_T = TypeVar("_T")
_S = TypeVar("_S")


if _TYPE_CHECKING:
    from typing import Optional, Iterable

    Op1 = Callable[[_T], _T]
    Op2 = Callable[[_T, _T], _T]
    OpN = Callable[[Iterable[_T]], _T]


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
        atom_type_cast = cast("AtomType[_AnyValue]", atom_type)
        func_cast = cast("Callable[[_AnyValue], _T]", func)
        self.atoms[atom_type_cast] = func_cast

    def add_op1(self, head: TreeExpr, func: Op1[_T]) -> None:
        """Add an evaluation rule for a particular head."""
        self.operations[head] = (func, True)

    def add_op2(self, head: TreeExpr, func: Op2[_T]) -> None:
        """Add an evaluation rule for a particular head."""
        self.operations[head] = (func, True)

    def add_opn(self, head: TreeExpr, func: OpN[_T]) -> None:
        """Add an evaluation rule for a particular head."""
        self.operations[head] = (func, False)

    def eval_atom(self, atom: TreeAtom[_S]) -> _T:
        """Evaluate an atom."""
        atom_value = atom.value
        atom_func = self.atoms.get(atom_value.atom_type)  # type: ignore
        if atom_func is not None:
            return atom_func(atom_value.value)
        else:
            msg = "No rule for AtomType: " + atom_value.atom_type.name
            raise NoEvaluationRuleError(msg)

    def eval_operation(self, head: TreeExpr, argvals: Iterable[_T]) -> _T:
        """Evaluate one function with some values."""
        op_func, star_args = self.operations[head]
        if star_args:
            result = op_func(*argvals)
        else:
            result = op_func(argvals)
        return result

    def evaluate(self, expr: TreeExpr, values: dict[TreeExpr, _T]) -> _T:
        """Evaluate the expression using the registered rules."""
        return self.eval_forward(expr, values)

    def eval_recursive(self, expr: TreeExpr, values: dict[TreeExpr, _T]) -> _T:
        """Evaluate the expression using recursion."""
        if expr in values:
            # Use an explicit value if given
            return values[expr]
        elif isinstance(expr, TreeAtom):
            # Convert an Atom to _T
            return self.eval_atom(expr)
        else:
            # Recursively evaluate children and then apply this operation.
            head = expr.children[0]
            children = expr.children[1:]
            argvals = [self.eval_recursive(c, values) for c in children]
            return self.eval_operation(head, argvals)

    def eval_forward(self, expr: TreeExpr, values: dict[TreeExpr, _T]) -> _T:
        """Evaluate the expression using forward evaluation."""
        # Convert expr to the forward graph representation
        graph = forward_graph(expr)
        stack = []

        # Convert all atoms to _T
        for atom in graph.atoms:
            value_get = values.get(atom)
            if value_get is not None:
                value = value_get
            else:
                value = self.eval_atom(atom)  # type: ignore
            stack.append(value)

        # Run forward evaluation through the operations
        for head, indices in graph.operations:
            argvals = [stack[i] for i in indices]
            stack.append(self.eval_operation(head, argvals))

        # Now stack is the values of the topological sort of expr and stack[-1]
        # is the value of expr.
        return stack[-1]

    def __call__(
        self, expr: TreeExpr, values: Optional[dict[TreeExpr, _T]] = None
    ) -> _T:
        """Short-hand for evaluate."""
        if values is None:
            values = {}
        return self.evaluate(expr, values)


class Transformer(Evaluator[TreeExpr]):
    """Specialized Evaluator for TreeExpr -> TreeExpr operations.

    Whereas :class:`Evaluator` is used to evaluate an expression into a
    different type of object like ``float`` or ``str`` a :class:`Transformer`
    is used to transform a :class:`TreeExpr` into a new :class:`TreeExpr`.

    The difference between using ``Transformer`` and using
    ``Evaluator[TreeExpr]`` is that ``Transformer`` allows processing
    operations that have no associated rules leaving the expression unmodified.

    Examples
    ========

    We first import the pieces and define some functions and symbols.

    >>> from protosym.core.tree import funcs_symbols
    >>> from protosym.core.evaluate import Evaluator, Transformer
    >>> [f, g], [x, y] = funcs_symbols(['f', 'g'], ['x', 'y'])

    Now make a :class:`Transformer` to replace ``f(...)`` with ``g(...)``.

    >>> f2g = Transformer()
    >>> f2g.add_opn(f, lambda args: g(*args))
    >>> expr = f(g(x, f(y)), y)
    >>> print(expr)
    f(g(x, f(y)), y)
    >>> print(f2g(expr))
    g(g(x, g(y)), y)

    By contrast with ``Evaluator[TreeExpr]`` the above would fail because no
    rule has been defined for the head ``g`` or for ``Symbol`` (the
    :class:`AtomType` of ``x`` and ``y``).

    >>> f2g_eval = Evaluator[TreeExpr]()
    >>> f2g_eval.add_opn(f, lambda args: g(*args))
    >>> f2g_eval(expr)
    Traceback (most recent call last):
        ...
    protosym.core.exceptions.NoEvaluationRuleError: No rule for AtomType: Symbol
    """

    def eval_atom(self, atom: TreeAtom[_S]) -> TreeExpr:
        """Return the atom as is."""
        return atom

    def eval_operation(self, head: TreeExpr, argvals: Iterable[TreeExpr]) -> TreeExpr:
        """Return unevaluated operation if no rule supplied."""
        if head not in self.operations:
            return head(*argvals)
        else:
            return super().eval_operation(head, argvals)
