"""Define the core evaluation code."""
from __future__ import annotations

from typing import Callable
from typing import cast
from typing import Generic
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import TypeVar

from protosym.core.exceptions import NoEvaluationRuleError
from protosym.core.tree import forward_graph
from protosym.core.tree import Tree


if _TYPE_CHECKING:
    from protosym.core.atom import AnyValue as _AnyValue
    from protosym.core.atom import AtomType


__all__ = ["Evaluator"]


_T = TypeVar("_T")
_S = TypeVar("_S")


if _TYPE_CHECKING:
    from typing import Optional, Sequence, Any

    Op1 = Callable[[_T], _T]
    Op2 = Callable[[_T, _T], _T]
    OpN = Callable[[Sequence[_T]], _T]


def _generic_operation_error(head: Tree, argvals: Sequence[_T]) -> _T:
    """Error fallback rule for handling unknown heads."""
    msg = "No rule for head: " + repr(head)
    raise NoEvaluationRuleError(msg)


def _generic_atom_error(value: Tree) -> Any:
    """Error fallback rule for handling unknown atoms."""
    msg = "No rule for atom: " + repr(value)
    raise NoEvaluationRuleError(msg)


class Evaluator(Generic[_T]):
    """Objects that evaluate expressions.

    Examples
    ========

    First define some symbols and functions:

    >>> import math
    >>> from protosym.core.atom import AtomType
    >>> from protosym.core.tree import Tr
    >>> from protosym.core.evaluate import Evaluator
    >>> Integer = AtomType('Integer', int)
    >>> Symbol = AtomType('Symbol', str)
    >>> Function = AtomType('Function', str)
    >>> sin = Tr(Function('sin'))
    >>> cos = Tr(Function('cos'))
    >>> x = Tr(Symbol('x'))
    >>> one = Tr(Integer(1))

    Now we can make an :class:`Evaluator` to evaluate this kind of expression:

    >>> evalf = Evaluator[float]()
    >>> evalf.add_atom(Integer, float)
    >>> evalf.add_op1(sin, math.sin)
    >>> evalf.add_op1(cos, math.cos)

    We can now use this to evaluate an expression:

    >>> expr1 = cos(one)
    >>> print(expr1)
    cos(1)
    >>> evalf(expr1)
    0.5403023058681398

    We can also supply values for any atoms e.g. symbols when evaluating:

    >>> expr2 = cos(x)
    >>> print(expr2)
    cos(x)
    >>> evalf(expr2, {x: 1.0})
    0.5403023058681398
    """

    atoms: dict[AtomType[_AnyValue], Callable[[_AnyValue], _T]]
    operations: dict[Tree, tuple[Callable[..., _T], bool]]
    generic_operation_func: Callable[[Tree, Sequence[_T]], _T]
    generic_atom_func: Callable[[Tree], _T]

    def __init__(self) -> None:
        """Create an empty evaluator."""
        self.atoms = {}
        self.operations = {}
        self.generic_operation_func = _generic_operation_error
        self.generic_atom_func = _generic_atom_error

    def add_atom(self, atom_type: AtomType[_S], func: Callable[[_S], _T]) -> None:
        """Add an evaluation rule for a particular AtomType."""
        atom_type_cast = cast("AtomType[_AnyValue]", atom_type)
        func_cast = cast("Callable[[_AnyValue], _T]", func)
        self.atoms[atom_type_cast] = func_cast

    def add_atom_generic(self, func: Callable[[Any], _T]) -> None:
        """Add a generic fallback rule for atoms."""
        self.generic_atom_func = func

    def add_op1(self, head: Tree, func: Op1[_T]) -> None:
        """Add an evaluation rule for a particular head."""
        self.operations[head] = (func, True)

    def add_op2(self, head: Tree, func: Op2[_T]) -> None:
        """Add an evaluation rule for a particular head."""
        self.operations[head] = (func, True)

    def add_opn(self, head: Tree, func: OpN[_T]) -> None:
        """Add an evaluation rule for a particular head."""
        self.operations[head] = (func, False)

    def add_op_generic(self, func: Callable[[Tree, Sequence[_T]], _T]) -> None:
        """Add a generic fallback rule for heads."""
        self.generic_operation_func = func

    def eval_atom(self, atom: Tree) -> _T:
        """Evaluate an atom."""
        atom_value = atom.value
        atom_func = self.atoms.get(atom_value.atom_type)
        if atom_func is None:
            return self.generic_atom_func(atom)
        return atom_func(atom_value.value)

    def eval_operation(self, head: Tree, argvals: Sequence[_T]) -> _T:
        """Evaluate one function with some values."""
        func_star = self.operations.get(head)

        if func_star is None:
            return self.generic_operation_func(head, argvals)

        op_func, star_args = func_star

        if star_args:
            result = op_func(*argvals)
        else:
            result = op_func(argvals)

        return result

    def evaluate(self, expr: Tree, values: dict[Tree, _T]) -> _T:
        """Evaluate the expression using the registered rules."""
        return self.eval_forward(expr, values)

    def eval_recursive(self, expr: Tree, values: dict[Tree, _T]) -> _T:
        """Evaluate the expression using recursion."""
        if expr in values:
            # Use an explicit value if given
            return values[expr]
        elif not expr.children:
            # Convert an Atom to _T
            return self.eval_atom(expr)
        else:
            # Recursively evaluate children and then apply this operation.
            head = expr.children[0]
            children = expr.children[1:]
            argvals = [self.eval_recursive(c, values) for c in children]
            return self.eval_operation(head, argvals)

    def eval_forward(self, expr: Tree, values: dict[Tree, _T]) -> _T:
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
                value = self.eval_atom(atom)
            stack.append(value)

        # Run forward evaluation through the operations
        for head, indices in graph.operations:
            argvals = [stack[i] for i in indices]
            stack.append(self.eval_operation(head, argvals))

        # Now stack is the values of the topological sort of expr and stack[-1]
        # is the value of expr.
        return stack[-1]

    def __call__(self, expr: Tree, values: Optional[dict[Tree, _T]] = None) -> _T:
        """Short-hand for evaluate."""
        if values is None:
            values = {}
        return self.evaluate(expr, values)


class Transformer(Evaluator[Tree]):
    """Specialized Evaluator for Tree -> Tree operations.

    Whereas :class:`Evaluator` is used to evaluate an expression into a
    different type of object like ``float`` or ``str`` a :class:`Transformer`
    is used to transform a :class:`Tree` into a new :class:`Tree`.

    The difference between using ``Transformer`` and using
    ``Evaluator[Tree]`` is that ``Transformer`` allows processing
    operations that have no associated rules leaving the expression unmodified.

    Examples
    ========

    We first import the pieces and define some functions and symbols.

    >>> from protosym.core.tree import Tree, funcs_symbols
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

    By contrast with ``Evaluator[Tree]`` the above would fail because no
    rule has been defined for the head ``g`` or for ``Symbol`` (the
    :class:`AtomType` of ``x`` and ``y``).

    >>> expr = f(g(x, f(y)), y)
    >>> f2g_eval = Evaluator[Tree]()
    >>> f2g_eval.add_opn(f, lambda args: g(*args))
    >>> f2g_eval(expr)  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
        ...
    protosym.core.exceptions.NoEvaluationRuleError: No rule for atom: Tr(Symbol('x'))

    We can add a fallback rule for symbols. Then it fails because it needs a
    rule for ``g``:

    >>> f2g_eval.add_atom_generic(lambda x: x)
    >>> f2g_eval(expr)  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
        ...
    protosym.core.exceptions.NoEvaluationRuleError: No rule for head: Tr(Function('g'))

    If we also add a rule for ``g`` then it should work:

    >>> f2g_eval.add_op_generic(lambda head, args: head(*args))
    >>> print(f2g_eval(expr))
    g(g(x, g(y)), y)

    At this point ``f2g_eval`` is equivalent to ``f2g``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_atom_generic(lambda atom: atom)
        self.add_op_generic(lambda head, args: head(*args))
