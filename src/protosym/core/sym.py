"""The :class:`Sym` class.

This module defines the :class:`Sym` class which should be used as the
superclass for all user-facing symbolic classes. The :class:`Sym` class only
has a handful of attributes and methods that are used to make it compatible
with the rest of the machinery defined in `protosym.core`.
"""
from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Generic
from typing import Optional
from typing import Sequence
from typing import Type
from typing import TypeVar
from weakref import WeakValueDictionary as _WeakDict

from protosym.core.atom import AtomType
from protosym.core.evaluate import Evaluator
from protosym.core.tree import TreeAtom
from protosym.core.tree import TreeExpr

__all__ = ["Sym", "SymAtomType", "SymEvaluator"]

T_sym = TypeVar("T_sym", bound="Sym")
T_val = TypeVar("T_val")
S_val = TypeVar("S_val")


class Sym:
    """Base class for user-facing symbolic classes.

    This class should not be used directly but rather subclassed to make a
    user-facing symbolic expression type:

    >>> from protosym.core.sym import Sym

    The :class:`Sym` class should be subclassed to add any desired methods such
    as `__add__`. The :class:`Sym` class defines object construction to ensure
    that all instances of :class:`Sym` for any given subclass and underlying
    :class:`TreeExpr` are unique. Each :class:`Sym` instance holds an internal
    `rep` attribute which to which all methods are delegated.

    >>> class Expr(Sym):
    ...     def __call__(self: Expr, *args: Expr) -> Expr:
    ...         args_rep = [arg.rep for arg in args]
    ...         return Expr(self.rep(*args_rep))
    ...     def __add__(self: Expr, other: Expr) -> Expr:
    ...         if not isinstance(other, Expr):
    ...             return NotImplemented
    ...         return Add(self, other)
    ...

    Now we can define some atom types and instances. These are wrappers around
    :class:`AtomType` and :class:`TreeAtom`:

    >>> Integer = Expr.new_atom('Integer', int)
    >>> Integer
    Integer
    >>> one = Integer(1)
    >>> one
    Sym(TreeAtom(Integer(1)))
    >>> print(one)
    1

    We can also define heads and operations to make compound expressions:

    >>> Function = Expr.new_atom('Function', str)
    >>> Add = Function('Add')
    >>> print(Add)
    Add
    >>> expr = Add(one, one)
    >>> print(expr)
    Add(1, 1)

    The `repr` shows the raw representation of these objects:

    >>> Add
    Sym(TreeAtom(Function('Add')))
    >>> expr
    Sym(TreeNode(TreeAtom(Function('Add')), TreeAtom(Integer(1)), TreeAtom(Integer(1))))

    Subclasses should probably implement prettier printing.

    See Also
    ========

    protosym.simplecas::Expr - A subclass of :class:`Sym`.
    """

    _all_expressions: _WeakDict[Any, Any] = _WeakDict()

    rep: TreeExpr

    def __new__(cls, tree_expr: TreeExpr) -> Sym:
        """Create a new Sym wrapping `tree_expr`.

        If an equivalent Sym instance already exists then the same object will
        be returned.
        """
        if not isinstance(tree_expr, TreeExpr):
            raise TypeError("First argument to Sym should be TreeExpr")

        key = (cls, tree_expr)

        expr = cls._all_expressions.get(key, None)
        if expr is not None:
            return expr  # type:ignore

        obj = super().__new__(cls)
        obj.rep = tree_expr

        obj = cls._all_expressions.setdefault(key, obj)

        return obj

    def __repr__(self) -> str:
        """Raw text representation of a Sym.

        >>> from protosym.core.sym import Sym
        >>> Integer = Sym.new_atom('Integer', int)
        >>> Integer(1)
        Sym(TreeAtom(Integer(1)))
        """
        return f"Sym({self.rep!r})"

    def __str__(self) -> str:
        """Prettier text representation of a Sym.

        >>> from protosym.core.sym import Sym
        >>> Integer = Sym.new_atom('Integer', int)
        >>> print(Integer(1))
        1
        """
        return str(self.rep)

    @classmethod
    def new_atom(
        cls: Type[T_sym], name: str, typ: Type[T_val]
    ) -> SymAtomType[T_sym, T_val]:
        """Create a new atom type for a given :class:`Sym` subclass.

        >>> from protosym.core.sym import Sym
        >>> Integer = Sym.new_atom('Integer', int)
        >>> one = Integer(1)
        >>> one
        Sym(TreeAtom(Integer(1)))
        """
        return SymAtomType(name, cls, typ)

    @classmethod
    def new_evaluator(
        cls: Type[T_sym], name: str, typ: Type[T_val]
    ) -> SymEvaluator[T_sym, T_val]:
        """Create a :class:`SymEvaluator` for a :class:`Sym` subclass.

        >>> from protosym.core.sym import Sym
        >>> Integer = Sym.new_atom('Integer', int)
        >>> one = Integer(1)
        >>> eval_f64 = Sym.new_evaluator('eval_f64', float)
        >>> eval_f64
        eval_f64
        >>> eval_f64.add_atom(Integer, float)
        >>> eval_f64(one)
        1.0

        See Also
        ========

        SymEvaluator
        """
        return SymEvaluator(name)


class SymAtomType(Generic[T_sym, T_val]):
    """Wrapper around AtomType to construct atoms as Sym."""

    name: str
    sym: Callable[[TreeAtom[T_val]], T_sym]
    atom_type: AtomType[T_val]

    def __init__(
        self, name: str, sym: Callable[[TreeAtom[T_val]], T_sym], typ: Type[T_val]
    ) -> None:
        """New SymAtomType."""
        self.name = name
        self.sym = sym
        self.atom_type = AtomType(name, typ)

    def __repr__(self) -> str:
        """The name of the SymAtomType."""
        return self.name

    def __call__(self, value: T_val) -> T_sym:
        """Create a new Atom as a Sym."""
        atom = self.atom_type(value)
        return self.sym(TreeAtom[T_val](atom))


class SymEvaluator(Generic[T_sym, T_val]):
    """Evaluator for a given :class:`Sym` subclass.

    These should not be created directly but rather using the
    :meth:`Sym.new_evaluator` method. First create some atom types:

    >>> import math
    >>> from protosym.core.sym import Sym
    >>> Integer = Sym.new_atom('Integer', int)
    >>> Function = Sym.new_atom('Function', int)
    >>> cos = Function('cos')

    Now we can make an :class:`Evaluator` and add rules to it:

    >>> eval_f64 = Sym.new_evaluator('eval_f64', float)
    >>> eval_f64
    eval_f64
    >>> eval_f64.add_atom(Integer, float)
    >>> eval_f64.add_op1(cos, math.cos)

    Now make an expression and evaluate the expression:

    >>> one = Integer(1)
    >>> cos_one = Sym(cos.rep(one.rep))
    >>> print(cos_one)
    cos(1)
    >>> eval_f64(cos_one) # doctest: +ELLIPSIS
    0.5403023058681...

    See Also
    ========

    protosym.core.evaluate::Evaluator
    """

    def __init__(self, name: str):
        """Create a new SymEvaluator."""
        self.name = name
        self.evaluator = Evaluator[T_val]()

    def __repr__(self) -> str:
        """Print as the name of the Evaluator."""
        return self.name

    def add_atom(
        self, atom_type: SymAtomType[T_sym, S_val], func: Callable[[S_val], T_val]
    ) -> None:
        """Add a rule for an atom type."""
        self.evaluator.add_atom(atom_type.atom_type, func)

    def add_atom_generic(self, func: Callable[[Any], T_val]) -> None:
        """Add a generic fallback rule for atoms."""
        self.evaluator.add_atom_generic(func)

    def add_op1(self, head: T_sym, func: Callable[[T_val], T_val]) -> None:
        """Add a rule for an unary head."""
        self.evaluator.add_op1(head.rep, func)

    def add_op2(self, head: T_sym, func: Callable[[T_val, T_val], T_val]) -> None:
        """Add a rule for a binary head."""
        self.evaluator.add_op2(head.rep, func)

    def add_opn(self, head: T_sym, func: Callable[[Sequence[T_val]], T_val]) -> None:
        """Add a rule for an nary head."""
        self.evaluator.add_opn(head.rep, func)

    def add_op_generic(
        self, func: Callable[[TreeExpr, Sequence[T_val]], T_val]
    ) -> None:
        """Add a generic fallback rule for heads."""
        self.evaluator.add_op_generic(func)

    def __call__(
        self, expr: T_sym, values: Optional[dict[T_sym, T_val]] = None
    ) -> T_val:
        """Evaluate a given expression using the rules."""
        values_rep = {}
        if values is not None:
            values_rep = {e.rep: v for e, v in values.items()}
        return self.evaluator(expr.rep, values_rep)
