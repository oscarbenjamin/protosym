"""The :class:`Sym` class.

This module defines the :class:`Sym` class which should be used as the
superclass for all user-facing symbolic classes. The :class:`Sym` class only
has a handful of attributes and methods that are used to make it compatible
with the rest of the machinery defined in `protosym.core`. The idea here is to
build a nicer syntax over the lower-level classes that can be inherited for use
by user-facing classes that derive from :class:`Sym`.
"""
from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Generic
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Type
from typing import TypeVar
from weakref import WeakValueDictionary as _WeakDict

from protosym.core.atom import AtomType
from protosym.core.differentiate import diff_forward
from protosym.core.differentiate import DiffProperties
from protosym.core.evaluate import Evaluator
from protosym.core.exceptions import BadRuleError
from protosym.core.tree import SubsFunc
from protosym.core.tree import Tr
from protosym.core.tree import Tree


__all__ = ["Sym", "SymAtomType", "SymEvaluator"]


T_sym = TypeVar("T_sym", bound="Sym")
T_val = TypeVar("T_val")
S_val = TypeVar("S_val")
T_op = TypeVar("T_op")


Wild = AtomType("Wild", str)


class SymAtomType(Generic[T_sym, T_val]):
    """Wrapper around AtomType to construct atoms as Sym."""

    name: str
    sym: Type[T_sym]
    atom_type: AtomType[T_val]

    def __init__(self, name: str, sym: Type[T_sym], typ: Type[T_val]) -> None:
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
        return self.sym(Tr(atom))

    def __getitem__(self, wild: T_sym) -> SymAtomValue[T_sym, T_val]:
        """Represent extracting the value from self."""
        return SymAtomValue(self, wild)


class SymAtomValue(Generic[T_sym, T_val]):
    """Representation of extracting a value from an Atom."""

    def __init__(self, atom_type: SymAtomType[T_sym, T_val], wild: T_sym) -> None:
        self.atom_type = atom_type
        self.args = (wild,)


class Sym:
    """Base class for user-facing symbolic classes.

    This class should not be used directly but rather subclassed to make a
    user-facing symbolic expression type:

    >>> from protosym.core.sym import Sym

    The :class:`Sym` class should be subclassed to add any desired methods such
    as `__add__`. The :class:`Sym` class defines object construction to ensure
    that all instances of :class:`Sym` for any given subclass and underlying
    :class:`Tree` are unique. Each :class:`Sym` instance holds an internal
    `rep` attribute to which all methods are delegated.

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
    :class:`AtomType` and :class:`Tree`:

    >>> Integer = Expr.new_atom('Integer', int)
    >>> Integer
    Integer
    >>> one = Integer(1)
    >>> one
    Sym(Tr(Integer(1)))
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
    Sym(Tr(Function('Add')))
    >>> expr
    Sym(Tree(Tr(Function('Add')), Tr(Integer(1)), Tr(Integer(1))))

    Subclasses should probably implement prettier printing.

    See Also
    ========

    protosym.simplecas::Expr - A subclass of :class:`Sym`.
    """

    _all_expressions: _WeakDict[Any, Any] = _WeakDict()

    rep: Tree

    def __new__(cls, tree_expr: Tree) -> Sym:
        """Create a new Sym wrapping `tree_expr`.

        If an equivalent Sym instance already exists then the same object will
        be returned.
        """
        if not isinstance(tree_expr, Tree):
            raise TypeError("First argument to Sym should be Tree")

        key = (cls, tree_expr)

        expr = cls._all_expressions.get(key, None)
        if expr is not None:
            return expr  # type:ignore

        obj = super().__new__(cls)
        obj.rep = tree_expr

        obj = cls._all_expressions.setdefault(key, obj)

        return obj

    @property
    def head(self: T_sym) -> T_sym:
        """Head of the expression as a Sym."""
        return type(self)(self.rep.children[0])

    @property
    def args(self: T_sym) -> tuple[T_sym, ...]:
        """Args of the expression as a tuple."""
        cls = type(self)
        return tuple(cls(child) for child in self.rep.children[1:])

    def __repr__(self) -> str:
        """Raw text representation of a Sym.

        >>> from protosym.core.sym import Sym
        >>> Integer = Sym.new_atom('Integer', int)
        >>> Integer(1)
        Sym(Tr(Integer(1)))
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
        Sym(Tr(Integer(1)))
        """
        return SymAtomType(name, cls, typ)

    @classmethod
    def new_wild(cls: Type[T_sym], name: str) -> T_sym:
        """Create a new wild for a given :class:`Sym` subclass.

        >>> from protosym.core.sym import Sym
        >>> a = Sym.new_wild('a')
        >>> print(a)
        a
        >>> a
        Sym(Tr(Wild('a')))
        """
        return cls(Tr(Wild(name)))

    @classmethod
    def new_evaluator(
        cls: Type[T_sym], name: str, typ: Type[T_val]
    ) -> SymEvaluator[T_sym, T_val]:
        """Create a :class:`SymEvaluator` for a :class:`Sym` subclass.

        >>> from protosym.core.sym import Sym, PyFunc1
        >>> a = Sym.new_wild('a')
        >>> Integer = Sym.new_atom('Integer', int)
        >>> one = Integer(1)
        >>> int_to_float = PyFunc1(float)
        >>> eval_f64 = Sym.new_evaluator('eval_f64', float)
        >>> eval_f64
        eval_f64
        >>> eval_f64[Integer[a]] = int_to_float(a)
        >>> eval_f64(one)
        1.0

        See Also
        ========

        SymEvaluator
        """
        return SymEvaluator(name)


class PyFunc:
    """Base class for wrapping callable Python functions.

    This should not be used directly but rather its subclasses that have
    specific signatures should be used. The purpose of these different classes
    is really just so that a type checker can understand what type of function
    is expected by an :class:`Evaluator` and can check the types when a rule is
    added. Putting together an :class:`Evaluator` at runtime inherently
    involves dynamic typing so the checker needs a bit of help to understand
    what is happening.

    See Also
    ========

    PyOp1
    PyOp2
    PyOpN
    HeadOp
    AtomFunc
    PyFunc1
    """

    __slots__ = ()


class WildCall(Generic[T_sym, T_op]):
    __slots__ = ("op", "args")

    def __init__(self, op: T_op, *args: T_sym):
        self.op = op
        self.args = args


class PyOp(Generic[T_val], PyFunc):
    """Base class for PyFuncs T -> T, (T, T) -> T etc.

    See Also
    ========

    PyFunc
    """

    __slots__ = ()


class PyOp1(PyOp[T_val]):
    """Wrapper for an unary func T_val -> T_val.

    See Also
    ========

    PyFunc
    """

    __slots__ = ("func",)

    def __init__(self, func: Callable[[T_val], T_val]):
        self.func = func

    def __call__(self: T_op, arg1: T_sym) -> WildCall[T_sym, T_op]:
        return WildCall(self, arg1)


class PyOp2(PyOp[T_val]):
    """Wrapper for an binary func (T_val, T_val) -> T_val.

    See Also
    ========

    PyFunc
    """

    __slots__ = ("func",)

    def __init__(self, func: Callable[[T_val, T_val], T_val]):
        self.func = func

    def __call__(self: T_op, arg1: T_sym, arg2: T_sym) -> WildCall[T_sym, T_op]:
        return WildCall(self, arg1, arg2)


class PyOpN(PyOp[T_val]):
    """Wrapper for a sequence func [T_val, ...] -> T_val.

    See Also
    ========

    PyFunc
    """

    __slots__ = ("func",)

    def __init__(self, func: Callable[[Sequence[T_val]], T_val]):
        self.func = func

    def __call__(self: T_op, args: T_sym) -> WildCall[T_sym, T_op]:
        return WildCall(self, args)


class PyFunc1(Generic[S_val, T_val], PyFunc):
    """Wrapper for unary func S_val -> T_val.

    See Also
    ========

    PyFunc
    """

    __slots__ = ("func",)

    def __init__(self, func: Callable[[S_val], T_val]):
        self.func = func

    def __call__(self: T_op, arg: T_sym) -> WildCall[T_sym, T_op]:
        return WildCall(self, arg)


class AtomFunc(Generic[T_val], PyFunc):
    """Wrapper for unary func object -> T_val.

    See Also
    ========

    PyFunc
    """

    __slots__ = ("func",)

    def __init__(self, func: Callable[[object], T_val]):
        self.func = func

    def __call__(self: T_op, arg: T_sym) -> WildCall[T_sym, T_op]:
        return WildCall(self, arg)


class HeadOp(Generic[T_val], PyFunc):
    """Wrapper for generic head function.

    See Also
    ========

    PyFunc
    """

    __slots__ = ("func",)

    # XXX: Ideally this function would not operate on Tree because it means
    # that the function wrapped by HeadOp needs to know how about Tree which
    # mixes up the levels we want to work with. Not sure yet what is the best
    # way to handle this. Currently where HeadOp is used the Tree argument is
    # either ignored or just converted to a string.

    def __init__(self, func: Callable[[Tree, Sequence[T_val]], T_val]):
        self.func = func

    def __call__(self: T_op, head: T_sym, args: T_sym) -> WildCall[T_sym, T_op]:
        return WildCall(self, head, args)


class _AtomRuleType:
    """Internal type for the AtomRule marker."""

    __slots__ = ()

    def __getitem__(self, wild: T_sym) -> AtomRuleType[T_sym]:
        return AtomRuleType(wild)


class AtomRuleType(Generic[T_sym]):
    """Generic atom rule representation."""

    __slots__ = ("args",)

    def __init__(self, wild: T_sym):
        self.args = (wild,)


class _HeadRuleType:
    """Internal type for the HeadRule marker."""

    __slots__ = ()

    def __call__(self, head: T_sym, args: T_sym) -> HeadRuleType[T_sym]:
        return HeadRuleType(head, args)


class HeadRuleType(Generic[T_sym]):
    """Generic atom rule representation."""

    __slots__ = ("args",)

    def __init__(self, head: T_sym, args: T_sym):
        self.args = (head, args)


AtomRule = _AtomRuleType()
HeadRule = _HeadRuleType()


Star = Tr(AtomType("Star", str)("Star"))


def star(wild: T_sym) -> T_sym:
    """Represent star-args e.g. ``f(*a)``."""
    return type(wild)(Star(wild.rep))


class SymEvaluator(Generic[T_sym, T_val]):
    """Evaluator for a given :class:`Sym` subclass.

    These should not be created directly but rather using the
    :meth:`Sym.new_evaluator` method. The demonstration here is somewhat
    awkward because :class:`Sym` does not define `__call__` although it is
    expected that most subclasses would.

    First create some atom types:

    >>> import math
    >>> from protosym.core.sym import Sym, PyOp1, PyFunc1
    >>> Integer = Sym.new_atom('Integer', int)
    >>> Function = Sym.new_atom('Function', str)
    >>> cos = Function('cos').rep
    >>> a = Sym.new_wild('a')
    >>> ar = a.rep
    >>> one = Integer(1).rep

    Make some symbolically typed functions for evaluation:

    >>> f64_from_int = PyFunc1(float)
    >>> f64_cos = PyOp1(math.cos)

    Now we can make an :class:`Evaluator` and add rules to it:

    >>> eval_f64 = Sym.new_evaluator('eval_f64', float)
    >>> eval_f64
    eval_f64
    >>> eval_f64[Integer[a]] = f64_from_int(a)
    >>> eval_f64[Sym(cos(ar))] = f64_cos(a)

    Now make an expression and evaluate the expression:

    >>> cos_one = Sym(cos(one))
    >>> print(cos_one)
    cos(1)
    >>> eval_f64(cos_one) # doctest: +ELLIPSIS
    0.5403023058681...

    See Also
    ========

    protosym.core.evaluate::Evaluator
    protosym.simplecas::Expr.new_evaluator
    """

    def __init__(self, name: str):
        """Create a new SymEvaluator."""
        self.name = name
        self.evaluator = Evaluator[T_val]()

    # e.g. eval_f64[cos(a)] = f64_cos(a)
    @overload
    def __setitem__(  # noqa: D105
        self,
        pattern: T_sym,
        call: WildCall[T_sym, PyOp1[T_val]]
        | WildCall[T_sym, PyOp2[T_val]]
        | WildCall[T_sym, PyOpN[T_val]],
    ) -> None:
        ...

    # e.g. eval_f64[Integer[a]] = f64_from_int
    @overload
    def __setitem__(  # noqa: D105
        self,
        pattern: SymAtomValue[T_sym, S_val],
        call: WildCall[T_sym, PyFunc1[S_val, T_val]],
    ) -> None:
        ...

    # e.g. eval_repr[AtomRule[a]] = AtomFunc(repr)
    @overload
    def __setitem__(  # noqa: D105
        self,
        pattern: AtomRuleType[T_sym],
        call: WildCall[T_sym, AtomFunc[T_val]],
    ) -> None:
        ...

    # e.g. eval_repr[HeadRule(a, b)] = HeadOp(...)
    @overload
    def __setitem__(  # noqa: D105
        self,
        pattern: HeadRuleType[T_sym],
        call: WildCall[T_sym, HeadOp[T_val]],
    ) -> None:
        ...

    def __setitem__(  # noqa: C901
        self,
        pattern: T_sym
        | SymAtomValue[T_sym, S_val]
        | AtomRuleType[T_sym]
        | HeadRuleType[T_sym],
        call: WildCall[T_sym, PyOp1[T_val]]
        | WildCall[T_sym, PyOp2[T_val]]
        | WildCall[T_sym, PyOpN[T_val]]
        | WildCall[T_sym, PyFunc1[S_val, T_val]]
        | WildCall[T_sym, AtomFunc[T_val]]
        | WildCall[T_sym, HeadOp[T_val]],
    ) -> None:
        """Add an evaluation rule."""
        if not isinstance(call, WildCall):
            raise BadRuleError("Rule function should be a symbolic call.")

        if isinstance(call.op, PyOpN):
            (callarg,) = call.args
            if pattern.args != (star(callarg),):
                raise BadRuleError("varargs function needs a star-rule.")
        elif pattern.args != call.args:
            raise BadRuleError("Pattern and rule signatures do not match.")

        if isinstance(pattern, AtomRuleType):
            if not isinstance(call.op, AtomFunc):
                raise BadRuleError("AtomRule func should be of type AtomFunc")
            self.evaluator.add_atom_generic(call.op.func)

        elif isinstance(pattern, HeadRuleType):
            if not isinstance(call.op, HeadOp):
                raise BadRuleError("HeadRule func should be of type HeadOp")
            self.evaluator.add_op_generic(call.op.func)

        elif isinstance(pattern, SymAtomValue):
            if not isinstance(call.op, PyFunc1):
                raise BadRuleError("Rule for an Atom should be pf type PyFunc1")
            self.evaluator.add_atom(pattern.atom_type.atom_type, call.op.func)

        else:
            if not isinstance(call.op, PyOp):
                raise BadRuleError("Rule for a head should by of type PyOp")
            self.add_op(pattern.head, call.op)

    def add_op(
        self,
        head: T_sym,
        op: PyOp1[T_val] | PyOp2[T_val] | PyOpN[T_val],
    ) -> None:
        """Add an evaluation rule like sin(a) -> evalf_64(a)."""
        if isinstance(op, PyOp1):
            self.evaluator.add_op1(head.rep, op.func)
        elif isinstance(op, PyOp2):
            self.evaluator.add_op2(head.rep, op.func)
        elif isinstance(op, PyOpN):
            self.evaluator.add_opn(head.rep, op.func)
        else:
            raise BadRuleError("Not a PyFunc")

    def __repr__(self) -> str:
        """Print as the name of the Evaluator."""
        return self.name

    def __call__(
        self, expr: T_sym, values: Optional[dict[T_sym, T_val]] = None
    ) -> T_val:
        """Evaluate a given expression using the rules."""
        values_rep = {}
        if values is not None:
            values_rep = {e.rep: v for e, v in values.items()}
        return self.evaluator(expr.rep, values_rep)


class SymDifferentiator(Generic[T_sym]):
    """Representation of differentiation rules.

    The Differentiator is created and then given rules for differentiation.

    >>> from protosym.simplecas import (
    ...     Function, Symbol, Expr, Add, Mul, zero, one, a, b)
    >>> from protosym.core.sym import SymDifferentiator
    >>> diff = SymDifferentiator(Expr, add=Add, mul=Mul, zero=zero, one=one)
    >>> x = Symbol('x')
    >>> tan = Function('tan')
    >>> diff[tan(a), a] = 1 + tan(a)**2
    >>> diff[a**b, a] = b * a**(b + (-1))
    >>> diff(tan(tan(x)), x)
    ((1 + tan(tan(x))**2)*(1 + tan(x)**2))
    """

    def __init__(
        self,
        new_sym: Type[T_sym],
        *,
        add: T_sym,
        mul: T_sym,
        zero: T_sym,
        one: T_sym,
    ):
        """Create a new :class:`Differentiator`."""
        self.new_sym = new_sym
        self.diff_props = DiffProperties(
            zero=zero.rep, one=one.rep, add=add.rep, mul=mul.rep
        )

    def add_distributive_rule(self, head: T_sym) -> None:
        """Register that differentiation can distribute over ``head``.

        This describes a rule like :math:`f(x, y)' = f(x', y')`.
        """
        self.diff_props.add_distributive(head.rep)

    def __setitem__(self, expr_sym: tuple[T_sym, T_sym], dexpr: T_sym) -> None:
        """Register a function rule like ``diff[sin(a), a] = cos(a)``."""
        if not isinstance(expr_sym, tuple) or len(expr_sym) != 2:
            raise TypeError("Pattern should be an expr-sym pair like diff[cos(a), a]")

        expr, sym = expr_sym

        expr_r = expr.rep
        dexpr_r = dexpr.rep
        sym_r = sym.rep

        head = expr_r.children[0]
        args_lhs = expr_r.children[1:]
        if args_lhs.count(sym_r) != 1:
            raise TypeError("Multiple occurrences of wild symbol in pattern.")
        index = args_lhs.index(sym_r)

        rhsfunc = SubsFunc(dexpr_r, list(args_lhs))
        self.diff_props.add_diff_rule(head, index, rhsfunc)

    def __call__(self, expr: T_sym, sym: T_sym, ntimes: int = 1) -> T_sym:
        """Compute the derivative of ``expr`` wrt ``sym`` ``ntimes``."""
        d_expr = expr.rep
        symrep = sym.rep
        for _ in range(ntimes):
            d_expr = diff_forward(d_expr, symrep, self.diff_props)
        return self.new_sym(d_expr)
