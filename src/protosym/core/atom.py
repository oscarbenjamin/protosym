"""protosym.core.atom module.

This module defines the :class:`AtomType` and :class:`Atom` types.
"""
from __future__ import annotations

from typing import cast
from typing import Generic as _Generic
from typing import Hashable as _Hashable
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import TypeVar as _TypeVar
from weakref import WeakValueDictionary as _WeakDict


__all__ = [
    "Atom",
    "AtomType",
]


_T = _TypeVar("_T", bound=_Hashable)

AnyValue = _Hashable


if _TYPE_CHECKING:
    from typing import Callable, Any

    KeyType = tuple["AtomType[AnyValue]", AnyValue]
    AtomStoreType = _WeakDict[KeyType, "Atom[AnyValue]"]


#
# The global store of Atoms. This maps from AtomTypes and values to Atoms.
# Whenever an Atom is constructed this store is used to ensure that there is
# only ever a unique object representing a given Atom by returning a
# preexisting object if there is one.
#
_all_atoms = _WeakDict()  # type: ignore


class AtomType(_Generic[_T]):
    """Identifier to distinguish different kinds of atoms.

    :ivar name: Name of this :class:`AtomType`.
    :ivar converter: Converter function for :attr:`Atom.value`.

    An :class:`AtomType` is not an Expression but is used to construct atomic
    expressions.

    >>> from protosym.core.atom import AtomType
    >>> Integer = AtomType('Integer', int)
    >>> Integer
    Integer
    >>> Integer(1)
    Integer(1)
    >>> Symbol = AtomType('Symbol', str)
    >>> Symbol
    Symbol
    >>> Symbol('x')
    Symbol('x')

    The second parameter to :class:`AtomType` is a converter that will convert
    arguments to the expected internal type to be used for the internal
    ``value`` of any associated :class:`Atom`.

    See Also
    ========

    Atom: The actual instances of atomic expressions.
    protosym.core.tree.TreeAtom: The :class:`TreeExpr` representation of atomic
        expressions.
    """

    __slots__ = (
        "name",
        "converter",
    )

    name: str
    converter: Callable[[Any], Any]

    def __init__(self, name: str, converter: Callable[[Any], Any]):
        """New type of Atom e.g. Integer or Symbol.

        Args:
            name (str): The name of this kind of Atom e.g. ``"Integer"``.
            converter (function): Function for converting Python objects to the
                internal value representation for this type of Atom e.g. `int`.
        """
        self.name = name
        self.converter = converter

    def __repr__(self) -> str:
        """Name of the Atom as a string."""
        return self.name

    def __call__(self, value: _T) -> Atom[_T]:
        """Create an Atom of this type."""
        return Atom(self, value)


class Atom(_Generic[_T]):
    """Low level representation of atomic expressions.

    :ivar atom_type: The associated :class:`AtomType` for this :class:`Atom`.
    :ivar value: The value object associated with this :class:`Atom`.

    At the root of any expression tree or graph are the elementary atomic
    expressions. These are expressions that do not have any ``children``. They
    do hold an internal value but it is not considered to be a child in the
    sense of the ``children`` that other nodes of the expression graph have.

    Every :class:`Atom` has both an :class:`AtomType` and an internal ``value``. An
    :class:`Atom` is not intended to be constructed directly but rather is created
    by calling an :class:`AtomType`.

    >>> from protosym.core.atom import AtomType, Atom
    >>> Integer = AtomType('Integer', int)
    >>> one = Integer(1)
    >>> one
    Integer(1)
    >>> print(one)
    1
    >>> type(one) is Atom
    True
    >>> one.atom_type
    Integer
    >>> one.value
    1
    >>> type(one)
    <class 'protosym.core.atom.Atom'>
    >>> type(one.value)
    <class 'int'>

    We can rebuild the :class:`Atom` from its ``atom_type`` and ``value``. This is
    useful if we want to make an atom with a modified ``value``. Giving exactly
    the same ``value`` will return precisely the same object because there can
    only be a unique copy of an atom with any given ``atom_type`` and
    ``value``. A global store is used to ensure that creating a new :class:`Atom` of
    the same :class:`AtomType` and ``value`` will always return the same object.
    For this to work the value used to construct an :class:`Atom` is required to be
    hashable.

    >>> one == one.atom_type(one.value)
    True
    >>> one is one.atom_type(one.value)
    True
    >>> Integer(2) is Integer(2)
    True

    Apart from holding a reference to the :class:`AtomType` and also the internal
    value the only property that an :class:`Atom` has is that it can be compared to
    other objects with `==` and is itself hashable.

    See Also
    ========

    AtomType: The class of types of :class:`Atom`.
    protosym.core.tree.TreeAtom: The higher-level :class:`TreeExpr`
        representation of an :class:`Atom`.
    """

    __slots__ = (
        "__weakref__",
        "atom_type",
        "value",
    )

    atom_type: AtomType[_T]
    value: _T

    def __new__(cls, atom_type: AtomType[_T], value: _T) -> Atom[_T]:
        """New Atom or an existing Atom from the global store."""
        key = (atom_type, value)

        previous = _all_atoms.get(key, None)
        if previous is not None:
            return cast(Atom[_T], previous)

        obj = object.__new__(cls)
        obj.atom_type = atom_type
        obj.value = value

        # Use setdefault to avoid race conditions in a multithreaded context.
        # If another thread created this atom in between the lookup above and
        # here then we will make sure to retrieve the object created there or
        # otherwise force the other thread to accept the value we have created
        # here.
        obj = cast(Atom[_T], _all_atoms.setdefault(key, obj))

        return obj

    def __repr__(self) -> str:
        """Explicit representation as e.g. ``'Integer(1)'``."""
        return f"{self.atom_type}({self.value!r})"

    def __str__(self) -> str:
        """Pretty representation as e.g. ``'1'``."""
        return str(self.value)


if _TYPE_CHECKING:
    AnyAtom = Atom[_Hashable]
