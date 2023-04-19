"""Simple Matrix class."""
from __future__ import annotations

from typing import Any
from typing import Sequence
from typing import TYPE_CHECKING as _TYPE_CHECKING

from protosym.simplecas.expr import Add
from protosym.simplecas.expr import Expr
from protosym.simplecas.expr import expressify
from protosym.simplecas.expr import List
from protosym.simplecas.expr import zero


if _TYPE_CHECKING:
    from protosym.simplecas.expr import Expressifiable


class Matrix:
    """Matrix of Expr."""

    nrows: int
    ncols: int
    shape: tuple[int, int]
    elements: list[Expr]
    elements_graph: Expr
    entrymap: dict[tuple[int, int], int]

    def __new__(cls, entries: Sequence[Sequence[Expressifiable]]) -> Matrix:
        """New Matrix from a list of lists."""
        if not isinstance(entries, list) or not all(
            isinstance(row, list) for row in entries
        ):
            raise TypeError("Input should be a list of lists.")

        nrows = len(entries)
        ncols = len(entries[0])
        if not all(len(row) == ncols for row in entries):
            raise TypeError("All rows should be the same length.")

        entries_expr = [[expressify(e) for e in row] for row in entries]

        elements: list[Expr] = []
        entrymap = {}
        for i, row in enumerate(entries_expr):
            for j, entry in enumerate(row):
                if entry != zero:
                    entrymap[(i, j)] = len(elements)
                    elements.append(entry)

        return cls._new(nrows, ncols, elements, entrymap)

    @classmethod
    def _new(
        cls,
        nrows: int,
        ncols: int,
        elements: list[Expr],
        entrymap: dict[tuple[int, int], int],
    ) -> Matrix:
        """New matrix from the internal representation."""
        obj = super().__new__(cls)
        obj.nrows = nrows
        obj.ncols = ncols
        obj.shape = (nrows, ncols)
        obj.elements = list(elements)
        obj.elements_graph = List(*elements)
        obj.entrymap = entrymap
        return obj

    def __getitem__(self, ij: tuple[int, int]) -> Expr:
        """Element indexing ``M[i, j]``."""
        if isinstance(ij, tuple) and len(ij) == 2:
            i, j = ij
            if isinstance(i, int) and isinstance(j, int):
                if not (0 <= i < self.nrows and 0 <= j < self.ncols):
                    raise IndexError("Indices out of bounds.")
                if ij in self.entrymap:
                    return self.elements[self.entrymap[ij]]
                else:
                    return zero
        raise TypeError("Matrix indices should be a pair of integers.")

    def tolist(self) -> list[list[Expr]]:
        """Convert to list of lists format."""
        entries = [[zero] * self.ncols for _ in range(self.nrows)]
        for (i, j), n in self.entrymap.items():
            entries[i][j] = self.elements[n]
        return entries

    def to_sympy(self) -> Any:
        """Convert a simplecas Matrix to a SymPy Matrix."""
        from protosym.simplecas.sympy_conversions import to_sympy_matrix

        return to_sympy_matrix(self)

    @classmethod
    def from_sympy(cls, mat: Any) -> Matrix:
        """Convert a SymPy Matrix to a simplecas Matrix."""
        from protosym.simplecas.sympy_conversions import from_sympy_matrix

        return from_sympy_matrix(mat)

    def __repr__(self) -> str:
        """Convert to pretty representation."""
        # Inefficient because does not use a graph...
        # (This computes separate repr for each element)
        return f"Matrix({self.tolist()!r})"

    def __add__(self, other: Matrix) -> Matrix:
        """Matrix addition A + B -> C."""
        if not isinstance(other, Matrix):
            return NotImplemented
        return self.binop(other, Add)

    def binop(self, other: Matrix, func: Expr) -> Matrix:
        """Elementwise binary operaton on two matrices."""
        if self.shape != other.shape:
            raise TypeError("Shape mismatch.")
        new_elements = self.elements.copy()
        new_entrymap = self.entrymap.copy()
        for ij, n_other in other.entrymap.items():
            if ij in new_entrymap:
                self_ij = new_elements[new_entrymap[ij]]
                other_ij = other.elements[n_other]
                result = func(self_ij, other_ij)
                new_elements[new_entrymap[ij]] = result
            else:
                new_entrymap[ij] = len(new_elements)
                new_elements.append(other.elements[n_other])
        return self._new(self.nrows, self.ncols, new_elements, new_entrymap)

    def diff(self, sym: Expr) -> Matrix:
        """Differentiate Matrix wrt ``sym``."""
        if not isinstance(sym, Expr):
            raise TypeError("Differentiation var should be a symbol.")
        # Use the element_graph rather than differentiating each element
        # separately.
        elements_diff = self.elements_graph.diff(sym)
        new_elements: list[Expr] = list(elements_diff.args)
        return self._new(self.nrows, self.ncols, new_elements, self.entrymap)

    def to_llvm_ir(self, symargs: list[Expr]) -> str:
        """Return LLVM IR code evaluating this Matrix.

        >>> from protosym.simplecas import sin, cos, x, Matrix
        >>> M = Matrix([[cos(x), sin(x)], [-sin(x), cos(x)]])
        >>> print(M.to_llvm_ir([x]))
        ; ModuleID = "mod1"
        target triple = "unknown-unknown-unknown"
        target datalayout = ""
        <BLANKLINE>
        declare double    @llvm.pow.f64(double %Val1, double %Val2)
        declare double    @llvm.sin.f64(double %Val)
        declare double    @llvm.cos.f64(double %Val)
        <BLANKLINE>
        define void @"jit_func1"(double* %"_out", double %"x")
        {
        %".0" = call double @llvm.cos.f64(double %"x")
        %".1" = call double @llvm.sin.f64(double %"x")
        %".2" = fmul double 0xbff0000000000000, %".1"
        %".3" = getelementptr double, double* %"_out", i32 0
        store double %".0", double* %".3"
        %".4" = getelementptr double, double* %"_out", i32 1
        store double %".1", double* %".4"
        %".5" = getelementptr double, double* %"_out", i32 2
        store double %".2", double* %".5"
        %".6" = getelementptr double, double* %"_out", i32 3
        store double %".0", double* %".6"
        ret void
        }

        """
        from protosym.simplecas.lambdification import _to_llvm_f64_matrix

        return _to_llvm_f64_matrix([arg.rep for arg in symargs], self)
