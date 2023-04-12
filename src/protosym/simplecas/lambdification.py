"""lambdification with LLVM."""
from __future__ import annotations

import ctypes
import struct
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING as _TYPE_CHECKING

from protosym.core.tree import forward_graph
from protosym.simplecas.exceptions import LLVMNotImplementedError
from protosym.simplecas.expr import Add
from protosym.simplecas.expr import bin_expand
from protosym.simplecas.expr import cos
from protosym.simplecas.expr import Expr
from protosym.simplecas.expr import Integer
from protosym.simplecas.expr import Mul
from protosym.simplecas.expr import Pow
from protosym.simplecas.expr import sin
from protosym.simplecas.matrix import Matrix


if _TYPE_CHECKING:
    from protosym.core.tree import Tree


def _double_to_hex(f: float) -> str:
    return hex(struct.unpack("<Q", struct.pack("<d", f))[0])


_llvm_header = """
; ModuleID = "mod1"
target triple = "unknown-unknown-unknown"
target datalayout = ""

declare double    @llvm.pow.f64(double %Val1, double %Val2)
declare double    @llvm.sin.f64(double %Val)
declare double    @llvm.cos.f64(double %Val)

"""


def _to_llvm_f64(symargs: list[Tree], expression: Tree) -> str:
    """Code for LLVM IR function computing ``expression`` from ``symargs``."""
    expression = bin_expand(expression)

    graph = forward_graph(expression)

    argnames = {s: f'%"{s}"' for s in symargs}  # noqa

    identifiers = []
    for a in graph.atoms:
        if a in symargs:
            identifiers.append(argnames[a])
        elif a.value is not None and a.value.atom_type == Integer.atom_type:
            identifiers.append(_double_to_hex(a.value.value))  # type: ignore
        else:
            raise LLVMNotImplementedError("No LLVM rule for: " + repr(a))

    args = ", ".join(f"double {argnames[arg]}" for arg in symargs)
    signature = f'define double @"jit_func1"({args})'

    instructions: list[str] = []
    for func, indices in graph.operations:
        n = len(instructions)
        identifier = f'%".{n}"'
        identifiers.append(identifier)
        argids = [identifiers[i] for i in indices]

        if func == Add.rep:
            line = f"{identifier} = fadd double " + ", ".join(argids)
        elif func == Mul.rep:
            line = f"{identifier} = fmul double " + ", ".join(argids)
        elif func == Pow.rep:
            args = f"double {argids[0]}, double {argids[1]}"
            line = f"{identifier} = call double @llvm.pow.f64({args})"
        elif func == sin.rep:
            line = f"{identifier} = call double @llvm.sin.f64(double {argids[0]})"
        elif func == cos.rep:
            line = f"{identifier} = call double @llvm.cos.f64(double {argids[0]})"
        else:
            raise LLVMNotImplementedError("No LLVM rule for: " + repr(func))

        instructions.append(line)

    instructions.append(f"ret double {identifiers[-1]}")

    function_lines = [signature, "{", *instructions, "}"]
    module_code = _llvm_header + "\n".join(function_lines)
    return module_code


def lambdify(args: list[Expr], expression: Expr | Matrix) -> Callable[..., Any]:
    """Turn ``expression`` into an efficient callable function of ``args``.

    >>> from protosym.simplecas import Symbol, sin, lambdify
    >>> x = Symbol('x')
    >>> f = lambdify([x], sin(x))
    >>> f(1)
    0.8414709848078965
    >>> import math; math.sin(1)
    0.8414709848078965
    """
    args_rep = [arg.rep for arg in args]
    if isinstance(expression, Expr):
        return _lambdify_llvm(args_rep, expression.rep)
    elif isinstance(expression, Matrix):
        return _lambdify_llvm_matrix(args_rep, expression)
    else:
        raise TypeError("Expression should be Expr or Matrix.")


_exe_eng = []


def _compile_llvm(module_code: str) -> Any:
    try:
        import llvmlite.binding as llvm
    except ImportError:  # pragma: no cover
        msg = "llvmlite needs to be installed to use lambdify_llvm."
        raise ImportError(msg) from None

    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    llmod = llvm.parse_assembly(module_code)

    pmb = llvm.create_pass_manager_builder()
    pmb.opt_level = 2
    pass_manager = llvm.create_module_pass_manager()
    pmb.populate(pass_manager)

    pass_manager.run(llmod)

    target_machine = llvm.Target.from_default_triple().create_target_machine()
    exe_eng = llvm.create_mcjit_compiler(llmod, target_machine)
    exe_eng.finalize_object()
    _exe_eng.append(exe_eng)

    fptr = exe_eng.get_function_address("jit_func1")
    return fptr


def _lambdify_llvm(args: list[Tree], expression: Tree) -> Callable[..., float]:
    """Lambdify using llvmlite."""
    module_code = _to_llvm_f64(args, expression)

    fptr = _compile_llvm(module_code)

    rettype = ctypes.c_double
    argtypes = [ctypes.c_double] * len(args)

    cfunc = ctypes.CFUNCTYPE(rettype, *argtypes)(fptr)
    return cfunc


def _to_llvm_f64_matrix(symargs: list[Tree], mat: Matrix) -> str:  # noqa [C901]
    """Code for LLVM IR function computing ``expression`` from ``symargs``."""
    elements_graph = bin_expand(mat.elements_graph.rep)

    graph = forward_graph(elements_graph)

    argnames = {s: f'%"{s}"' for s in symargs}  # noqa

    identifiers = []
    for a in graph.atoms:
        if a in symargs:
            identifiers.append(argnames[a])
        elif a.value is not None and a.value.atom_type == Integer.atom_type:
            identifiers.append(_double_to_hex(a.value.value))  # type: ignore
        else:
            raise LLVMNotImplementedError("No LLVM rule for: " + repr(a))

    all_args = ['double* %"_out"'] + [f"double {argnames[arg]}" for arg in symargs]
    all_args_str = ", ".join(all_args)
    signature = f'define void @"jit_func1"({all_args_str})'

    instructions: list[str] = []
    for func, indices in graph.operations[:-1]:
        n = len(instructions)
        identifier = f'%".{n}"'
        identifiers.append(identifier)
        argids = [identifiers[i] for i in indices]

        if func == Add.rep:
            line = f"{identifier} = fadd double " + ", ".join(argids)
        elif func == Mul.rep:
            line = f"{identifier} = fmul double " + ", ".join(argids)
        elif func == Pow.rep:
            args = f"double {argids[0]}, double {argids[1]}"
            line = f"{identifier} = call double @llvm.pow.f64({args})"
        elif func == sin.rep:
            line = f"{identifier} = call double @llvm.sin.f64(double {argids[0]})"
        elif func == cos.rep:
            line = f"{identifier} = call double @llvm.cos.f64(double {argids[0]})"
        else:
            raise LLVMNotImplementedError("No LLVM rule for: " + repr(func))

        instructions.append(line)

    # The above loop stops short of the final operation which should be the
    # List at the top of the stack. Now all values are computed and just need
    # to be copied to the relevant locations in the _out array.
    _, indices = graph.operations[-1]

    ncols = mat.ncols
    identifier_count = len(instructions)
    for (i, j), n in sorted(mat.entrymap.items()):
        raw_index = i * ncols + j
        identifier_value = identifiers[indices[n]]
        ptr = f'%".{identifier_count}"'
        identifier_count += 1
        line1 = f'{ptr} = getelementptr double, double* %"_out", i32 {raw_index}'
        line2 = f"store double {identifier_value}, double* {ptr}"
        instructions.append(line1)
        instructions.append(line2)

    instructions.append("ret void")

    function_lines = [signature, "{", *instructions, "}"]
    module_code = _llvm_header + "\n".join(function_lines)
    return module_code


def _lambdify_llvm_matrix(args: list[Tree], mat: Matrix) -> Callable[..., Any]:
    """Lambdify a matrix.

    >>> from protosym.simplecas import lambdify, Matrix
    >>> f = lambdify([], Matrix([[1, 2], [3, 4]]))
    >>> f()
    array([[1., 2.],
           [3., 4.]])
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        msg = "numpy needs to be installed to use lambdify_matrix."
        raise ImportError(msg) from None

    module_code = _to_llvm_f64_matrix(args, mat)

    fptr = _compile_llvm(module_code)

    c_float64 = ctypes.POINTER(ctypes.c_double)
    rettype = ctypes.c_double
    argtypes = [c_float64] + [ctypes.c_double] * len(args)

    cfunc = ctypes.CFUNCTYPE(rettype, *argtypes)(fptr)

    def npfunc(*args: float) -> Any:
        arr = np.zeros(mat.shape, np.float64)
        arr_p = arr.ctypes.data_as(c_float64)
        cfunc(arr_p, *args)
        return arr

    return npfunc
