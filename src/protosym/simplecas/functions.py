"""Basic functions and operations."""
import math

from protosym.core.sym import (
    AtomFunc,
    AtomRule,
    HeadOp,
    HeadRule,
    PyFunc1,
    PyOp1,
    PyOp2,
    PyOpN,
    star,
)
from protosym.simplecas.expr import (
    Add,
    Expr,
    Function,
    Integer,
    Mul,
    Symbol,
    cos,
    eval_f64,
    eval_latex,
    eval_repr,
    sin,
)

a = Expr.new_wild("a")
b = Expr.new_wild("b")


# ------------------------------------------------------------------------- #
#                                                                           #
#     eval_f64: 64 bit floating point evaluation.                           #
#                                                                           #
# ------------------------------------------------------------------------- #

f64_from_int = PyFunc1[int, float](float)
f64_add = PyOpN[float](math.fsum)
f64_mul = PyOpN[float](math.prod)
f64_pow = PyOp2[float](math.pow)
f64_sin = PyOp1[float](math.sin)
f64_cos = PyOp1[float](math.cos)

eval_f64[Integer[a]] = f64_from_int(a)
eval_f64[Add(star(a))] = f64_add(a)
eval_f64[Mul(star(a))] = f64_mul(a)
eval_f64[a**b] = f64_pow(a, b)
eval_f64[sin(a)] = f64_sin(a)
eval_f64[cos(a)] = f64_cos(a)

# ------------------------------------------------------------------------- #
#                                                                           #
#     eval_repr: Pretty string representation                               #
#                                                                           #
# ------------------------------------------------------------------------- #

repr_atom = AtomFunc[str](str)
repr_call = HeadOp[str](lambda head, args: f'{head}({", ".join(args)})')
str_from_int = PyFunc1[int, str](str)
str_from_str = PyFunc1[str, str](str)
repr_add = PyOpN[str](lambda args: f'({" + ".join(args)})')
repr_mul = PyOpN[str](lambda args: f'({"*".join(args)})')
repr_pow = PyOp2[str](lambda b, e: f"{b}**{e}")


eval_repr[HeadRule(a, b)] = repr_call(a, b)
eval_repr[AtomRule[a]] = repr_atom(a)
eval_repr[Integer[a]] = str_from_int(a)
eval_repr[Symbol[a]] = str_from_str(a)
eval_repr[Function[a]] = str_from_str(a)
eval_repr[Add(star(a))] = repr_add(a)
eval_repr[Mul(star(a))] = repr_mul(a)
eval_repr[a**b] = repr_pow(a, b)

# ------------------------------------------------------------------------- #
#                                                                           #
#     latex: LaTeX string representation                                    #
#                                                                           #
# ------------------------------------------------------------------------- #

latex_add = PyOpN(lambda args: f'({" + ".join(args)})')
latex_mul = PyOpN(lambda args: "(%s)" % r" \times ".join(args))
latex_pow = PyOp2(lambda b, e: f"{b}^{{{e}}}")
latex_sin = PyOp1(lambda a: rf"\sin({a})")
latex_cos = PyOp1(lambda a: rf"\cos({a})")

eval_latex[HeadRule(a, b)] = repr_call(a, b)
eval_latex[Integer[a]] = str_from_int(a)
eval_latex[Symbol[a]] = str_from_str(a)
eval_latex[Function[a]] = str_from_str(a)
eval_latex[Add(star(a))] = latex_add(a)
eval_latex[Mul(star(a))] = latex_mul(a)
eval_latex[a**b] = latex_pow(a, b)
eval_latex[sin(a)] = latex_sin(a)
eval_latex[cos(a)] = latex_cos(a)
