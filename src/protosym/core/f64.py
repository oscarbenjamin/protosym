"""Common functions for 64-bit floating point."""
import math
import operator

from protosym.core.pyexpr import PyExpr

py_str = PyExpr.function(str)
py_cos = PyExpr.function(math.cos)
x = PyExpr.symbol("x")
y = PyExpr.symbol("y")

expr = py_str(py_cos(x + 1 - x / y**2))

f64_sin = PyExpr.function(math.sin)
f64_cos = PyExpr.function(math.cos)
f64_pow = PyExpr.function(math.pow)
f64_mul = PyExpr.function(operator.mul)
f64_prod = PyExpr.function(math.prod)
f64_fsum = PyExpr.function(math.fsum)
