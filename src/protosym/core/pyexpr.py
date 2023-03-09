"""pyexpr: Symbolic representation of Python expressions."""
from __future__ import annotations

import functools
import operator
from typing import Any
from typing import TYPE_CHECKING as _TYPE_CHECKING
from weakref import WeakValueDictionary as _WeakDict

from protosym.core.atom import AtomType
from protosym.core.evaluate import Evaluator
from protosym.core.tree import forward_graph
from protosym.core.tree import TreeAtom
from protosym.core.tree import TreeExpr


PyFunction = AtomType("PyFunction", object)
PyClass = AtomType("PyClass", type)
PyObject = AtomType("PyObject", object)
PySymbol = AtomType("PySymbol", str)
PyOperation = AtomType("PyOperation", str)

PyCall = TreeAtom(PyOperation("PyCall"))
PyAdd = TreeAtom(PyOperation("PyAdd"))
PySub = TreeAtom(PyOperation("PySub"))
PyMul = TreeAtom(PyOperation("PyMul"))
PyTrueDiv = TreeAtom(PyOperation("PyTrueDiv"))
PyPow = TreeAtom(PyOperation("PyPow"))

eval_repr = Evaluator[str]()
eval_repr.add_atom(PyObject, str)
eval_repr.add_atom(PyFunction, lambda x: getattr(x, "__name__", "<noname>"))
eval_repr.add_atom(PySymbol, str)
eval_repr.add_opn(PyCall, lambda ch: f'{ch[0]}({", ".join(ch[1:])})')
eval_repr.add_op2(PyAdd, lambda a1, a2: f"({a1} + {a2})")
eval_repr.add_op2(PySub, lambda a1, a2: f"({a1} - {a2})")
eval_repr.add_op2(PyMul, lambda a1, a2: f"({a1}*{a2})")
eval_repr.add_op2(PyTrueDiv, lambda a1, a2: f"({a1}/{a2})")
eval_repr.add_op2(PyPow, lambda a1, a2: f"({a1}**{a2})")

evaluator = Evaluator[Any]()
evaluator.add_atom(PyObject, lambda x: x)
evaluator.add_atom(PyFunction, lambda x: x)
evaluator.add_opn(PyCall, lambda ch: ch[0](*ch[1:]))
evaluator.add_op2(PyAdd, operator.add)
evaluator.add_op2(PySub, operator.sub)
evaluator.add_op2(PyMul, operator.mul)
evaluator.add_op2(PyTrueDiv, operator.truediv)
evaluator.add_op2(PyPow, operator.pow)


if _TYPE_CHECKING:
    from typing import Callable, Union, Iterable

    PyExprBinOp = Callable[["PyExpr", "PyExpr"], "PyExpr"]
    PyExprBinOpConvert = Callable[["PyExpr", Union["PyExpr", object]], "PyExpr"]


def _convert_other(method: PyExprBinOp) -> PyExprBinOpConvert:
    """Decorator to convert other operand into PyExpr."""

    @functools.wraps(method)
    def new_method(self: PyExpr, other: PyExpr | object) -> PyExpr:
        if isinstance(other, PyExpr):
            return method(self, other)
        else:
            return method(self, PyExpr.pyobject(other))

    return new_method


class PyExpr:
    """Symbolic representation of Python code.

    >>> from math import cos
    >>> from protosym.core.pyexpr import PyExpr
    >>> x = PyExpr.symbol('x')
    >>> py_cos = PyExpr.function(cos)
    >>> py_cos(x)
    cos(x)
    >>> py_cos(x).evaluate({x:1}) # doctest: +ELLIPSIS
    0.5403023058681...
    """

    _all_expressions: _WeakDict[Any, Any] = _WeakDict()

    rep: TreeExpr

    def __new__(cls, tree_expr: TreeExpr) -> PyExpr:
        """Create an Expr from a TreeExpr."""
        if not isinstance(tree_expr, TreeExpr):
            raise TypeError("First argument to Expr should be TreeExpr")

        key = tree_expr

        expr = cls._all_expressions.get(tree_expr, None)
        if expr is not None:
            return expr  # type:ignore

        obj = super().__new__(cls)
        obj.rep = tree_expr

        obj = cls._all_expressions.setdefault(key, obj)

        return obj

    @classmethod
    def function(cls, func: Callable[..., Any]) -> PyExpr:
        """Wrap a Python function as a PyExpr."""
        return cls(TreeAtom(PyFunction(func)))

    @classmethod
    def symbol(cls, name: str) -> PyExpr:
        """Create a PyExpr symbol."""
        return cls(TreeAtom(PySymbol(name)))

    @classmethod
    def pyobject(cls, obj: object) -> PyExpr:
        """Wrap an arbitrary Python object as PyExpr."""
        return cls(TreeAtom(PyObject(obj)))

    def __repr__(self) -> str:
        """Pretty print."""
        return eval_repr(self.rep)

    def evaluate(self, values: dict[PyExpr, Any] | None = None) -> Any:
        """Evaluate the expression optionally replacing symbols with values."""
        if values is not None:
            values_rep = {expr.rep: value for expr, value in values.items()}
        else:
            values_rep = None
        return evaluator(self.rep, values_rep)

    def as_function(self, *args: PyExpr) -> SymFunction:
        """Convert this expression into a callable Python function."""
        return SymFunction(args, self)

    def __call__(self, *args: PyExpr | object) -> PyExpr:
        """Call a callable expression."""
        newargs = []
        for arg in args:
            if isinstance(arg, PyExpr):
                newargs.append(arg.rep)
            else:
                newargs.append(self.pyobject(arg).rep)
        return self._call(newargs)

    def _call(self, args: Iterable[TreeExpr]) -> PyExpr:
        return PyExpr(PyCall(self.rep, *args))

    @_convert_other
    def __add__(self, other: PyExpr) -> PyExpr:
        """Add PyExpr.

        >>> from protosym.core.pyexpr import PyExpr
        >>> x = PyExpr.symbol('x')
        >>> x + x
        (x + x)
        """
        return PyExpr(PyAdd(self.rep, other.rep))

    @_convert_other
    def __radd__(self, other: PyExpr) -> PyExpr:
        """Add PyExpr.

        >>> from protosym.core.pyexpr import PyExpr
        >>> x = PyExpr.symbol('x')
        >>> () + x
        (() + x)
        """
        return PyExpr(PyAdd(other.rep, self.rep))

    @_convert_other
    def __sub__(self, other: PyExpr) -> PyExpr:
        """Subtract PyExpr.

        >>> from protosym.core.pyexpr import PyExpr
        >>> x = PyExpr.symbol('x')
        >>> x - 1
        (x - 1)
        """
        return PyExpr(PySub(self.rep, other.rep))

    @_convert_other
    def __rsub__(self, other: PyExpr) -> PyExpr:
        """Subtract PyExpr.

        >>> from protosym.core.pyexpr import PyExpr
        >>> x = PyExpr.symbol('x')
        >>> 1 - x
        (1 - x)
        """
        return PyExpr(PySub(other.rep, self.rep))

    @_convert_other
    def __mul__(self, other: PyExpr) -> PyExpr:
        """Multiply two PyExpr.

        >>> from protosym.core.pyexpr import PyExpr
        >>> x = PyExpr.symbol('x')
        >>> x * x
        (x*x)
        """
        return PyExpr(PyMul(self.rep, other.rep))

    @_convert_other
    def __rmul__(self, other: PyExpr) -> PyExpr:
        """Multiply PyExpr.

        >>> from protosym.core.pyexpr import PyExpr
        >>> x = PyExpr.symbol('x')
        >>> 2 * x
        (2*x)
        """
        return PyExpr(PyMul(other.rep, self.rep))

    @_convert_other
    def __truediv__(self, other: PyExpr) -> PyExpr:
        """Divide PyExpr.

        >>> from protosym.core.pyexpr import PyExpr
        >>> x = PyExpr.symbol('x')
        >>> x / 2
        (x/2)
        """
        return PyExpr(PyTrueDiv(self.rep, other.rep))

    @_convert_other
    def __rtruediv__(self, other: PyExpr) -> PyExpr:
        """Divide PyExpr.

        >>> from protosym.core.pyexpr import PyExpr
        >>> x = PyExpr.symbol('x')
        >>> 2 / x
        (2/x)
        """
        return PyExpr(PyTrueDiv(other.rep, self.rep))

    @_convert_other
    def __pow__(self, other: PyExpr) -> PyExpr:
        """Raise PyExpr to power.

        >>> from protosym.core.pyexpr import PyExpr
        >>> x = PyExpr.symbol('x')
        >>> x ** 2
        (x**2)
        """
        return PyExpr(PyPow(self.rep, other.rep))

    @_convert_other
    def __rpow__(self, other: PyExpr) -> PyExpr:
        """Raise PyExpr to power.

        >>> from protosym.core.pyexpr import PyExpr
        >>> x = PyExpr.symbol('x')
        >>> 2 ** x
        (2**x)
        """
        return PyExpr(PyPow(other.rep, self.rep))


class SymFunction:
    """Represents a function in symbolic form.

    >>> from protosym.core.pyexpr import PyExpr, SymFunction
    >>> from math import sqrt
    >>> x = PyExpr.symbol('x')
    >>> y = PyExpr.symbol('y')
    >>> py_sqrt = PyExpr.function(sqrt)
    >>> expr = x + py_sqrt(y)
    >>> expr
    (x + sqrt(y))
    >>> f1 = SymFunction((x, y), expr)
    >>> f1
    SymFunction((x, y), (x + sqrt(y)))
    >>> f1(2, 1)
    3.0
    >>> f2 = expr.as_function(x, y)
    >>> f2(2, 1)
    3.0
    """

    params: tuple[PyExpr, ...]
    expr: PyExpr

    def __init__(self, params: tuple[PyExpr, ...], expr: PyExpr):
        """Create a new SymFunction directly.

        Usually :meth:`PyExpr.as_function` should be used instead.
        """
        self.params = params
        self.expr = expr

    def __repr__(self) -> str:
        """Code representation of this ``SymFunction``."""
        return f"SymFunction({self.params}, {self.expr})"

    def __call__(self, *args: Any) -> Any:
        """Call this function with Python objects as arguments.

        This evaluates calling this function by simulating a Python
        interpreter. The ``compile`` function will generate a function that is
        evaluated directly by the actual Python interpreter and can also be
        called to give equivalent reults.
        """
        if len(args) != len(self.params):
            raise TypeError()
        params_args = zip(self.params, args)
        values = {param: arg for param, arg in params_args}
        return self.expr.evaluate(values)

    def to_code(
        self, function_name: str = "_generated_function"
    ) -> tuple[str, dict[str, Any]]:
        """Convert function to code.

        >>> from protosym.core.pyexpr import PyExpr
        >>> x = PyExpr.symbol('x')
        >>> f = (x ** 2).as_function(x)
        >>> f
        SymFunction((x,), (x**2))
        >>> code, _ = f.to_code('square')
        >>> print(code)
        def square(x):
            x2 = x ** 2
            return x2

        When the expression involves an external function or other object that
        object will be injected into the ``namespace``:

        >>> from math import sqrt
        >>> py_sqrt = PyExpr.function(sqrt)
        >>> f = py_sqrt(x).as_function(x)
        >>> code, namespace = f.to_code('my_sqrt')
        >>> print(code)
        def my_sqrt(x):
            x2 = x0(x)
            return x2
        >>> namespace
        {'x0': <built-in function sqrt>}
        """
        bin_ops: dict[TreeExpr, str] = {
            PyAdd: "+",
            PySub: "-",
            PyMul: "*",
            PyTrueDiv: "/",
            PyPow: "**",
        }

        params = [p.rep for p in self.params]

        graph = forward_graph(self.expr.rep)

        namespace = {}
        total_vars = len(graph.atoms) + len(graph.operations)
        varnames = [f"x{i}" for i in range(total_vars)]

        for n1, atom in enumerate(graph.atoms):
            if atom in params:
                varnames[n1] = str(atom)
            elif isinstance(atom.value.value, int):
                varnames[n1] = str(atom.value.value)
            else:
                namespace[varnames[n1]] = atom.value.value

        argslist = ", ".join(map(str, self.params))
        sig_line = f"def {function_name}({argslist}):"

        body_lines = []

        for n2, (head, indices) in enumerate(graph.operations, n1 + 1):
            names = [varnames[i] for i in indices]
            if head == PyCall:
                funcname = names[0]
                argnames = names[1:]
                argliststr = ", ".join(argnames)
                expr = f"{funcname}({argliststr})"
            elif head in bin_ops:
                arg1, arg2 = names
                expr = f"{arg1} {bin_ops[head]} {arg2}"
            else:  # pragma: no cover
                raise NotImplementedError

            assignment = f"    {varnames[n2]} = {expr}"
            body_lines.append(assignment)

        return_line = f"    return {varnames[-1]}"

        lines = [sig_line, *body_lines, return_line]
        code = "\n".join(lines)
        return code, namespace

    def compile(self, function_name: str = "_generated_function") -> Callable[..., Any]:
        """Create a callable function by compiling the code from ``to_code``."""
        code, namespace = self.to_code(function_name)
        exec(code, namespace)  # noqa
        return namespace[function_name]  # type: ignore
