from protosym.core.sym import SymAtomType
from protosym.simplecas import (
    Add,
    Expr,
    Function,
    Integer,
    List,
    Matrix,
    Mul,
    Pow,
    Symbol,
    a,
    b,
    cos,
    diff,
    expressify,
    f,
    g,
    lambdify,
    negone,
    one,
    sin,
    x,
    y,
    zero,
)
from protosym.simplecas.exceptions import ExpressifyError, LLVMNotImplementedError
from pytest import raises, skip

from .utils import requires_llvmlite, requires_numpy

two = Integer(2)


def test_simplecas_types() -> None:
    """Basic tests for type of Expr."""
    assert type(x) == Expr
    assert type(Mul) == Expr
    assert type(Integer) == SymAtomType
    assert type(Integer(1)) == Expr
    assert type(Mul(x, Integer(1))) == Expr
    raises(TypeError, lambda: Expr([]))  # type: ignore


def test_simplecas_equality() -> None:
    """Test equality of Expr."""
    unequal_pairs = [
        (x, y),
        (one, two),
        (sin(x), sin(y)),
    ]
    for e1, e2 in unequal_pairs:
        assert (e1 == e1) is True
        assert (e1 == e2) is False
        assert (e1 != e1) is False
        assert (e1 != e2) is True


def test_simplecas_identity() -> None:
    """Test that equal expressions are the same object."""
    identical_pairs = [
        (Symbol("x"), Symbol("x")),
        (Integer(3), Integer(3)),
        (sin(x), sin(x)),
    ]
    for e1, e2 in identical_pairs:
        assert e1 is e2


def test_xreplace() -> None:
    """Test simple substitutions with xreplace."""
    expr = x**2 + 1
    assert expr.xreplace({x: 1}) == Integer(1) ** 2 + 1
    assert expr.xreplace({x: y}) == y**2 + 1
    assert expr.xreplace({x**2: y}) == y + 1
    assert expr.xreplace({x**2 + 1: y}) == y
    assert expr.xreplace({y: 1}) == x**2 + 1
    assert expr.xreplace({1: 2}) == x**2 + 2


def test_simplecas_operations() -> None:
    """Test arithmetic operations with Expr."""
    assert +x == x
    assert -x == Mul(negone, x)
    assert x + x == Add(x, x)
    assert x - x == Add(x, Mul(negone, x))
    assert two * x == Mul(two, x)
    assert two / x == Mul(two, Pow(x, negone))
    assert two**x == Pow(two, x)


def test_simplecas_operations_expressify() -> None:
    """Test arithmetic operations with Expr."""
    assert x + 2 == x + two == Add(x, two)
    assert 2 + x == two + x == Add(two, x)
    assert x - 2 == x - two == Add(x, Mul(negone, two))
    assert 2 - x == two - x == Add(two, Mul(negone, x))
    assert x * 2 == x * two == Mul(x, two)
    assert 2 * x == two * x == Mul(two, x)
    assert x / 2 == x / two == Mul(x, Pow(two, negone))
    assert 2 / x == two / x == Mul(two, Pow(x, negone))
    assert x**2 == x**two == Pow(x, two)
    assert 2**x == two**x == Pow(two, x)


def test_simplecas_operations_bad_type() -> None:
    """Test arithmetic operations fail for Expr and other types."""
    bad_pairs = [(x, ()), ((), x)]
    for op1, op2 in bad_pairs:
        raises(TypeError, lambda: op1 + op2)  # type:ignore
        raises(TypeError, lambda: op1 - op2)  # type:ignore
        raises(TypeError, lambda: op1 * op2)  # type:ignore
        raises(TypeError, lambda: op1 / op2)  # type:ignore
        raises(TypeError, lambda: op1**op2)  # type:ignore


def test_simplecas_expressify() -> None:
    """Test that the expressify function works in basic cases."""
    assert expressify(1) == Integer(1)
    assert expressify(x) == x
    raises(ExpressifyError, lambda: expressify([]))


def test_simplecas_as_function() -> None:
    """Basic test for as_function."""
    assert cos(x).as_function(x)(1) == cos(1)
    assert cos(x + y).as_function(x)(1) == cos(1 + y)
    assert cos(x + y).as_function(y)(1) == cos(x + 1)
    assert (cos(x) + sin(y)).as_function(x, y)(1, 2) == cos(1) + sin(2)
    assert (cos(x) + sin(y)).as_function(y, x)(1, 2) == cos(2) + sin(1)
    assert cos(x).as_function(y)(1) == cos(x)
    assert cos(x).as_function(cos(x))(1) == Integer(1)


def test_simplecas_repr() -> None:
    """Test basic operations with simplecas."""
    assert str(Integer) == "Integer"
    assert str(x) == "x"
    assert str(y) == "y"
    assert str(f) == "f"
    assert str(sin) == "sin"
    assert str(f(x)) == "f(x)"
    assert str(sin(cos(x))) == "sin(cos(x))"
    assert str(x + y) == "(x + y)"
    assert str(one + two) == "(1 + 2)"
    assert str(x * y) == "(x*y)"
    assert str(x**two) == "x**2"
    assert str(x + x + x) == "((x + x) + x)"
    assert repr(x + x + x) == "((x + x) + x)"


def test_simplecas_latex() -> None:
    """Test basic operations with simplecas."""
    assert x.eval_latex() == r"x"
    assert y.eval_latex() == r"y"
    assert f(x).eval_latex() == "f(x)"
    assert sin(x).eval_latex() == r"\sin(x)"
    assert sin(cos(x)).eval_latex() == r"\sin(\cos(x))"
    assert (x + y).eval_latex() == r"(x + y)"
    assert (one + two).eval_latex() == r"(1 + 2)"
    assert (x * y).eval_latex() == r"(x \times y)"
    assert (x**two).eval_latex() == r"x^{2}"
    assert (x + x + x).eval_latex() == r"((x + x) + x)"


def test_simplecas_repr_latex() -> None:
    """Test IPython/Jupyter hook."""
    assert sin(x)._repr_latex_() == r"$\sin(x)$"


def test_simplecas_to_sympy() -> None:
    """Test converting a simplecas expression to a SymPy expression."""
    try:
        import sympy
    except ImportError:
        raise skip("SymPy not installed") from None

    x_sym = sympy.Symbol("x")
    sinx_sym = sympy.sin(x_sym)
    cosx_sym = sympy.cos(x_sym)
    f_sym = sympy.Function("f")

    test_cases = [
        (sin(x), sinx_sym),
        (cos(x), cosx_sym),
        (cos(x) ** 2 + sin(x) ** 2, cosx_sym**2 + sinx_sym**2),  # pyright: ignore
        (cos(x) * sin(x), cosx_sym * sinx_sym),  # pyright: ignore
        (f(x), f_sym(x_sym)),
    ]
    for expr, sympy_expr in test_cases:
        # XXX: Converting to SymPy and back does not in general round-trip
        # unless evaluate=False is used because SymPy otherwise modifies the
        # expression implicitly. for now it is useful to be able to convert to
        # SymPy and have it perform automatic evaluation but really there
        # should be a way to create a SymPy expression passing evaluate=False.
        #
        # Provided SymPy's automatic evaluation is idempotent an evaluated
        # SymPy expression will always round-trip through Expr though.
        assert expr.to_sympy() == sympy_expr
        assert Expr.from_sympy(sympy_expr) == expr
        assert expr == Expr.from_sympy(expr.to_sympy())
        assert sympy_expr == Expr.from_sympy(sympy_expr).to_sympy()

        # _sympy_ is used by sympify
        assert expr._sympy_() == sympy_expr
        assert sympy.sympify(expr) == sympy_expr

        # XXX: Ideally these would not compare equal because it could get
        # confusing. Unfortunately if sympify works then __eq__ will use it and
        # then compare the two objects. Maybe allowing sympify is a bad idea...
        assert expr == sympy_expr

    # No reason why li(x) in particular should be considered invalid. This test
    # just chooses an example of an expression that is not (yet) supported by
    # simplecas to verify that the appropriate error is raised. If this passes
    # in future because li support is added then a different example should be
    # chosen.
    raises(NotImplementedError, lambda: Expr.from_sympy(sympy.li(x_sym)))
    raises(NotImplementedError, lambda: Expr.from_sympy(sympy.ord0))


def test_simplecas_to_sympy_matrix() -> None:
    """Test converting to SymPy Matrix and back."""
    try:
        import sympy
    except ImportError:
        raise skip("SymPy not installed") from None

    x_sym = sympy.Symbol("x")
    sinx_sym = sympy.sin(x_sym)
    cosx_sym = sympy.cos(x_sym)
    M = sympy.Matrix([[sinx_sym, cosx_sym], [-cosx_sym, sinx_sym]])  # pyright: ignore
    assert M == Matrix.from_sympy(M).to_sympy()


def test_simplecas_eval_f64() -> None:
    """Test basic float evaluation with eval_f64."""
    assert sin(cos(x)).eval_f64({x: 1.0}) == 0.5143952585235492
    assert (x + one).eval_f64({x: 1.0}) == 2.0
    assert (x - one).eval_f64({x: 1.0}) == 0.0
    assert (x / two).eval_f64({x: 1.0}) == 0.5
    assert (x * two).eval_f64({x: 1.0}) == 2.0
    assert (x**two).eval_f64({x: 2.0}) == 4.0
    assert x.eval_f64({x: 1.0}) == 1.0
    assert one.eval_f64() == 1.0


def test_simplecas_count_ops() -> None:
    """Test count_ops_graph and count_ops_tree."""

    def make_expression(n: int) -> Expr:
        e = x**2 + x
        for _ in range(n):
            e = e**2 + e
        return e

    test_cases = [
        (x, 1, 1),
        (one, 1, 1),
        (sin(x), 2, 2),
        (sin(sin(x)) + sin(x), 4, 6),
        (sin(x) ** 2 + sin(x), 5, 7),
        (make_expression(10), 24, 8189),
        (make_expression(20), 44, 8388605),
        (make_expression(100), 204, 10141204801825835211973625643005),
    ]

    for expr, ops_graph, ops_tree in test_cases:
        assert expr.count_ops_graph() == ops_graph
        assert expr.count_ops_tree() == ops_tree


def test_simplecas_differentation() -> None:
    """Test derivatives of simplecas expressions."""
    assert one.diff(x) == zero
    assert x.diff(x) == one
    assert sin(1).diff(x) == zero
    assert (2 * sin(x)).diff(x) == 2 * cos(x)
    assert (x**3).diff(x) == 3 * x ** (Add(3, -1))
    assert sin(x).diff(x) == cos(x)
    assert cos(x).diff(x) == -sin(x)
    assert (sin(x) + cos(x)).diff(x) == cos(x) + -1 * sin(x)
    assert (sin(x) ** 2).diff(x) == 2 * sin(x) ** Add(2, -1) * cos(x)
    assert (x * sin(x)).diff(x) == 1 * sin(x) + x * cos(x)


def test_simplecas_differentiation_rules() -> None:
    """Test setting new differentation rules."""
    f = Function("f")
    diff[f(a), a] = 1 + f(a) ** 2
    assert diff(f(f(x)), x) == (1 + f(f(x)) ** 2) * (1 + f(x) ** 2)

    def set_bad1() -> None:
        diff[f(a), a, b] = f(a)  # type: ignore

    def set_bad2() -> None:
        diff[f(a, a), a] = f(a)

    raises(TypeError, set_bad1)
    raises(TypeError, set_bad2)


def test_simplecas_bin_expand() -> None:
    """Test Expr.bin_expand()."""
    expr1 = Add(1, 2, 3, 4)
    assert expr1.bin_expand() == Add(Add(Add(1, 2), 3), 4)
    assert str(expr1) == "(1 + 2 + 3 + 4)"
    assert str(expr1.bin_expand()) == "(((1 + 2) + 3) + 4)"

    expr2 = Add(x, y, Mul(x, y, 1, f(x)))
    assert expr2.bin_expand() == Add(Add(x, y), Mul(Mul(Mul(x, y), 1), f(x)))
    assert str(expr2) == "(x + y + (x*y*1*f(x)))"
    assert str(expr2.bin_expand()) == "((x + y) + (((x*y)*1)*f(x)))"


def test_simplecas_Matrix_constructor() -> None:
    """Test creating a Matrix."""
    M = Matrix([[1, 2, 3], [x, 0, 0]])
    assert isinstance(M, Matrix)
    assert M.nrows == 2
    assert M.ncols == 3
    assert M.shape == (2, 3)
    elements = [Integer(1), Integer(2), Integer(3), Symbol("x")]
    assert M.elements == elements
    assert M.elements_graph == List(*elements)
    assert M.entrymap == {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 0): 3}

    raises(TypeError, lambda: Matrix({}))  # type:ignore
    raises(TypeError, lambda: Matrix([{}]))  # type:ignore
    raises(TypeError, lambda: Matrix([[1, 2], [3]]))
    raises(TypeError, lambda: Matrix([[1, 2], [3, {}]]))  # type:ignore


def test_simplecas_Matrix_getitem() -> None:
    """Test indexing a Matrix."""
    M = Matrix([[1, 2], [x, 0]])
    assert M[0, 0] == Integer(1)
    assert M[0, 1] == Integer(2)
    assert M[1, 0] == Symbol("x")
    assert M[1, 1] == Integer(0)

    raises(TypeError, lambda: M[0])  # type:ignore
    raises(TypeError, lambda: M[0, 1, 2])  # type:ignore
    raises(TypeError, lambda: M[[], []])  # type:ignore
    raises(IndexError, lambda: M[-1, -1])
    raises(IndexError, lambda: M[2, 2])


def test_simplecas_Matrix_tolist() -> None:
    """Test converting Matrix to list of lists."""
    items = [[one, zero], [x, zero]]
    assert Matrix(items).tolist() == items


def test_simplecas_Matrix_repr() -> None:
    """Test Matrix repr."""
    M = Matrix([[1, 2], [x, 0]])
    assert repr(M) == "Matrix([[1, 2], [x, 0]])"


def test_simplecas_Matrix_add() -> None:
    """Test Matrix repr."""
    M1 = Matrix([[1, 0], [x, 0]])
    M2 = Matrix([[0, 1], [x, 0]])
    assert (M1 + M2).tolist() == [[one, one], [x + x, zero]]

    M3 = Matrix([[0]])
    raises(TypeError, lambda: M1 + M3)
    raises(TypeError, lambda: M1 + [])  # type:ignore


def test_simplecas_Matrix_diff() -> None:
    """Test Matrix repr."""
    M = Matrix([[1, 2], [x, 0]])
    assert M.diff(x).tolist() == [[zero, zero], [one, zero]]

    M = Matrix([[1, 2], [3, 4]])
    assert M.diff(x).tolist() == [[zero, zero], [zero, zero]]

    raises(TypeError, lambda: M.diff([]))  # type:ignore


def test_simplecas_to_llvm_ir() -> None:
    """Test converting Expr to LLVM IR."""
    expr1 = sin(cos(x)) * x**2 + 1
    expected = """
; ModuleID = "mod1"
target triple = "unknown-unknown-unknown"
target datalayout = ""

declare double    @llvm.pow.f64(double %Val1, double %Val2)
declare double    @llvm.sin.f64(double %Val)
declare double    @llvm.cos.f64(double %Val)

define double @"jit_func1"(double %"x")
{
%".0" = call double @llvm.cos.f64(double %"x")
%".1" = call double @llvm.sin.f64(double %".0")
%".2" = call double @llvm.pow.f64(double %"x", double 0x4000000000000000)
%".3" = fmul double %".1", %".2"
%".4" = fadd double %".3", 0x3ff0000000000000
ret double %".4"
}"""
    assert expr1.to_llvm_ir([x]) == expected

    raises(LLVMNotImplementedError, lambda: f(x).to_llvm_ir([x]))
    raises(LLVMNotImplementedError, lambda: f(x).to_llvm_ir([]))


@requires_llvmlite
def test_simplecas_lambdify_llvm() -> None:
    """Test simplecas lambdify function for a simple expression."""
    expr1 = sin(cos(x)) * x**2 + 1
    val1 = expr1.eval_f64({x: 1.0})
    f = lambdify([x], expr1)
    assert f(1) == f(1.0) == val1


@requires_numpy
@requires_llvmlite
def test_simplecas_lambdify_llvm_mat() -> None:
    """Test simplecas lambdify for a simple matrix."""
    import numpy as np

    M = Matrix([[1, 2], [3, 4]])
    f = lambdify([], M)
    assert np.all(f() == np.array([[1, 2], [3, 4]], float))

    M = Matrix([[cos(x), sin(x)], [-sin(x), cos(x)]])
    f = lambdify([x], M)
    expected = np.array([[np.cos(1), np.sin(1)], [-np.sin(1), np.cos(1)]])
    assert np.allclose(f(1), expected)

    M = Matrix([[x, y], [x + y, x**y]])
    f = lambdify([x, y], M)
    expected = np.array([[2, 3], [5, 8]])
    assert np.allclose(f(2, 3), expected)

    M = Matrix([[1, 2, 0], [4, 0, 6]])
    f = lambdify([], M)
    expected = np.array([[1, 2, 0], [4, 0, 6]], np.float64)
    assert np.all(f() == expected)

    z = Symbol("z")
    M = Matrix([[Add(x, y, z), Mul(x, y, z)]])
    f = lambdify([x, y, z], M)
    expected = np.array([[9, 24]], np.float64)
    assert np.all(f(2, 3, 4) == expected)

    raises(LLVMNotImplementedError, lambda: lambdify([x], Matrix([[g(x)]])))
    raises(LLVMNotImplementedError, lambda: lambdify([], Matrix([[x]])))


def test_simplecas_lambdify_llvm_bad() -> None:
    """Test lambdify unrecognised expression."""
    raises(TypeError, lambda: lambdify([x], {}))  # type:ignore


def test_simplecas_to_llvm_ir_matrix() -> None:
    """Test IR generation for a Matrix."""
    M = Matrix([[cos(x), sin(x)], [-sin(x), cos(x)]])
    assert (
        M.to_llvm_ir([x])
        == """
; ModuleID = "mod1"
target triple = "unknown-unknown-unknown"
target datalayout = ""

declare double    @llvm.pow.f64(double %Val1, double %Val2)
declare double    @llvm.sin.f64(double %Val)
declare double    @llvm.cos.f64(double %Val)

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
}"""
    )
