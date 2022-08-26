import pytest
from sympy import sympify

from teaal.trans.coord_access import CoordAccess


def test_build_expr_symbol():
    assert CoordAccess.build_expr(sympify("x")).gen() == "x"


def test_build_expr_int():
    assert CoordAccess.build_expr(sympify(3)).gen() == "3"


def test_build_expr_rational():
    assert CoordAccess.build_expr(sympify(1) / 2).gen() == "1 / 2"


def test_build_expr_add():
    assert CoordAccess.build_expr(sympify("a + b + c")).gen() == "a + b + c"


def test_build_expr_mul():
    assert CoordAccess.build_expr(sympify("a * b * c")).gen() == "a * b * c"


def test_build_expr_unknown_func():
    with pytest.raises(ValueError) as excinfo:
        CoordAccess.build_expr(sympify("a ^ b"))

    assert str(
        excinfo.value) == "Unable to translate operator <class 'sympy.core.power.Pow'>"
