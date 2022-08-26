from lark.tree import Tree
import pytest
from sympy import symbols

from teaal.ir.coord_math import CoordMath
from teaal.ir.loop_order import LoopOrder
from teaal.ir.partitioning import Partitioning
from teaal.ir.tensor import Tensor
from teaal.parse.equation import EquationParser
from teaal.parse.mapping import Mapping
from tests.utils.parse_tree import *


def build_loop_order():
    equation = EquationParser.parse("Z[m, n] = sum(K).(A[k, m] * B[k, n])")

    output = Tensor("Z", ["M", "N"])
    output.set_is_output(True)

    return LoopOrder(equation, output)


def build_loop_order_conv():
    equation = EquationParser.parse("O[q] = sum(S).(I[q + s] * F[s])")

    output = Tensor("O", ["Q"])
    output.set_is_output(True)

    return LoopOrder(equation, output)


def build_partitioning(parts):
    yaml = """
    mapping:
        partitioning:
            Z:
    """ + parts
    dict_ = Mapping.from_str(yaml).get_partitioning()["Z"]

    k, m, n = symbols("k m n")
    eqn_exprs = {k: k, m: m, n: n}

    return Partitioning(dict_, ["K", "M", "N"], eqn_exprs)


def build_partitioning_conv(parts):
    yaml = """
    mapping:
        partitioning:
            O:
    """ + parts
    dict_ = Mapping.from_str(yaml).get_partitioning()["O"]

    q, s, w = symbols("q s w")
    eqn_exprs = {q: q, s: s, w: q + s}

    return Partitioning(dict_, ["Q", "S", "W"], eqn_exprs)


def build_coord_math():
    coord_math = CoordMath()

    coord_math.add(Tensor("A", ["K", "M"]), make_tranks(["k", "m"]))
    coord_math.add(Tensor("B", ["K", "N"]), make_tranks(["k", "n"]))
    coord_math.add(Tensor("Z", ["M", "N"]), make_tranks(["m", "n"]))

    return coord_math


def build_coord_math_conv():
    coord_math = CoordMath()

    coord_math.add(Tensor("I", ["W"]), Tree(
        "tranks", [make_iplus(["q", "s"])]))
    coord_math.add(Tensor("F", ["S"]), make_tranks(["s"]))
    coord_math.add(Tensor("O", ["Q"]), make_tranks(["q"]))

    return coord_math


def test_add_specified_no_partitioning():
    loop_order = build_loop_order()

    partitioning = build_partitioning("")
    coord_math = build_coord_math()
    loop_order.add(["K", "M", "N"], coord_math, partitioning)

    assert loop_order.get_ranks() == ["K", "M", "N"]


def test_add_default_no_partitioning():
    loop_order = build_loop_order()

    partitioning = build_partitioning("")
    coord_math = build_coord_math()
    loop_order.add(None, coord_math, partitioning)

    assert loop_order.get_ranks() == ["M", "N", "K"]


def test_add_specified_partitioning():
    loop_order = build_loop_order()

    parts = """
                K: [uniform_shape(4)]
                M: [uniform_shape(6), uniform_occupancy(A.3)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = build_partitioning(parts)
    coord_math = build_coord_math()
    order = ["K2", "N2", "K1", "M1", "N1", "K0", "M0", "N0"]
    loop_order.add(order, coord_math, partitioning)

    assert loop_order.get_ranks() == order


def test_apply_unconfigured():
    loop_order = build_loop_order()
    A = Tensor("A", ["M", "K", "N"])

    with pytest.raises(ValueError) as excinfo:
        loop_order.apply(A)

    assert str(
        excinfo.value) == "Unconfigured loop order. Make sure to first call add()"


def test_apply():
    parts = """
                K: [uniform_occupancy(A.6), uniform_occupancy(A.3)]
    """
    order = ["K2", "M", "K1", "N", "K0"]

    loop_order = build_loop_order()
    partitioning = build_partitioning(parts)
    coord_math = build_coord_math()

    loop_order.add(order, coord_math, partitioning)
    coord_math.prune(order, partitioning)

    A = Tensor("A", ["M", "K", "N"])
    loop_order.apply(A)
    assert A.get_ranks() == ["K", "M", "N"]

    A.update_ranks(partitioning.partition_tensor(A, {"K"}, False))
    loop_order.apply(A)
    assert A.get_ranks() == ["K2", "M", "K1I", "N"]

    A.update_ranks(partitioning.partition_tensor(A, {"K1I"}, False))
    loop_order.apply(A)
    assert A.get_ranks() == order


def test_apply_conv():
    loop_order = build_loop_order_conv()
    partitioning = build_partitioning_conv("")
    coord_math = build_coord_math_conv()

    order = ["W", "Q"]
    loop_order.add(order, coord_math, partitioning)
    coord_math.prune(order, partitioning)

    I = Tensor("I", ["W"])
    loop_order.apply(I)
    assert I.get_ranks() == ["W"]

    F = Tensor("F", ["S"])
    loop_order.apply(F)
    assert F.get_ranks() == ["S"]

    Z = Tensor("Z", ["Q"])
    loop_order.apply(Z)
    assert Z.get_ranks() == ["Q"]


def test_get_ranks_unconfigured():
    loop_order = build_loop_order()

    with pytest.raises(ValueError) as excinfo:
        loop_order.get_ranks()

    assert str(
        excinfo.value) == "Unconfigured loop order. Make sure to first call add()"


def test_default_loop_order_no_partitioning():
    loop_order = build_loop_order()

    with pytest.raises(ValueError) as excinfo:
        loop_order._LoopOrder__default_loop_order()

    assert str(excinfo.value) == "Must configure partitioning before loop order"


def test_default_loop_order_after_partitioning():
    loop_order = build_loop_order()

    parts = """
                K: [uniform_shape(4)]
                M: [uniform_shape(20), uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = build_partitioning(parts)
    coord_math = build_coord_math()
    loop_order.add(None, coord_math, partitioning)

    assert loop_order.get_ranks() == [
        "M2", "M1", "M0", "N2", "N1", "N0", "K1", "K0"]


def test_default_loop_order_conv():
    # Note that this test is not strictly necessary (as nothing changes when we
    # introduce coords as expressions), but it is a sanity check
    loop_order = build_loop_order_conv()
    partitioning = build_partitioning_conv("")
    coord_math = build_coord_math_conv()

    loop_order.add(None, coord_math, partitioning)

    assert loop_order.get_ranks() == ["Q", "S"]


def test_is_ready_unconfigured():
    loop_order = build_loop_order()

    with pytest.raises(ValueError) as excinfo:
        loop_order.is_ready("K", 2)

    assert str(
        excinfo.value) == "Unconfigured loop order. Make sure to first call add()"


def test_innermost_rank_unconfigured():
    loop_order = build_loop_order()

    with pytest.raises(ValueError) as excinfo:
        loop_order._LoopOrder__innermost_rank("K")

    assert str(
        excinfo.value) == "Unconfigured loop order. Make sure to first call add()"


def test_is_ready():
    loop_order = build_loop_order()

    parts = """
                K: [uniform_shape(6), uniform_occupancy(A.3)]
                M: [uniform_shape(4)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = build_partitioning(parts)
    coord_math = build_coord_math()

    order = ["K2", "N2", "K1", "M1", "N1", "K0", "M0", "N0"]
    loop_order.add(order, coord_math, partitioning)
    coord_math.prune(loop_order.get_ranks(), partitioning)

    assert loop_order.is_ready("N2", 1)
    assert loop_order.is_ready("K0", 5)
    assert not loop_order.is_ready("M0", 5)
    assert not loop_order.is_ready("N1", 5)


def test_is_ready_conv():
    loop_order = build_loop_order_conv()
    partitioning = build_partitioning_conv("")
    coord_math = build_coord_math_conv()

    order = ["W", "Q"]
    loop_order.add(order, coord_math, partitioning)
    coord_math.prune(loop_order.get_ranks(), partitioning)

    assert loop_order.is_ready("Q", 1)
    assert loop_order.is_ready("S", 1)
    assert not loop_order.is_ready("W", 1)


def test_eq():
    loop_order1 = build_loop_order()
    loop_order2 = build_loop_order()
    parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = build_partitioning(parts)
    coord_math = build_coord_math()
    loop_order1.add(
        ["N2", "K1", "M1", "N1", "K0", "M0", "N0"], coord_math, partitioning)
    loop_order2.add(
        ["N2", "K1", "M1", "N1", "K0", "M0", "N0"], coord_math, partitioning)

    assert loop_order1 == loop_order2


def test_neq_loop_order():
    loop_order1 = build_loop_order()
    loop_order2 = build_loop_order()
    parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = build_partitioning(parts)
    coord_math = build_coord_math()
    loop_order1.add(
        ["N2", "K1", "M1", "N1", "K0", "M0", "N0"], coord_math, partitioning)

    assert loop_order1 != loop_order2


def test_neq_other_type():
    loop_order1 = build_loop_order()
    parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = build_partitioning(parts)
    coord_math = build_coord_math()
    loop_order1.add(
        ["N2", "K1", "M1", "N1", "K0", "M0", "N0"], coord_math, partitioning)

    assert loop_order1 != ""
