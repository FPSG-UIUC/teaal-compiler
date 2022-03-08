import pytest

from es2hfa.ir.loop_order import LoopOrder
from es2hfa.ir.partitioning import Partitioning
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.equation import EquationParser
from es2hfa.parse.mapping import Mapping


def build_loop_order():
    equation = EquationParser.parse("Z[m, n] = sum(K).(A[k, m] * B[k, n])")

    output = Tensor("Z", ["M", "N"])
    output.set_is_output(True)

    return LoopOrder(equation, output)


def build_partitioning(loop_order, parts):
    yaml = """
    mapping:
        partitioning:
            Z:
    """ + parts
    dict_ = Mapping.from_str(yaml).get_partitioning()["Z"]
    return Partitioning(dict_, loop_order.get_unpartitioned_ranks())


def test_add_specified_no_partitioning():
    loop_order = build_loop_order()

    partitioning = build_partitioning(loop_order, "")
    loop_order.add(["K", "M", "N"], partitioning)

    assert loop_order.get_ranks() == ["K", "M", "N"]


def test_add_default_no_partitioning():
    loop_order = build_loop_order()

    partitioning = build_partitioning(loop_order, "")
    loop_order.add(None, partitioning)

    assert loop_order.get_ranks() == ["M", "N", "K"]


def test_add_specified_partitioning():
    loop_order = build_loop_order()

    parts = """
                K: [uniform_shape(4)]
                M: [uniform_shape(6), uniform_occupancy(A.3)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = build_partitioning(loop_order, parts)
    order = ["K2", "N2", "K1", "M1", "N1", "K0", "M0", "N0"]
    loop_order.add(order, partitioning)

    assert loop_order.get_ranks() == order


def test_apply_unconfigured():
    loop_order = build_loop_order()
    A = Tensor("A", ["M", "K", "N"])

    with pytest.raises(ValueError) as excinfo:
        loop_order.apply(A)

    assert str(
        excinfo.value) == "Unconfigured loop order. Make sure to first call add()"


def test_apply():
    loop_order = build_loop_order()
    parts = """
                K: [uniform_occupancy(A.6), uniform_occupancy(A.3)]
    """
    partitioning = build_partitioning(loop_order, parts)
    loop_order.add(
        ["K2", "M", "K1", "N", "K0"], partitioning)

    A = Tensor("A", ["M", "K", "N"])
    loop_order.apply(A)
    assert A.get_ranks() == ["K", "M", "N"]

    A.partition(partitioning, {"K"}, False)
    loop_order.apply(A)
    assert A.get_ranks() == ["K2", "M", "K1I", "N"]

    A.partition(partitioning, {"K1I"}, False)
    loop_order.apply(A)
    assert A.get_ranks() == ["K2", "M", "K1", "N", "K0"]


def test_default_loop_order_after_partitioning():
    loop_order = build_loop_order()

    parts = """
                K: [uniform_shape(4)]
                M: [uniform_shape(20), uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = build_partitioning(loop_order, parts)
    loop_order.add(None, partitioning)

    assert loop_order.get_ranks() == [
        "M2", "M1", "M0", "N2", "N1", "N0", "K1", "K0"]


def test_get_unpartitioned_ranks():
    loop_order = build_loop_order()
    assert loop_order.get_unpartitioned_ranks() == ["M", "N", "K"]


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


def test_eq():
    loop_order1 = build_loop_order()
    loop_order2 = build_loop_order()
    parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = build_partitioning(loop_order1, parts)
    loop_order1.add(
        ["N2", "K1", "M1", "N1", "K0", "M0", "N0"], partitioning)
    loop_order2.add(
        ["N2", "K1", "M1", "N1", "K0", "M0", "N0"], partitioning)

    assert loop_order1 == loop_order2


def test_neq_loop_order():
    loop_order1 = build_loop_order()
    loop_order2 = build_loop_order()
    parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = build_partitioning(loop_order1, parts)
    loop_order1.add(
        ["N2", "K1", "M1", "N1", "K0", "M0", "N0"], partitioning)

    assert loop_order1 != loop_order2


def test_neq_other_type():
    loop_order1 = build_loop_order()
    parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = build_partitioning(loop_order1, parts)
    loop_order1.add(
        ["N2", "K1", "M1", "N1", "K0", "M0", "N0"], partitioning)

    assert loop_order1 != ""
