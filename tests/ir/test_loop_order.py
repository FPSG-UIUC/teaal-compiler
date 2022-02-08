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


def test_add_loop_order_specified_no_partitioning():
    loop_order = build_loop_order()

    partitioning = build_partitioning(loop_order, "")
    loop_order.add_loop_order(["K", "M", "N"], partitioning)

    assert loop_order.get_curr_loop_order() == ["K", "M", "N"]


def test_add_loop_order_default_no_partitioning():
    loop_order = build_loop_order()

    partitioning = build_partitioning(loop_order, "")
    loop_order.add_loop_order(None, partitioning)

    assert loop_order.get_curr_loop_order() == ["M", "N", "K"]


def test_add_loop_order_specified_partitioning():
    loop_order = build_loop_order()

    parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = build_partitioning(loop_order, parts)
    loop_order.add_loop_order(
        ["N2", "K1", "M1", "N1", "K0", "M0", "N0"], partitioning)

    assert loop_order.get_curr_loop_order() == ["N", "K", "M"]
    assert loop_order.get_final_loop_order() == [
        "N2", "K1", "M1", "N1", "K0", "M0", "N0"]


def test_default_loop_order_after_partitioning():
    loop_order = build_loop_order()

    parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = build_partitioning(loop_order, parts)
    loop_order.add_loop_order(None, partitioning)

    partitioning.partition_rank("K")
    partitioning.partition_rank("M")
    partitioning.partition_rank("N")
    loop_order.update_loop_order()

    assert loop_order.get_curr_loop_order() == [
        "M1", "M0", "N2", "N1", "N0", "K1", "K0"]


def test_get_unpartitioned_ranks():
    loop_order = build_loop_order()
    assert loop_order.get_unpartitioned_ranks() == ["M", "N", "K"]


def test_get_curr_loop_order_unconfigured():
    loop_order = build_loop_order()

    with pytest.raises(ValueError) as excinfo:
        loop_order.get_curr_loop_order()

    assert str(
        excinfo.value) == "Unconfigured loop order. Make sure to first call add_loop_order()"


def test_get_final_loop_order_unconfigured():
    loop_order = build_loop_order()

    with pytest.raises(ValueError) as excinfo:
        loop_order.get_final_loop_order()

    assert str(
        excinfo.value) == "Unconfigured loop order. Make sure to first call add_loop_order()"


def test_update_loop_order_unconfigured():
    loop_order = build_loop_order()

    with pytest.raises(ValueError) as excinfo:
        loop_order.update_loop_order()

    assert str(
        excinfo.value) == "Unconfigured loop order. Make sure to first call add_loop_order()"


def test_update_loop_order_partitioning():
    loop_order = build_loop_order()

    parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = build_partitioning(loop_order, parts)
    loop_order.add_loop_order(
        ["N2", "K1", "M1", "N1", "K0", "M0", "N0"], partitioning)

    partitioning.partition_rank("N")
    loop_order.update_loop_order()

    assert loop_order.get_curr_loop_order() == ["N2", "K", "M", "N1", "N0"]


def test_default_loop_order_no_partitioning():
    loop_order = build_loop_order()

    with pytest.raises(ValueError) as excinfo:
        loop_order._LoopOrder__default_loop_order()

    assert str(excinfo.value) == "Must configure partitioning before loop order"
