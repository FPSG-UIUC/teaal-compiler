from lark.tree import Tree
import pytest
from sympy import symbols

from es2hfa.ir.index_math import IndexMath
from es2hfa.ir.loop_order import LoopOrder
from es2hfa.ir.partitioning import Partitioning
from es2hfa.ir.program import Program
from es2hfa.ir.spacetime import SpaceTime
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.spacetime import SpaceTimeParser
from es2hfa.parse.equation import EquationParser
from es2hfa.parse.einsum import Einsum
from es2hfa.parse.mapping import Mapping
from tests.utils.parse_tree import *


def create_default():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
            C: [M, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    """
    return Program(Einsum.from_str(yaml), Mapping.from_str(yaml))


def create_loop_ordered():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
        loop-order:
            Z: [K, N, M]
    """
    return Program(Einsum.from_str(yaml), Mapping.from_str(yaml))


def create_partitioned():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
        partitioning:
            Z:
                K: [uniform_occupancy(A.5)]
                M: [uniform_shape(6), uniform_shape(3)]
    """
    return Program(Einsum.from_str(yaml), Mapping.from_str(yaml))


def create_rank_ordered():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
        rank-order:
            Z: [N, M]
            A: [M, K]
    """
    return Program(Einsum.from_str(yaml), Mapping.from_str(yaml))


def create_displayed(time):
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
        spacetime:
            Z:
                space: [N]
                time: """ + time
    return Program(Einsum.from_str(yaml), Mapping.from_str(yaml))


def create_conv():
    yaml = """
    einsum:
        declaration:
            F: [S]
            I: [W]
            O: [Q]
        expressions:
            - O[q] = sum(S).(I[q + s] + F[s])
    """
    return Program(Einsum.from_str(yaml), Mapping.from_str(yaml))


def test_add_einsum_missing_decl():
    yaml = """
    einsum:
        declaration:
            A: []
            B: []
        expressions:
            - A[] = B[] + C[]
    """
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))

    with pytest.raises(ValueError) as excinfo:
        program.add_einsum(0)
    assert str(excinfo.value) == "Undeclared tensor: C"


def test_apply_all_partitioning_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.apply_all_partitioning(Tensor("A", ["K", "M"]))
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_apply_all_partitioning_default():
    program = create_default()
    program.add_einsum(0)
    program.apply_all_partitioning(program.get_tensors()[1])

    assert program.get_tensors()[1] == Tensor("A", ["K", "M"])


def test_apply_all_partitioning_mapped():
    program = create_partitioned()
    program.add_einsum(0)
    program.apply_all_partitioning(program.get_tensors()[1])

    assert program.get_tensors()[1] == Tensor(
        "A", ["K1", "K0", "M2", "M1", "M0"])


def test_apply_partitioning_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.apply_partitioning(Tensor("A", ["K", "M"]), "K")
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_apply_partitioning():
    program = create_partitioned()
    program.add_einsum(0)
    program.apply_partitioning(program.get_tensors()[1], "M")

    assert program.get_tensors()[1] == Tensor("A", ["K", "M2", "M1", "M0"])


def test_get_einsum_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.get_einsum()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_get_einsum():
    program = create_default()
    program.add_einsum(0)

    equation = EquationParser.parse("Z[m, n] = sum(K).(A[k, m] * B[k, n])")
    assert program.get_einsum() == equation


def test_get_output_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.get_output()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_get_output():
    program = create_default()
    program.add_einsum(0)

    result = Tensor("Z", ["M", "N"])
    result.set_is_output(True)

    assert program.get_output() == result


def test_get_index_math_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.get_index_math()
    assert str(excinfo.value) == \
        "Unconfigured program. Make sure to first call add_einsum()"


def test_get_index_math_conv():
    program = create_conv()
    program.add_einsum(0)

    # Add the tensors
    index_math = IndexMath()
    index_math.add(Tensor("O", ["Q"]), make_tranks(["q"]))
    index_math.add(Tensor("I", ["W"]), Tree(
        "tranks", [make_iplus(["q", "s"])]))
    index_math.add(Tensor("F", ["S"]), make_tranks(["s"]))
    index_math.prune(
        program.get_loop_order().get_ranks(),
        program.get_partitioning())

    assert program.get_index_math() == index_math


def test_get_index_math_rank_ordered():
    program = create_rank_ordered()
    program.add_einsum(0)

    k, m, n = symbols("k m n")
    assert program.get_index_math().get_all_exprs("k") == [k]
    assert program.get_index_math().get_all_exprs("m") == [m]
    assert program.get_index_math().get_all_exprs("n") == [n]


def test_get_loop_order_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.get_loop_order()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_get_loop_order():
    program = create_loop_ordered()
    program.add_einsum(0)
    equation = EquationParser.parse("Z[m, n] = sum(K).(A[k, m] * B[k, n])")

    loop_order = LoopOrder(equation, program.get_output())
    ranks = ["K", "N", "M"]
    loop_order.add(ranks, program.get_index_math(), Partitioning({}, ranks))

    assert program.get_loop_order() == loop_order


def test_get_partitioning_unconfigured():
    program = create_partitioned()

    with pytest.raises(ValueError) as excinfo:
        program.get_partitioning()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_get_partitioning():
    program = create_default()
    program.add_einsum(0)

    assert program.get_partitioning() == Partitioning({}, ["M", "N", "K"])


def test_get_spacetime_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.get_spacetime()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_get_spacetime_unspecified():
    program = create_default()
    program.add_einsum(0)
    assert program.get_spacetime() is None


def test_get_spacetime_specified():
    program = create_displayed("[K.pos, M.coord]")
    program.add_einsum(0)

    yaml = {
        "space": [
            SpaceTimeParser.parse("N")],
        "time": [
            SpaceTimeParser.parse("K.pos"),
            SpaceTimeParser.parse("M.coord")]}
    spacetime = SpaceTime(yaml, Partitioning(
        {}, ["M", "N", "K"]), program.get_output().root_name())
    assert program.get_spacetime() == spacetime


def test_get_tensor_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.get_tensor("Z")
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_get_tensor():
    program = create_default()
    program.add_einsum(0)

    Z = Tensor("Z", ["M", "N"])
    Z.set_is_output(True)

    assert program.get_tensor("Z") == Z


def test_get_tensor_missing_tensor():
    program = create_default()
    program.add_einsum(0)

    with pytest.raises(ValueError) as excinfo:
        program.get_tensor("C")
    assert str(excinfo.value) == "Unknown tensor C"


def test_get_tensors_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.get_tensors()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_get_tensors():
    program = create_default()
    program.add_einsum(0)

    Z = Tensor("Z", ["M", "N"])
    Z.set_is_output(True)
    A = Tensor("A", ["K", "M"])
    B = Tensor("B", ["K", "N"])

    assert program.get_tensors() == [Z, A, B]


def test_get_tensors_ordered():
    program = create_rank_ordered()
    program.add_einsum(0)

    Z = Tensor("Z", ["N", "M"])
    Z.set_is_output(True)
    A = Tensor("A", ["M", "K"])
    B = Tensor("B", ["K", "N"])

    assert program.get_tensors() == [Z, A, B]


def test_reset():
    program = create_rank_ordered()
    program.add_einsum(0)

    for tensor in program.get_tensors():
        program.get_loop_order().apply(tensor)

    program.reset()

    with pytest.raises(ValueError) as excinfo:
        program.get_loop_order()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"

    with pytest.raises(ValueError) as excinfo:
        program.get_einsum()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"

    with pytest.raises(ValueError) as excinfo:
        program.get_partitioning()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"

    with pytest.raises(ValueError) as excinfo:
        program.get_tensors()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"

    program.add_einsum(0)

    Z = Tensor("Z", ["N", "M"])
    Z.set_is_output(True)
    A = Tensor("A", ["M", "K"])
    B = Tensor("B", ["K", "N"])

    assert program.get_tensors() == [Z, A, B]
