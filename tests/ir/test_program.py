from lark.tree import Tree
import pytest
from sympy import symbols

from teaal.ir.coord_math import CoordMath
from teaal.ir.equation import Equation
from teaal.ir.loop_order import LoopOrder
from teaal.ir.partitioning import Partitioning
from teaal.ir.program import Program
from teaal.ir.spacetime import SpaceTime
from teaal.ir.tensor import Tensor
from teaal.parse.spacetime import SpaceTimeParser
from teaal.parse.equation import EquationParser
from teaal.parse.einsum import Einsum
from teaal.parse.mapping import Mapping
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
            - Z[m, n] = A[k, m] * B[k, n]
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
            - Z[m, n] = A[k, m] * B[k, n]
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
            - Z[m, n] = A[k, m] * B[k, n]
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
            - Z[m, n] = A[k, m] * B[k, n]
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
            - Z[m, n] = A[k, m] * B[k, n]
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
            - O[q] = I[q + s] * F[s]
    """
    return Program(Einsum.from_str(yaml), Mapping.from_str(yaml))


def test_apply_all_partitioning_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.apply_all_partitioning(Tensor("A", ["K", "M"]))
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_apply_all_partitioning_default():
    program = create_default()
    program.add_einsum(0)
    program.apply_all_partitioning(program.get_equation().get_tensor("A"))

    assert program.get_equation().get_tensor("A") == Tensor("A", ["K", "M"])


def test_apply_all_partitioning_mapped():
    program = create_partitioned()
    program.add_einsum(0)
    program.apply_all_partitioning(program.get_equation().get_tensor("A"))

    assert program.get_equation().get_tensor("A") == Tensor(
        "A", ["K1", "K0", "M2", "M1", "M0"])


def test_apply_partitioning_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.apply_partitioning(Tensor("A", ["K", "M"]), ("K",))
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_all_rank_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program._Program__all_ranks()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_all_ranks():
    yaml = """
    einsum:
        declaration:
            Z: [K]
            A: [J]
        expressions:
            - Z[k] = A[2 * k]
    """
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    assert program._Program__all_ranks() == {"J", "K"}


def test_apply_partitioning():
    program = create_partitioned()
    program.add_einsum(0)
    program.apply_partitioning(program.get_equation().get_tensor("A"), ("M",))

    assert program.get_equation().get_tensor(
        "A") == Tensor("A", ["K", "M2", "M1", "M0"])


def test_apply_partition_swizzling_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.apply_partition_swizzling(Tensor("A", ["K", "M"]))
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_apply_partition_swizzling():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [J, K, M, N]
        expressions:
            - Z[m, n] = A[j, k, m, n]
    mapping:
        partitioning:
            Z:
                K: [uniform_shape(4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
    """
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    A = program.get_equation().get_tensor("A")
    program.apply_partition_swizzling(A)
    assert A.get_ranks() == ["J", "K", "M", "N"]

    program.apply_partitioning(A, ("K",))
    program.apply_partition_swizzling(A)
    assert A.get_ranks() == ["J", "K1", "N", "M", "K0"]


def test_get_equation_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.get_equation()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_get_equation():
    program = create_default()
    program.add_einsum(0)

    tensors = {"Z": Tensor("Z", ["M", "N"]), "A": Tensor("A", ["K", "M"]),
               "B": Tensor("B", ["K", "N"]), "C": Tensor("C", ["M", "N"])}
    tensors["Z"].set_is_output(True)
    equation = Equation(EquationParser.parse(
        "Z[m, n] = A[k, m] * B[k, n]"), tensors)
    assert program.get_equation() == equation


def test_get_einsum_ind_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.get_einsum_ind()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_get_einsum_ind():
    program = create_default()
    program.add_einsum(0)

    assert program.get_einsum_ind() == 0


def test_get_coord_math_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.get_coord_math()
    assert str(excinfo.value) == \
        "Unconfigured program. Make sure to first call add_einsum()"


def test_get_coord_math_conv():
    program = create_conv()
    program.add_einsum(0)

    # Add the tensors
    coord_math = CoordMath()
    coord_math.add(Tensor("O", ["Q"]), make_ranks(["q"]))
    coord_math.add(Tensor("I", ["W"]), Tree(
        "ranks", [make_iplus(["q", "s"])]))
    coord_math.add(Tensor("F", ["S"]), make_ranks(["s"]))
    coord_math.prune({"Q", "S"})

    assert program.get_coord_math() == coord_math


def test_get_coord_math_rank_ordered():
    program = create_rank_ordered()
    program.add_einsum(0)

    k, m, n = symbols("k m n")
    assert program.get_coord_math().get_all_exprs("k") == [k]
    assert program.get_coord_math().get_all_exprs("m") == [m]
    assert program.get_coord_math().get_all_exprs("n") == [n]


def test_get_loop_order_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.get_loop_order()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_get_loop_order():
    program = create_loop_ordered()
    program.add_einsum(0)
    einsum = EquationParser.parse("Z[m, n] = A[k, m] * B[k, n]")
    equation = Equation(einsum, program.tensors)

    loop_order = LoopOrder(equation)
    ranks = ["K", "N", "M"]
    loop_order.add(
        ranks,
        program.get_coord_math(),
        Partitioning(
            {},
            ranks,
            program.get_coord_math()))

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

    assert program.get_partitioning() == Partitioning(
        {}, ["M", "N", "K"], program.get_coord_math())


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
    spacetime = SpaceTime(
        yaml,
        Partitioning({}, ["M", "N", "K"], program.get_coord_math()),
        program.get_equation().get_output().root_name())
    assert program.get_spacetime() == spacetime


def test_reset():
    program = create_rank_ordered()
    program.add_einsum(0)

    for tensor in program.get_equation().get_tensors():
        program.get_loop_order().apply(tensor)

    program.reset()

    with pytest.raises(ValueError) as excinfo:
        program.get_loop_order()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"

    with pytest.raises(ValueError) as excinfo:
        program.get_equation()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"

    with pytest.raises(ValueError) as excinfo:
        program.get_partitioning()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"

    program.add_einsum(0)

    Z = Tensor("Z", ["N", "M"])
    Z.set_is_output(True)
    A = Tensor("A", ["M", "K"])
    B = Tensor("B", ["K", "N"])

    assert program.get_equation().get_tensors() == [Z, A, B]
