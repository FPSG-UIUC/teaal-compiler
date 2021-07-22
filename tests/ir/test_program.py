import pytest

from es2hfa.ir.partitioning import Partitioning
from es2hfa.ir.spacetime import SpaceTime
from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.spacetime import SpaceTimeParser
from es2hfa.parse.equation import EquationParser
from es2hfa.parse.einsum import Einsum
from es2hfa.parse.mapping import Mapping
from tests.utils.parse_tree import make_uniform_shape


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


def test_missing_decl():
    yaml = """
    einsum:
        declaration:
            A: []
        expressions:
            - A[] = b
    mapping:
        rank-order:
            A: []
            B: []
    """
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    with pytest.raises(ValueError) as excinfo:
        Program(einsum, mapping)
    assert str(excinfo.value) == "Undeclared tensor: B"


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


def test_apply_dyn_partitioning_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.apply_dyn_partitioning(Tensor("A", ["K", "M"]), "K")
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_apply_dyn_partitioning_():
    program = create_partitioned()
    program.add_einsum(0)
    program.apply_dyn_partitioning(program.get_tensors()[1], "K")

    assert program.get_tensors()[1] == Tensor("A", ["K1", "K0", "M"])


def test_apply_curr_loop_order_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.apply_curr_loop_order(Tensor("A", ["K", "M"]))
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_apply_curr_loop_order():
    program = create_loop_ordered()
    program.add_einsum(0)
    program.apply_curr_loop_order(program.get_output())

    Z = Tensor("Z", ["N", "M"])
    Z.set_is_output(True)
    assert program.get_output() == Z


def test_apply_final_loop_order_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.apply_final_loop_order(Tensor("A", ["K", "M"]))
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_apply_final_loop_order():
    program = create_loop_ordered()
    program.add_einsum(0)
    program.apply_final_loop_order(program.get_output())

    Z = Tensor("Z", ["N", "M"])
    Z.set_is_output(True)
    assert program.get_output() == Z


def test_apply_final_loop_order_partitioned():
    program = create_partitioned()
    program.add_einsum(0)

    program.apply_all_partitioning(program.get_output())
    program.apply_final_loop_order(program.get_output())

    Z = Tensor("Z", ["M2", "M1", "M0", "N"])
    Z.set_is_output(True)
    assert program.get_output() == Z


def test_apply_static_partitioning_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.apply_static_partitioning(Tensor("A", ["K", "M"]))
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_apply_static_partitioning_default():
    program = create_default()
    program.add_einsum(0)
    program.apply_static_partitioning(program.get_tensors()[1])

    assert program.get_tensors()[1] == Tensor("A", ["K", "M"])


def test_apply_static_partitioning_mapped():
    program = create_partitioned()
    program.add_einsum(0)
    program.apply_static_partitioning(program.get_tensors()[1])

    assert program.get_tensors()[1] == Tensor("A", ["K", "M2", "M1", "M0"])


def test_get_curr_loop_order_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.get_curr_loop_order()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_get_curr_loop_order():
    program = create_loop_ordered()
    program.add_einsum(0)

    assert program.get_curr_loop_order() == ["K", "N", "M"]


def test_get_curr_loop_order_partitioned():
    program = create_partitioned()
    program.add_einsum(0)
    assert program.get_curr_loop_order() == ["M", "N", "K"]

    program.start_partitioning("M")
    assert program.get_curr_loop_order() == ["M2", "M1", "M0", "N", "K"]


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


def test_get_final_loop_order_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.get_final_loop_order()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_get_final_loop_order():
    program = create_loop_ordered()
    program.add_einsum(0)

    assert program.get_final_loop_order() == ["K", "N", "M"]


def test_get_final_loop_order_partitioned():
    program = create_partitioned()
    program.add_einsum(0)
    assert program.get_final_loop_order() == [
        "M2", "M1", "M0", "N", "K1", "K0"]


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
        program.apply_curr_loop_order(tensor)

    program.reset()

    with pytest.raises(ValueError) as excinfo:
        program.get_curr_loop_order()
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


def test_start_partitioning_unconfigured():
    program = create_partitioned()

    with pytest.raises(ValueError) as excinfo:
        program.start_partitioning("M")
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_start_partitioning_mapped():
    program = create_partitioned()
    program.add_einsum(0)

    program.start_partitioning("M")

    assert program.get_curr_loop_order() == ["M2", "M1", "M0", "N", "K"]
