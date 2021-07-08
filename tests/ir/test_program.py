import pytest

from es2hfa.ir.display import Display
from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.einsum import EinsumParser
from es2hfa.parse.input import Input
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
    return Program(Input.from_str(yaml))


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
    return Program(Input.from_str(yaml))


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
                K: [uniform_shape(6), uniform_shape(3)]
                M: [uniform_shape(5)]
    """
    return Program(Input.from_str(yaml))


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
    return Program(Input.from_str(yaml))


def create_displayed(time, style):
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
        display:
            Z:
                space: [N]
                time: """ + time + """
                """ + style
    return Program(Input.from_str(yaml))


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
    input_ = Input.from_str(yaml)
    with pytest.raises(ValueError) as excinfo:
        Program(input_)
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
    program = Program(Input.from_str(yaml))

    with pytest.raises(ValueError) as excinfo:
        program.add_einsum(0)
    assert str(excinfo.value) == "Undeclared tensor: C"


def test_apply_loop_order_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.apply_loop_order(Tensor("A", ["K", "M"]))
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_apply_loop_order_default():
    program = create_default()
    program.add_einsum(0)
    program.apply_loop_order(program.get_tensors()[2])

    assert program.get_tensors()[2] == Tensor("B", ["N", "K"])


def test_apply_loop_order_ordered():
    program = create_loop_ordered()
    program.add_einsum(0)
    program.apply_loop_order(program.get_tensors()[0])

    Z = Tensor("Z", ["N", "M"])
    Z.set_is_output(True)
    assert program.get_tensors()[0] == Z


def test_apply_partitioning_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.apply_partitioning(Tensor("A", ["K", "M"]))
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_apply_partitioning_default():
    program = create_default()
    program.add_einsum(0)
    program.apply_partitioning(program.get_tensors()[2])

    assert program.get_tensors()[2] == Tensor("B", ["K", "N"])


def test_apply_partitiong_mapped():
    program = create_partitioned()
    program.add_einsum(0)
    program.apply_partitioning(program.get_tensors()[2])

    assert program.get_tensors()[2] == Tensor("B", ["K2", "K1", "K0", "N"])


def test_get_display_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.get_display()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_get_display_unspecified():
    program = create_default()
    program.add_einsum(0)
    assert program.get_display() is None


def test_get_display_specified():
    program = create_displayed("[K, M]", "style: shape")
    program.add_einsum(0)

    yaml = {"space": ["N"], "time": ["M", "K"], "style": "shape"}
    display = Display(yaml, program.get_loop_order(), {},
                      program.get_output().root_name())
    assert program.get_display() == display


def test_get_einsum_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.get_einsum()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_get_einsum():
    program = create_default()
    program.add_einsum(0)

    einsum = EinsumParser.parse("Z[m, n] = sum(K).(A[k, m] * B[k, n])")
    assert program.get_einsum() == einsum


def test_get_loop_order_unconfigured():
    program = create_default()

    with pytest.raises(ValueError) as excinfo:
        program.get_loop_order()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_get_loop_order_default():
    program = create_default()
    program.add_einsum(0)

    assert program.get_loop_order() == ["M", "N", "K"]


def test_get_loop_order_ordered():
    program = create_loop_ordered()
    program.add_einsum(0)

    assert program.get_loop_order() == ["K", "N", "M"]


def test_get_loop_order_default_partitioned():
    program = create_partitioned()
    program.add_einsum(0)
    assert program.get_loop_order() == ["M1", "M0", "N", "K2", "K1", "K0"]


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
        program.get_partitioning(Tensor("A", ["K", "M"]))
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"


def test_get_partitioning_default():
    program = create_default()
    program.add_einsum(0)

    assert program.get_partitioning(Tensor("A", ["K", "M"])) == {}


def test_get_partitioning_mapped():
    program = create_partitioned()
    program.add_einsum(0)

    assert program.get_partitioning(Tensor("B", ["K", "N"])) == {
        "K": make_uniform_shape([6, 3])}


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
        program.apply_loop_order(tensor)

    program.reset()

    with pytest.raises(ValueError) as excinfo:
        program.get_display()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"

    with pytest.raises(ValueError) as excinfo:
        program.get_loop_order()
    assert str(
        excinfo.value) == "Unconfigured program. Make sure to first call add_einsum()"

    with pytest.raises(ValueError) as excinfo:
        program.get_partitioning(Tensor("A", ["M", "K"]))
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


def test_default_loop_order_unconfigured():
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
    input_ = Input.from_str(yaml)
    program = Program(input_)
    with pytest.raises(ValueError) as excinfo:
        program._Program__default_loop_order()
    assert str(excinfo.value) == "Must first set the einsum"


def test_default_loop_order_no_partitioning():
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
    input_ = Input.from_str(yaml)
    program = Program(input_)
    program.einsum = input_.get_expressions()[0]
    with pytest.raises(ValueError) as excinfo:
        program._Program__default_loop_order()
    assert str(excinfo.value) == "Must configure partitioning before loop order"
