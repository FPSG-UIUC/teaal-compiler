from es2hfa.ir.tensor import Tensor
from es2hfa.parse.einsum import EinsumParser
from es2hfa.parse.input import Input
from tests.utils.parse_tree import make_uniform_shape


def test_declaration():
    input_ = Input.from_file("tests/integration/test_input.yaml")

    tensors = {
        "A": [
            "K", "M"], "B": [
            "K", "N"], "C": [
                "M", "N"], "T1": [
                    "M", "N"], "Z": [
                        "M", "N"]}
    assert input_.get_declaration() == tensors


def test_display():
    input_ = Input.from_file("tests/integration/test_input.yaml")
    display = {"T1": {"space": ["N"], "time": ["K", "M"]}}

    assert input_.get_display() == display


def test_display_missing():
    input_ = Input.from_file("tests/integration/test_input_no_display.yaml")
    assert input_.get_display() == {}


def test_eq():
    input_ = Input.from_file("tests/integration/test_input.yaml")
    assert input_ != "foo"


def test_expressions():
    input_ = Input.from_file("tests/integration/test_input.yaml")

    T1 = EinsumParser.parse("T1[m, n] = sum(K).(A[k, m] * B[k, n])")
    Z = EinsumParser.parse("Z[m, n] = T1[m, n] + C[m, n]")

    assert input_.get_expressions() == [T1, Z]


def test_from():
    yaml = """
    einsum:
        declaration:
            A: [K, M]
            B: [K, N]
            C: [M, N]
            T1: [M, N]
            Z: [M, N]
        expressions:
            - T1[m, n] = sum(K).(A[k, m] * B[k, n])
            - Z[m, n] = T1[m, n] + C[m, n]
    mapping:
        rank-order:
            A: [M, K]
            C: [N, M]
            Z: [N, M]
        loop-order:
            T1: [K, N, M]
            Z: [M2, N2, M1, N1, M0, N0]
        partitioning:
            Z:
                M: [uniform_shape(4), uniform_shape(2)]
                N: [uniform_shape(6), uniform_shape(3)]
        display:
            T1:
                space: [N]
                time: [K, M]
    """
    from_file = Input.from_file("tests/integration/test_input.yaml")
    from_str = Input.from_str(yaml)
    assert from_file == from_str


def test_loop_orders():
    input_ = Input.from_file("tests/integration/test_input.yaml")
    assert input_.get_loop_orders() == {
        "T1": [
            "K", "N", "M"], "Z": [
            "M2", "N2", "M1", "N1", "M0", "N0"]}


def test_loop_orders_missing():
    input_ = Input.from_file("tests/integration/test_input_no_loop_order.yaml")
    assert input_.get_loop_orders() == {}


def test_no_mapping():
    input_ = Input.from_file("tests/integration/test_input_no_mapping.yaml")

    assert input_.get_display() == {}
    assert input_.get_loop_orders() == {}
    assert input_.get_partitioning() == {}
    assert input_.get_rank_orders() == {}


def test_partitioning():
    input_ = Input.from_file("tests/integration/test_input.yaml")
    partitioning = {"Z": {"M": make_uniform_shape(
        [4, 2]), "N": make_uniform_shape([6, 3])}}

    assert input_.get_partitioning() == partitioning


def test_partitioning_missing():
    input_ = Input.from_file(
        "tests/integration/test_input_no_partitioning.yaml")
    assert input_.get_partitioning() == {}


def test_rank_orders():
    input_ = Input.from_file("tests/integration/test_input.yaml")

    tensors = ["A[M, K]", "C[N, M]", "Z[N, M]"]
    tensors = {"A": ["M", "K"], "C": ["N", "M"], "Z": ["N", "M"]}

    assert input_.get_rank_orders() == tensors


def test_rank_orders_missing():
    input_ = Input.from_file("tests/integration/test_input_no_rank_order.yaml")
    tensors = {}
    assert input_.get_rank_orders() == tensors
