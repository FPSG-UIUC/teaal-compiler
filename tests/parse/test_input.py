from es2hfa.parse.einsum import EinsumParser
from es2hfa.parse.input import Input
from es2hfa.parse.tensor import TensorParser


def test_declaration():
    input_ = Input("tests/integration/test_input.yml")

    tensors = ["A[K, M]", "B[K, N]", "C[M, N]", "T1[M, N]", "Z[M, N]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]

    assert input_.get_declaration() == tensors


def test_expressions():
    input_ = Input("tests/integration/test_input.yml")

    T1 = EinsumParser.parse("T1[m, n] = sum(K).(A[k, m] * B[k, n])")
    Z = EinsumParser.parse("Z[m, n] = T1[m, n] + C[m, n]")

    assert input_.get_expressions() == [T1, Z]


def test_mapping():
    input_ = Input("tests/integration/test_input_no_mapping.yml")
    assert input_.get_rank_orders() == []
    assert input_.get_loop_orders() == {}


def test_rank_orders():
    input_ = Input("tests/integration/test_input.yml")

    tensors = ["A[M, K]", "C[N, M]", "Z[N, M]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]

    assert input_.get_rank_orders() == tensors


def test_rank_orders_missing():
    input_ = Input("tests/integration/test_input_no_rank_order.yml")
    tensors = []
    assert input_.get_rank_orders() == tensors


def test_loop_orders():
    input_ = Input("tests/integration/test_input.yml")
    assert input_.get_loop_orders() == {"T1": ["K", "N", "M"]}


def test_loop_orders_missing():
    input_ = Input("tests/integration/test_input_no_loop_order.yml")
    assert input_.get_loop_orders() == {}
