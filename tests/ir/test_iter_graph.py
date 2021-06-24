import pytest

from es2hfa.ir.iter_graph import IterationGraph
from es2hfa.ir.mapping import Mapping
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.input import Input


def test_peek_rank0():
    yaml = """
    einsum:
        declaration:
            A: []
        expressions:
            - A[] = b
    """
    mapping = Mapping(Input.from_str(yaml))
    mapping.add_einsum(0)
    graph = IterationGraph(mapping)

    tensor = Tensor("A", [])
    tensor.set_is_output(True)

    assert graph.peek() == (None, [tensor])


def test_peek_default():
    yaml = """
    einsum:
        declaration:
            A: [I, J]
            B: [I, K]
            C: [J, K]
        expressions:
            - A[i, j] = sum(K).(B[i, k] * C[j, k])
    """
    mapping = Mapping(Input.from_str(yaml))
    mapping.add_einsum(0)
    graph = IterationGraph(mapping)

    results = [Tensor("A", ["I", "J"]), Tensor("B", ["I", "K"])]
    results[0].set_is_output(True)

    assert graph.peek() == ("i", results)


def test_peek_order():
    yaml = """
    einsum:
        declaration:
            A: [I, J]
            B: [I, K]
            C: [J, K]
        expressions:
            - A[i, j] = sum(K).(B[i, k] * C[j, k])
    mapping:
        loop-order:
            A: [J, K, I]
    """
    mapping = Mapping(Input.from_str(yaml))
    mapping.add_einsum(0)

    for tensor in mapping.get_tensors():
        mapping.apply_loop_order(tensor)
    graph = IterationGraph(mapping)

    results = [Tensor("A", ["J", "I"]), Tensor("C", ["J", "K"])]
    results[0].set_is_output(True)

    assert graph.peek() == ("j", results)


def test_pop_default():
    yaml = """
    einsum:
        declaration:
            A: [I, J]
            B: [I, K]
            C: [J, K]
        expressions:
            - A[i, j] = sum(K).(B[i, k] * C[j, k])
    """
    mapping = Mapping(Input.from_str(yaml))
    mapping.add_einsum(0)
    graph = IterationGraph(mapping)

    A = Tensor("A", ["I", "J"])
    A.set_is_output(True)
    B = Tensor("B", ["I", "K"])
    C = Tensor("C", ["J", "K"])

    assert graph.pop() == ("i", [A, B])
    assert graph.pop() == ("j", [C, A])
    assert graph.pop() == ("k", [B, C])
    assert graph.peek() == (None, [A, B, C])


def test_pop_order():
    yaml = """
    einsum:
        declaration:
            A: [I, J]
            B: [I, K]
            C: [J, K]
        expressions:
            - A[i, j] = sum(K).(B[i, k] * C[j, k])
    mapping:
        loop-order:
            A: [J, K, I]
    """
    mapping = Mapping(Input.from_str(yaml))
    mapping.add_einsum(0)

    for tensor in mapping.get_tensors():
        mapping.apply_loop_order(tensor)
    graph = IterationGraph(mapping)

    A = Tensor("A", ["J", "I"])
    A.set_is_output(True)
    B = Tensor("B", ["K", "I"])
    C = Tensor("C", ["J", "K"])

    assert graph.pop() == ("j", [A, C])
    assert graph.pop() == ("k", [B, C])
    assert graph.pop() == ("i", [A, B])
    assert graph.peek() == (None, [C, A, B])
