import pytest

from es2hfa.ir.iter_graph import IterationGraph
from es2hfa.ir.mapping import Mapping
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.einsum import EinsumParser
from es2hfa.parse.tensor import TensorParser


def test_peek_rank0():
    tensors = [TensorParser.parse("A[]")]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[] = b")
    mapping.add_einsum(tree, {})
    graph = IterationGraph(mapping)

    tensor = Tensor(TensorParser.parse("A[]"))
    tensor.set_is_output(True)

    assert graph.peek() == (None, [tensor])


def test_peek_default():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, {})
    graph = IterationGraph(mapping)

    results = ["A[I, J]", "B[I, K]"]
    results = [Tensor(TensorParser.parse(tensor)) for tensor in results]
    results[0].set_is_output(True)

    assert graph.peek() == ("i", results)


def test_peek_order():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, {"A": ["J", "K", "I"]})
    for tensor in mapping.get_tensors():
        mapping.apply_loop_order(tensor)
    graph = IterationGraph(mapping)

    results = ["A[J, I]", "C[J, K]"]
    results = [Tensor(TensorParser.parse(tensor)) for tensor in results]
    results[0].set_is_output(True)

    assert graph.peek() == ("j", results)


def test_pop_default():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, {})
    graph = IterationGraph(mapping)

    A = Tensor(TensorParser.parse("A[I, J]"))
    A.set_is_output(True)
    B = Tensor(TensorParser.parse("B[I, K]"))
    C = Tensor(TensorParser.parse("C[J, K]"))

    assert graph.pop() == ("i", [A, B])
    assert graph.pop() == ("j", [C, A])
    assert graph.pop() == ("k", [B, C])
    assert graph.peek() == (None, [A, B, C])


def test_pop_order():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, {"A": ["J", "K", "I"]})
    for tensor in mapping.get_tensors():
        mapping.apply_loop_order(tensor)
    graph = IterationGraph(mapping)

    A = Tensor(TensorParser.parse("A[J, I]"))
    A.set_is_output(True)
    B = Tensor(TensorParser.parse("B[K, I]"))
    C = Tensor(TensorParser.parse("C[J, K]"))

    assert graph.pop() == ("j", [A, C])
    assert graph.pop() == ("k", [B, C])
    assert graph.pop() == ("i", [A, B])
    assert graph.peek() == (None, [C, A, B])
