import pytest

from es2hfa.ir.iter_graph import IterationGraph
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.einsum import EinsumParser
from tests.utils.parse_tree import make_output, make_plus, make_tensor


def test_bad_tree():
    tree = make_plus(["a", "b"])
    with pytest.raises(ValueError) as excinfo:
        IterationGraph(tree, None)

    assert str(excinfo.value) == "Input parse tree must be an einsum"


def test_peek_rank0():
    tree = EinsumParser.parse("A[] = b")
    graph = IterationGraph(tree, None)
    tensors = [Tensor(make_output("A", []))]
    assert graph.peek() == (None, tensors)


def test_peek_default():
    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    graph = IterationGraph(tree, None)
    tensors = [
        Tensor(
            make_output(
                "A", [
                    "i", "j"])), Tensor(
            make_tensor(
                "B", [
                    "i", "k"]))]
    assert graph.peek() == ("i", tensors)


def test_peek_order():
    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    graph = IterationGraph(tree, ["j", "k", "i"])
    tensors = [
        Tensor(
            make_output(
                "A", [
                    "j", "i"])), Tensor(
            make_tensor(
                "C", [
                    "j", "k"]))]
    assert graph.peek() == ("j", tensors)


def test_pop_default():
    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    graph = IterationGraph(tree, None)

    A = Tensor(make_output("A", ["i", "j"]))
    B = Tensor(make_tensor("B", ["i", "k"]))
    C = Tensor(make_tensor("C", ["j", "k"]))

    assert graph.pop() == ("i", [A, B])
    assert graph.pop() == ("j", [C, A])
    assert graph.pop() == ("k", [B, C])
    assert graph.peek() == (None, [A, B, C])


def test_pop_order():
    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    graph = IterationGraph(tree, ["j", "k", "i"])

    A = Tensor(make_output("A", ["j", "i"]))
    B = Tensor(make_tensor("B", ["k", "i"]))
    C = Tensor(make_tensor("C", ["j", "k"]))

    assert graph.pop() == ("j", [A, C])
    assert graph.pop() == ("k", [B, C])
    assert graph.pop() == ("i", [A, B])
    assert graph.peek() == (None, [C, A, B])
