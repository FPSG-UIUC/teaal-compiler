import pytest

from es2hfa.ir.loop.iter_graph import IterationGraph
from es2hfa.ir.loop.tensor import Tensor
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

    tensors1 = [
        Tensor(
            make_output(
                "A", ["j"])), Tensor(
            make_tensor(
                "B", ["k"]))]
    assert graph.pop() == ("i", tensors1)

    tensors2 = [Tensor(make_tensor("C", ["k"])), Tensor(make_output("A", []))]
    assert graph.pop() == ("j", tensors2)

    tensors3 = [Tensor(make_tensor("B", [])), Tensor(make_tensor("C", []))]
    assert graph.pop() == ("k", tensors3)

    tensors4 = [
        Tensor(
            make_output(
                "A", [])), Tensor(
            make_tensor(
                "B", [])), Tensor(
            make_tensor(
                "C", []))]
    assert graph.peek() == (None, tensors4)


def test_pop_order():
    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    graph = IterationGraph(tree, ["j", "k", "i"])

    tensors1 = [
        Tensor(
            make_output(
                "A", ["i"])), Tensor(
            make_tensor(
                "C", ["k"]))]
    assert graph.pop() == ("j", tensors1)

    tensors2 = [Tensor(make_tensor("B", ["i"])), Tensor(make_tensor("C", []))]
    assert graph.pop() == ("k", tensors2)

    tensors3 = [Tensor(make_output("A", [])), Tensor(make_tensor("B", []))]
    assert graph.pop() == ("i", tensors3)

    tensors4 = [
        Tensor(
            make_tensor(
                "C", [])), Tensor(
            make_output(
                "A", [])), Tensor(
            make_tensor(
                "B", []))]
    assert graph.peek() == (None, tensors4)
