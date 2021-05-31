from es2hfa.parse.tensor import TensorParser
from tests.utils.parse_tree import make_tensor


def test_no_args():
    tree = make_tensor("A", [])
    assert TensorParser.parse("A[]") == tree


def test_one_arg():
    tree = make_tensor("A", ["i"])
    assert TensorParser.parse("A[i]") == tree


def test_many_args():
    tree = make_tensor("A", ["i", "j", "k"])
    assert TensorParser.parse("A[i, j, k]") == tree
