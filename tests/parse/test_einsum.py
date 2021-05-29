from lark.lexer import Token
from lark.tree import Tree

from es2hfa.parse.einsum import EinsumParser
from tests.utils.parse_tree import *


def test_einsum():
    # Note: also tests rank-0 tensor and variable name parsing
    tree = make_einsum(make_output("A", []), Tree("times", [make_var("b")]))
    assert EinsumParser.parse("A[] = b") == tree


def test_tensor():
    # Note: also tests inds parsing
    tree = make_einsum(
        make_output(
            "A", [
                "i", "j"]), Tree(
            "times", [
                make_tensor(
                    "B", [
                        "i", "j"])]))
    assert EinsumParser.parse("A[i, j] = B[i, j]") == tree


def test_tensor_rank1():
    tree = make_einsum(make_output("A", ["i"]), Tree("times", [make_var("b")]))
    assert EinsumParser.parse("A[i] = b") == tree


def test_sum():
    sum_ = Tree("sum", [make_inds("sinds", ["K", "L"]),
                make_tensor("B", ["k", "l"])])
    tree = make_einsum(make_output("A", []), sum_)
    assert EinsumParser.parse("A[] = sum(K, L).B[k, l]") == tree


def test_plus():
    tree = make_einsum(make_output("A", []), make_plus("a", "b"))
    assert EinsumParser.parse("A[] = a + b") == tree


def test_times():
    tree = make_einsum(make_output("A", []), make_times(["a", "b"]))
    assert EinsumParser.parse("A[] = a * b") == tree
