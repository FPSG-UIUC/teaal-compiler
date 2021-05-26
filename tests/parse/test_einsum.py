from lark.lexer import Token
from lark.tree import Tree

from es2hfa.parse.einsum import EinsumParser
from tests.utils.parse_tree import make_inds, make_op, make_output, make_einsum, make_tensor


def test_einsum():
    # Note: also tests rank-0 tensor and variable name parsing
    tree = make_einsum(make_output("A", []), Token("NAME", "b"))
    assert EinsumParser.parse("A[] = b") == tree


def test_tensor():
    # Note: also tests inds parsing
    tree = make_einsum(
        make_output(
            "A", [
                "i", "j"]), make_tensor(
            "B", [
                "i", "j"]))
    assert EinsumParser.parse("A[i, j] = B[i, j]") == tree


def test_sum():
    sum_ = Tree("sum", [make_inds("sinds", ["K", "L"]),
                make_tensor("B", ["k", "l"])])
    tree = make_einsum(make_output("A", []), sum_)
    print(EinsumParser.parse("A[] = sum(K, L).B[k, l]"))
    print(tree)
    assert EinsumParser.parse("A[] = sum(K, L).B[k, l]") == tree


def test_plus():
    tree = make_einsum(make_output("A", []), make_op("a", "plus", "b"))
    assert EinsumParser.parse("A[] = a + b") == tree


def test_minus():
    tree = make_einsum(make_output("A", []), make_op("a", "minus", "b"))
    assert EinsumParser.parse("A[] = a - b") == tree


def test_times():
    tree = make_einsum(make_output("A", []), make_op("a", "times", "b"))
    assert EinsumParser.parse("A[] = a * b") == tree
