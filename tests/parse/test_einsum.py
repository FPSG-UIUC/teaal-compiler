from lark.lexer import Token
from lark.tree import Tree

from es2hfa.parse.einsum import EinsumParser


def make_inds(inds):
    return Tree("inds", [Token("NAME", i) for i in inds])


def make_op(var1, op, var2):
    return Tree(
        "expr", [
            Token(
                "NAME", var1), Tree(
                op, []), Token(
                    "NAME", var2)])


def make_start(lhs, rhs):
    return Tree("start", [lhs, rhs])


def make_tensor(name, inds):
    return Tree("tensor", [Token("NAME", name), make_inds(inds)])


def test_start():
    # Note: also tests rank-0 tensor and variable name parsing
    tree = make_start(make_tensor("A", []), Token("NAME", "b"))
    assert EinsumParser.parse("A[] = b") == tree


def test_tensor():
    # Note: also tests inds parsing
    tree = make_start(
        make_tensor(
            "A", [
                "i", "j"]), make_tensor(
            "B", [
                "i", "j"]))
    assert EinsumParser.parse("A[i, j] = B[i, j]") == tree


def test_parens():
    tree = make_start(
        make_tensor(
            "A", [
                "i", "j"]), make_tensor(
            "B", [
                "i", "j"]))
    assert EinsumParser.parse("A[i, j] = (B[i, j])") == tree


def test_sum():
    sum_ = Tree("sum", [make_inds(["K", "L"]), make_tensor("B", ["k", "l"])])
    tree = make_start(make_tensor("A", []), sum_)
    assert EinsumParser.parse("A[] = sum(K, L).B[k, l]") == tree


def test_plus():
    tree = make_start(make_tensor("A", []), make_op("a", "plus", "b"))
    assert EinsumParser.parse("A[] = a + b") == tree


def test_minus():
    tree = make_start(make_tensor("A", []), make_op("a", "minus", "b"))
    assert EinsumParser.parse("A[] = a - b") == tree


def test_times():
    tree = make_start(make_tensor("A", []), make_op("a", "times", "b"))
    assert EinsumParser.parse("A[] = a * b") == tree
