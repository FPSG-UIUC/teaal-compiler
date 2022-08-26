from lark.tree import Tree

from teaal.parse.equation import EquationParser
from tests.utils.parse_tree import *


def test_einsum():
    # Note: also tests rank-0 tensor and variable name parsing
    tree = make_einsum(make_output("A", []), Tree("plus", [make_times(["b"])]))
    assert EquationParser.parse("A[] = b") == tree


def test_tensor():
    # Note: also tests ranks parsing
    tree = make_einsum(make_output("A", ["i", "j"]),
                       Tree("plus", [Tree("times",
                                          [make_tensor("B", ["i", "j"])])]))
    assert EquationParser.parse("A[i, j] = B[i, j]") == tree


def test_tensor_rank1():
    tree = make_einsum(make_output("A", ["i"]), make_plus(["b"]))
    assert EquationParser.parse("A[i] = b") == tree


def test_sum_factor():
    sum_ = make_sum(["K", "L"], make_tensor("B", ["k", "l"]))
    tree = make_einsum(make_output("A", []), sum_)
    assert EquationParser.parse("A[] = sum(K, L).B[k, l]") == tree


def test_plus():
    tree = make_einsum(make_output("A", []), make_plus(["a", "b"]))
    assert EquationParser.parse("A[] = a + b") == tree


def test_sum_expr():
    sum_ = Tree("sum", [make_sranks(["K", "L"]), make_plus(["a", "b"])])
    tree = make_einsum(make_output("A", []), sum_)
    assert EquationParser.parse("A[] = sum(K, L).(a + b)") == tree


def test_times():
    tree = make_einsum(make_output("A", []), Tree(
        "plus", [make_times(["a", "b"])]))
    assert EquationParser.parse("A[] = a * b") == tree


def test_dot():
    tree = make_einsum(make_output("T1", ["k", "m", "n"]), Tree("plus", [
                       make_dot([make_tensor("A", ["k", "m"]),
                                 make_tensor("B", ["k", "n"])], 1)]))
    assert EquationParser.parse("T1[k, m, n] = dot(A[k, m], B[k, n], 1)")


def test_ind_plus():
    tree = make_einsum(make_output("Z", ["m"]), make_tensor_tranks(
        "A", Tree("tranks", [make_iplus(["m", "s"])])))
    assert EquationParser.parse("Z[m] = A[m + s]")


def test_ind_times():
    tree = make_einsum(make_output("Z", ["m"]), make_tensor_tranks(
        "A", Tree("tranks", [make_itimes(["2", "m"])])))
    assert EquationParser.parse("Z[m] = A[2 * m]")
