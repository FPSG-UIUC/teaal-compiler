import pytest

from es2hfa.ir.iter_graph import IterationGraph
from es2hfa.parse.einsum import EinsumParser
from es2hfa.trans.equation import Equation
from tests.utils.parse_tree import make_plus


def test_bad_tree():
    tree = make_plus(["a", "b"])
    with pytest.raises(ValueError) as excinfo:
        Equation(tree)

    assert str(excinfo.value) == "Input parse tree must be an einsum"


def test_mult_tensor_uses():
    tree = EinsumParser.parse("A[i] = B[i] * B[i]")
    with pytest.raises(ValueError) as excinfo:
        Equation(tree)

    assert str(excinfo.value) == "B appears multiple times in the einsum"


def test_tensor_in_out():
    tree = EinsumParser.parse("A[i] = A[i] * B[i]")
    with pytest.raises(ValueError) as excinfo:
        Equation(tree)

    assert str(excinfo.value) == "A appears multiple times in the einsum"


def test_make_iter_expr_no_tensors():
    tree = EinsumParser.parse("A[i] = B[i] * C[i]")
    eqn = Equation(tree)
    with pytest.raises(ValueError) as excinfo:
        eqn.make_iter_expr([])

    assert str(excinfo.value) == "Must iterate over at least one tensor"


def test_make_iter_expr():
    tree = EinsumParser.parse("A[] = sum(I).(B[i] * C[i] * D[i])")
    graph = IterationGraph(tree, None)
    _, tensors = graph.peek()
    eqn = Equation(tree)
    iter_expr = "b_i & (c_i & d_i)"
    assert eqn.make_iter_expr(tensors).gen() == iter_expr


def test_make_iter_expr_output():
    tree = EinsumParser.parse("A[i] = B[i] * C[i] * D[i]")
    graph = IterationGraph(tree, None)
    _, tensors = graph.peek()
    eqn = Equation(tree)
    iter_expr = "a_i << (b_i & (c_i & d_i))"
    assert eqn.make_iter_expr(tensors).gen() == iter_expr


def test_make_iter_expr_mult_terms():
    tree = EinsumParser.parse("A[i] = B[i] * C[i] + D[i] * E[i] + F[i]")
    graph = IterationGraph(tree, None)
    _, tensors = graph.peek()
    eqn = Equation(tree)
    iter_expr = "a_i << ((b_i & c_i) | ((d_i & e_i) | f_i))"
    assert eqn.make_iter_expr(tensors).gen() == iter_expr


def test_make_payload_no_tensors():
    tree = EinsumParser.parse("A[i] = B[i] * C[i]")
    eqn = Equation(tree)
    with pytest.raises(ValueError) as excinfo:
        eqn.make_payload([])

    assert str(
        excinfo.value) == "Must have at least one tensor to make the payload"


def test_make_payload():
    tree = EinsumParser.parse("A[] = sum(I).(B[i] * C[i] * D[i])")
    graph = IterationGraph(tree, None)
    _, tensors = graph.pop()
    eqn = Equation(tree)
    payload = "(b_val, (c_val, d_val))"
    assert eqn.make_payload(tensors).gen() == payload


def test_make_payload_output():
    tree = EinsumParser.parse("A[i] = B[i] * C[i] * D[i]")
    graph = IterationGraph(tree, None)
    _, tensors = graph.pop()
    eqn = Equation(tree)
    payload = "(a_ref, (b_val, (c_val, d_val)))"
    assert eqn.make_payload(tensors).gen() == payload


def test_make_payload_mult_terms():
    tree = EinsumParser.parse("A[i] = B[i] * C[i] + D[i] * E[i] + F[i]")
    graph = IterationGraph(tree, None)
    _, tensors = graph.pop()
    eqn = Equation(tree)
    payload = "(a_ref, (_, (b_val, c_val), (_, (d_val, e_val), f_val)))"
    assert eqn.make_payload(tensors).gen() == payload


def test_make_update():
    tree = EinsumParser.parse("A[] = sum(I).(B[i] * C[i] * D[i])")
    eqn = Equation(tree)
    stmt = "a_ref += b_val * c_val * d_val"
    assert eqn.make_update().gen(depth=0) == stmt


def test_make_update_vars():
    tree = EinsumParser.parse("A[] = b * c * d")
    eqn = Equation(tree)
    stmt = "a_ref += b * c * d"
    assert eqn.make_update().gen(depth=0) == stmt


def test_make_update_mult_terms():
    tree = EinsumParser.parse("A[i] = b * B[i] + c * C[i] + d * D[i]")
    eqn = Equation(tree)
    stmt = "a_ref += b * b_val + c * c_val + d * d_val"
    assert eqn.make_update().gen(depth=0) == stmt
