import pytest

from es2hfa.ir.iter_graph import IterationGraph
from es2hfa.ir.mapping import Mapping
from es2hfa.parse.einsum import EinsumParser
from es2hfa.parse.input import Input
from es2hfa.trans.equation import Equation
from tests.utils.parse_tree import make_plus


def make_basic():
    yaml = """
    einsum:
        declaration:
            A: []
            B: [I]
            C: [I]
            D: [I]
        expressions:
            - "A[] = sum(I).(B[i] * C[i] * D[i])"
    """
    input_ = Input.from_str(yaml)
    mapping = Mapping(input_)
    mapping.add_einsum(0)

    return IterationGraph(mapping), Equation(mapping)


def make_output():
    yaml = """
    einsum:
        declaration:
            A: [I]
            B: [I]
            C: [I]
            D: [I]
        expressions:
            - "A[I] = B[i] * C[i] * D[i]"
    """
    input_ = Input.from_str(yaml)
    mapping = Mapping(input_)
    mapping.add_einsum(0)

    return IterationGraph(mapping), Equation(mapping)


def make_mult_terms():
    yaml = """
    einsum:
        declaration:
            A: [I]
            B: [I]
            C: [I]
            D: [I]
            E: [I]
            F: [I]
        expressions:
            - "A[I] = B[i] * C[i] + D[i] * E[i] + F[i]"
    """
    input_ = Input.from_str(yaml)
    mapping = Mapping(input_)
    mapping.add_einsum(0)

    return IterationGraph(mapping), Equation(mapping)


def make_other(einsum):
    yaml = """
    einsum:
        declaration:
            A: [I]
            B: [I]
            C: [I]
            D: [I]
        expressions:
            - """ + einsum
    input_ = Input.from_str(yaml)
    mapping = Mapping(input_)
    mapping.add_einsum(0)

    return mapping


def test_mult_tensor_uses():
    mapping = make_other("A[i] = B[i] * B[i]")
    with pytest.raises(ValueError) as excinfo:
        Equation(mapping)

    assert str(excinfo.value) == "B appears multiple times in the einsum"


def test_tensor_in_out():
    mapping = make_other("A[i] = A[i] * B[i]")
    with pytest.raises(ValueError) as excinfo:
        Equation(mapping)

    assert str(excinfo.value) == "A appears multiple times in the einsum"


def test_make_iter_expr_no_tensors():
    _, eqn = make_basic()
    with pytest.raises(ValueError) as excinfo:
        eqn.make_iter_expr([])

    assert str(excinfo.value) == "Must iterate over at least one tensor"


def test_make_iter_expr():
    graph, eqn = make_basic()

    _, tensors = graph.peek()
    iter_expr = "b_i & (c_i & d_i)"

    assert eqn.make_iter_expr(tensors).gen() == iter_expr


def test_make_iter_expr_output():
    graph, eqn = make_output()

    _, tensors = graph.peek()
    iter_expr = "a_i << (b_i & (c_i & d_i))"

    assert eqn.make_iter_expr(tensors).gen() == iter_expr


def test_make_iter_expr_mult_terms():
    graph, eqn = make_mult_terms()

    _, tensors = graph.peek()
    iter_expr = "a_i << ((b_i & c_i) | ((d_i & e_i) | f_i))"

    assert eqn.make_iter_expr(tensors).gen() == iter_expr


def test_make_payload_no_tensors():
    _, eqn = make_basic()
    with pytest.raises(ValueError) as excinfo:
        eqn.make_payload([])

    assert str(
        excinfo.value) == "Must have at least one tensor to make the payload"


def test_make_payload():
    graph, eqn = make_basic()

    _, tensors = graph.pop()
    payload = "(b_val, (c_val, d_val))"

    assert eqn.make_payload(tensors).gen() == payload


def test_make_payload_output():
    graph, eqn = make_output()

    _, tensors = graph.pop()
    payload = "(a_ref, (b_val, (c_val, d_val)))"

    assert eqn.make_payload(tensors).gen() == payload


def test_make_payload_mult_terms():
    graph, eqn = make_mult_terms()

    _, tensors = graph.pop()
    payload = "(a_ref, (_, (b_val, c_val), (_, (d_val, e_val), f_val)))"

    assert eqn.make_payload(tensors).gen() == payload


def test_make_update():
    _, eqn = make_basic()
    stmt = "a_ref += b_val * c_val * d_val"
    assert eqn.make_update().gen(depth=0) == stmt


def test_make_update_vars():
    mapping = make_other("A[i] = b * c * d")
    eqn = Equation(mapping)
    stmt = "a_ref += b * c * d"
    assert eqn.make_update().gen(depth=0) == stmt


def test_make_update_mult_terms():
    mapping = make_other("A[i] = b * B[i] + c * C[i] + d * D[i]")
    eqn = Equation(mapping)
    stmt = "a_ref += b * b_val + c * c_val + d * d_val"
    assert eqn.make_update().gen(depth=0) == stmt
