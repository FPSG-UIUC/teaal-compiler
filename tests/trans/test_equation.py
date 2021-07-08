import pytest

from es2hfa.ir.iter_graph import IterationGraph
from es2hfa.ir.program import Program
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
    program = Program(input_)
    program.add_einsum(0)

    return IterationGraph(program), Equation(program)


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
    program = Program(input_)
    program.add_einsum(0)

    return IterationGraph(program), Equation(program)


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
    program = Program(input_)
    program.add_einsum(0)

    return IterationGraph(program), Equation(program)


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
    program = Program(input_)
    program.add_einsum(0)

    return program


def make_display(style):
    yaml = """
    einsum:
        declaration:
            A: []
            B: [I]
            C: [I]
            D: [I]
        expressions:
            - "A[] = sum(I).(B[i] * C[i] * D[i])"
    mapping:
        display:
            A:
                space: []
                time: [I]
                style: """ + style
    input_ = Input.from_str(yaml)
    program = Program(input_)
    program.add_einsum(0)

    return IterationGraph(program), Equation(program)


def test_mult_tensor_uses():
    program = make_other("A[i] = B[i] * B[i]")
    with pytest.raises(ValueError) as excinfo:
        Equation(program)

    assert str(excinfo.value) == "B appears multiple times in the einsum"


def test_tensor_in_out():
    program = make_other("A[i] = A[i] * B[i]")
    with pytest.raises(ValueError) as excinfo:
        Equation(program)

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


def test_make_iter_display_shape():
    graph, eqn = make_display("shape")

    _, tensors = graph.peek()
    iter_expr = "b_i & (c_i & d_i)"

    assert eqn.make_iter_expr(tensors).gen() == iter_expr


def test_make_iter_expr_display_occupancy():
    graph, eqn = make_display("occupancy")

    _, tensors = graph.peek()
    iter_expr = "enumerate(b_i & (c_i & d_i))"

    assert eqn.make_iter_expr(tensors).gen() == iter_expr


def test_make_payload_no_tensors():
    _, eqn = make_basic()
    with pytest.raises(ValueError) as excinfo:
        eqn.make_payload(None, [])

    assert str(
        excinfo.value) == "Must have at least one tensor to make the payload"


def test_make_payload():
    graph, eqn = make_basic()

    ind, tensors = graph.pop()
    payload = "i, (b_val, (c_val, d_val))"

    assert eqn.make_payload(ind, tensors).gen(False) == payload


def test_make_payload_output():
    graph, eqn = make_output()

    ind, tensors = graph.pop()
    payload = "i, (a_ref, (b_val, (c_val, d_val)))"

    assert eqn.make_payload(ind, tensors).gen(False) == payload


def test_make_payload_mult_terms():
    graph, eqn = make_mult_terms()

    ind, tensors = graph.pop()
    payload = "i, (a_ref, (_, (b_val, c_val), (_, (d_val, e_val), f_val)))"

    assert eqn.make_payload(ind, tensors).gen(False) == payload


def test_make_payload_display_shape():
    graph, eqn = make_display("shape")

    ind, tensors = graph.pop()
    payload = "i, (b_val, (c_val, d_val))"

    assert eqn.make_payload(ind, tensors).gen(False) == payload


def test_make_payload_display_occupancy():
    graph, eqn = make_display("occupancy")

    ind, tensors = graph.pop()
    payload = "i_pos, (i, (b_val, (c_val, d_val)))"

    assert eqn.make_payload(ind, tensors).gen(False) == payload


def test_make_update():
    _, eqn = make_basic()
    stmt = "a_ref += b_val * c_val * d_val"
    assert eqn.make_update().gen(depth=0) == stmt


def test_make_update_vars():
    program = make_other("A[i] = b * c * d")
    eqn = Equation(program)
    stmt = "a_ref += b * c * d"
    assert eqn.make_update().gen(depth=0) == stmt


def test_make_update_mult_terms():
    program = make_other("A[i] = b * B[i] + c * C[i] + d * D[i]")
    eqn = Equation(program)
    stmt = "a_ref += b * b_val + c * c_val + d * d_val"
    assert eqn.make_update().gen(depth=0) == stmt
