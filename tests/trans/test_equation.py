import pytest

from es2hfa.ir.iter_graph import IterationGraph
from es2hfa.ir.program import Program
from es2hfa.parse.einsum import Einsum
from es2hfa.parse.mapping import Mapping
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
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)
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
            - "A[i] = B[i] * C[i] * D[i]"
    """
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)
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
            - "A[i] = B[i] * C[i] + D[i] * E[i] + F[i]"
    """
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)
    program.add_einsum(0)

    return IterationGraph(program), Equation(program)


def make_dot():
    yaml = """
    einsum:
        declaration:
            A: [M]
            C: [M]
            Z: [M]
        expressions:
            - Z[m] = dot(A[m], b, C[m], 1)
    """
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)
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
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)
    program.add_einsum(0)

    return program


def make_display(style, opt):
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
        spacetime:
            A:
                space: []
                time: [I.""" + style + """]
                opt: """ + opt
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)
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
        eqn.make_iter_expr(None, [])

    assert str(excinfo.value) == "Must iterate over at least one tensor"


def test_make_iter_expr():
    graph, eqn = make_basic()

    ind, tensors = graph.peek()
    iter_expr = "b_i & (c_i & d_i)"

    assert eqn.make_iter_expr(ind, tensors).gen() == iter_expr


def test_make_iter_expr_output():
    graph, eqn = make_output()

    ind, tensors = graph.peek()
    iter_expr = "a_i << (b_i & (c_i & d_i))"

    assert eqn.make_iter_expr(ind, tensors).gen() == iter_expr


def test_make_iter_expr_mult_terms():
    graph, eqn = make_mult_terms()

    ind, tensors = graph.peek()
    iter_expr = "a_i << ((b_i & c_i) | ((d_i & e_i) | f_i))"

    assert eqn.make_iter_expr(ind, tensors).gen() == iter_expr


def test_make_iter_expr_dot():
    graph, eqn = make_dot()

    ind, tensors = graph.peek()
    iter_expr = "z_m << (a_m & c_m)"

    assert eqn.make_iter_expr(ind, tensors).gen() == iter_expr


def test_make_iter_display_coord():
    graph, eqn = make_display("coord", "")

    ind, tensors = graph.peek()
    iter_expr = "b_i & (c_i & d_i)"

    assert eqn.make_iter_expr(ind, tensors).gen() == iter_expr


def test_make_iter_expr_display_pos():
    graph, eqn = make_display("pos", "")

    ind, tensors = graph.peek()
    iter_expr = "enumerate(b_i & (c_i & d_i))"

    assert eqn.make_iter_expr(ind, tensors).gen() == iter_expr


def test_make_iter_expr_display_slip():
    graph, eqn = make_display("pos", "slip")

    ind, tensors = graph.peek()
    iter_expr = "b_i & (c_i & d_i)"

    assert eqn.make_iter_expr(ind, tensors).gen() == iter_expr


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


def test_make_payload_dot():
    graph, eqn = make_dot()

    ind, tensors = graph.pop()
    payload = "m, (z_ref, (a_val, c_val))"

    assert eqn.make_payload(ind, tensors).gen(False) == payload


def test_make_payload_display_coord():
    graph, eqn = make_display("coord", "")

    ind, tensors = graph.pop()
    payload = "i, (b_val, (c_val, d_val))"

    assert eqn.make_payload(ind, tensors).gen(False) == payload


def test_make_payload_display_pos():
    graph, eqn = make_display("pos", "")

    ind, tensors = graph.pop()
    payload = "i_pos, (i, (b_val, (c_val, d_val)))"

    assert eqn.make_payload(ind, tensors).gen(False) == payload


def test_make_payload_display_slip():
    graph, eqn = make_display("pos", "slip")

    ind, tensors = graph.pop()
    payload = "i, (b_val, (c_val, d_val))"

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


def test_make_update_dot():
    _, eqn = make_dot()
    stmt = "z_ref += b"
    assert eqn.make_update().gen(depth=0) == stmt
