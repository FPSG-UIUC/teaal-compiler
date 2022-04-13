import pytest
from sympy import symbols

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


def make_other(einsum, mapping):
    yaml = """
    einsum:
        declaration:
            A: [I]
            B: [I]
            C: [I]
            D: [I]
        expressions:
            - """ + einsum + """
    mapping: """ + mapping
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


def make_conv(coord_math, loop_order):
    yaml = """
    einsum:
        declaration:
            F: [S]
            I: [W]
            O: [P, Q]
        expressions:
            - O[p, q] = sum(S).(I[""" + coord_math + """] * F[s])
    mapping:
        loop-order:
            O: """ + loop_order

    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)
    program.add_einsum(0)

    return IterationGraph(program), Equation(program)


def test_mult_tensor_uses():
    program = make_other("A[i] = B[i] * B[i]", "")
    with pytest.raises(ValueError) as excinfo:
        Equation(program)

    assert str(excinfo.value) == "B appears multiple times in the einsum"


def test_tensor_in_out():
    program = make_other("A[i] = A[i] * B[i]", "")
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

    rank, tensors = graph.peek()
    iter_expr = "b_i & (c_i & d_i)"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_expr_output():
    graph, eqn = make_output()

    rank, tensors = graph.peek()
    iter_expr = "a_i << (b_i & (c_i & d_i))"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_expr_mult_terms():
    graph, eqn = make_mult_terms()

    rank, tensors = graph.peek()
    iter_expr = "a_i << ((b_i & c_i) | ((d_i & e_i) | f_i))"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_expr_dot():
    graph, eqn = make_dot()

    rank, tensors = graph.peek()
    iter_expr = "z_m << (a_m & c_m)"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_display_coord():
    graph, eqn = make_display("coord", "")

    rank, tensors = graph.peek()
    iter_expr = "b_i & (c_i & d_i)"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_expr_display_pos():
    graph, eqn = make_display("pos", "")

    rank, tensors = graph.peek()
    iter_expr = "enumerate(b_i & (c_i & d_i))"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_expr_display_slip():
    graph, eqn = make_display("pos", "slip")

    rank, tensors = graph.peek()
    iter_expr = "b_i & (c_i & d_i)"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_expr_output_only():
    program = make_other("A[i] = b", "")
    graph = IterationGraph(program)
    eqn = Equation(program)

    rank, tensors = graph.peek()
    iter_expr = "a_i.iterShapeRef()"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_expr_output_only_display():
    mapping = """
        spacetime:
            A:
                space: []
                time: [I]
    """
    program = make_other("A[i] = b", mapping)

    graph = IterationGraph(program)
    eqn = Equation(program)

    rank, tensors = graph.peek()
    iter_expr = "enumerate(a_i.iterShapeRef())"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_expr_conv():
    graph, eqn = make_conv("p + q + s", "[P, S, Q]")
    graph.pop()

    rank, tensors = graph.peek()
    iter_expr = "f_s"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr

    graph.pop()
    rank, tensors = graph.peek()
    iter_expr = "o_q << i_w.project(trans_fn=lambda w: w + -1 * p + -1 * s, interval=(0, Q))"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_expr_conv_frac():
    graph, eqn = make_conv("2 * q + s", "[P, S, Q]")
    graph.pop()

    rank, tensors = graph.peek()
    iter_expr = "f_s"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr

    graph.pop()
    rank, tensors = graph.peek()
    iter_expr = "o_q << i_w.project(trans_fn=lambda w: 1 / 2 * w + -1 / 2 * s, interval=(0, Q)).prune(trans_fn=lambda i, c, p: c % 1 == 0)"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_expr_conv_project_output():
    graph, eqn = make_conv("p + q + s", "[P, S, W]")
    graph.pop()
    graph.pop()

    with pytest.raises(ValueError) as excinfo:
        print(graph.peek())
        eqn.make_iter_expr(*graph.peek())

    assert str(
        excinfo.value) == "Cannot project into the output tensor. Replace W with Q in the loop order"


def test_make_payload_no_tensors():
    _, eqn = make_basic()
    with pytest.raises(ValueError) as excinfo:
        eqn.make_payload(None, [])

    assert str(
        excinfo.value) == "Must have at least one tensor to make the payload"


def test_make_payload():
    graph, eqn = make_basic()

    rank, tensors = graph.pop()
    payload = "i, (b_val, (c_val, d_val))"

    assert eqn.make_payload(rank, tensors).gen(False) == payload


def test_make_payload_output():
    graph, eqn = make_output()

    rank, tensors = graph.pop()
    payload = "i, (a_ref, (b_val, (c_val, d_val)))"

    assert eqn.make_payload(rank, tensors).gen(False) == payload


def test_make_payload_mult_terms():
    graph, eqn = make_mult_terms()

    rank, tensors = graph.pop()
    payload = "i, (a_ref, (_, (b_val, c_val), (_, (d_val, e_val), f_val)))"

    assert eqn.make_payload(rank, tensors).gen(False) == payload


def test_make_payload_dot():
    graph, eqn = make_dot()

    rank, tensors = graph.pop()
    payload = "m, (z_ref, (a_val, c_val))"

    assert eqn.make_payload(rank, tensors).gen(False) == payload


def test_make_payload_display_coord():
    graph, eqn = make_display("coord", "")

    rank, tensors = graph.pop()
    payload = "i, (b_val, (c_val, d_val))"

    assert eqn.make_payload(rank, tensors).gen(False) == payload


def test_make_payload_display_pos():
    graph, eqn = make_display("pos", "")

    rank, tensors = graph.pop()
    payload = "i_pos, (i, (b_val, (c_val, d_val)))"

    assert eqn.make_payload(rank, tensors).gen(False) == payload


def test_make_payload_display_slip():
    graph, eqn = make_display("pos", "slip")

    rank, tensors = graph.pop()
    payload = "i, (b_val, (c_val, d_val))"

    assert eqn.make_payload(rank, tensors).gen(False) == payload


def test_make_payload_output_only():
    program = make_other("A[i] = b", "")
    graph = IterationGraph(program)
    eqn = Equation(program)

    rank, tensors = graph.pop()
    iter_expr = "i, a_ref"

    assert eqn.make_payload(rank, tensors).gen(False) == iter_expr


def test_make_update():
    _, eqn = make_basic()
    stmt = "a_ref += b_val * c_val * d_val"
    assert eqn.make_update().gen(depth=0) == stmt


def test_make_update_vars():
    program = make_other("A[i] = b * c * d", "")
    eqn = Equation(program)
    stmt = "a_ref += b * c * d"
    assert eqn.make_update().gen(depth=0) == stmt


def test_make_update_mult_terms():
    program = make_other("A[i] = b * B[i] + c * C[i] + d * D[i]", "")
    eqn = Equation(program)
    stmt = "a_ref += b * b_val + c * c_val + d * d_val"
    assert eqn.make_update().gen(depth=0) == stmt


def test_make_update_dot():
    _, eqn = make_dot()
    stmt = "z_ref += b"
    assert eqn.make_update().gen(depth=0) == stmt


def test_iter_fiber_not_fiber():
    graph, eqn = make_conv("q + s", "[P, W, Q]")
    graph.pop()
    graph.pop()
    graph.pop()
    _, tensors = graph.peek()

    with pytest.raises(ValueError) as excinfo:
        eqn._Equation__iter_fiber(None, tensors[0])

    assert str(excinfo.value) == "Cannot iterate over payload o_ref"
