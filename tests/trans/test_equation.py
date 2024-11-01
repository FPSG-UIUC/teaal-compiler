import pytest
from sympy import symbols

from teaal.ir.hardware import Hardware
from teaal.ir.iter_graph import IterationGraph
from teaal.ir.metrics import Metrics
from teaal.ir.program import Program
from teaal.parse import *
from teaal.trans.equation import Equation
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
            - "A[] = B[i] * C[i] * D[i]"
    """
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)
    program.add_einsum(0)

    return IterationGraph(program), Equation(program, None)


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

    return IterationGraph(program), Equation(program, None)


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

    return IterationGraph(program), Equation(program, None)


def make_take():
    yaml = """
    einsum:
        declaration:
            A: [M]
            C: [M]
            Z: [M]
        expressions:
            - Z[m] = take(A[m], b, C[m], 1)
    """
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)
    program.add_einsum(0)

    return IterationGraph(program), Equation(program, None)


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

    for tensor in program.get_equation().get_tensors():
        program.apply_all_partitioning(tensor)
        program.get_loop_order().apply(tensor)

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
            - "A[] = B[i] * C[i] * D[i]"
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

    return IterationGraph(program), Equation(program, None)


def make_matmul(mapping):
    yaml = """
    einsum:
        declaration:
            A: [K, M]
            B: [K, N]
            Z: [M, N]
        expressions:
            - Z[m, n] = A[k, m] * B[k, n]
    mapping:
    """ + mapping
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)

    program = Program(einsum, mapping)
    program.add_einsum(0)

    part_ir = program.get_partitioning()
    for tensor in program.get_equation().get_tensors():
        program.apply_all_partitioning(tensor)
        program.get_loop_order().apply(tensor)

    return IterationGraph(program), Equation(program, None)


def make_conv(expr, loop_order):
    yaml = """
    einsum:
        declaration:
            F: [S]
            I: [W]
            J: [W]
            O: [P, Q]
        expressions:
            - """ + expr + """
    mapping:
        loop-order:
            O: """ + loop_order

    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)
    program.add_einsum(0)

    for tensor in program.get_equation().get_tensors():
        program.apply_all_partitioning(tensor)
        program.get_loop_order().apply(tensor)

    return IterationGraph(program), Equation(program, None)


def make_conv_part(expr, loop_order):
    yaml = """
    einsum:
        declaration:
            F: [S]
            I: [W]
            J: [W]
            O: [P, Q]
        expressions:
            - """ + expr + """
    mapping:
        partitioning:
            O:
                Q: [uniform_shape(10)]
                W: [follow(Q)]
        loop-order:
            O: """ + loop_order

    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)
    program.add_einsum(0)

    for tensor in program.get_equation().get_tensors():
        program.apply_all_partitioning(tensor)
        program.get_loop_order().apply(tensor)

    return IterationGraph(program), Equation(program, None)


def make_gamma():
    fname = "tests/integration/gamma.yaml"
    einsum = Einsum.from_file(fname)
    mapping = Mapping.from_file(fname)
    arch = Architecture.from_file(fname)
    bindings = Bindings.from_file(fname)
    format_ = Format.from_file(fname)

    program = Program(einsum, mapping)
    hardware = Hardware(arch, bindings, program)

    program.add_einsum(0)
    metrics = Metrics(program, hardware, format_)

    return IterationGraph(program), Equation(program, metrics)


def test_eager_inputs_one_fiber():
    expr = "O[p, q] = I[q + s] * F[s]"
    _, eqn = make_conv_part(expr, "[Q1, P, S, Q0]")
    hifiber = "inputs_q1 = Fiber.fromLazy(i_w1.project(trans_fn=lambda w1: w1))"

    assert eqn.make_eager_inputs("Q1", ["I"]).gen(0) == hifiber


def test_eager_inputs_multiple_fibers():
    expr = "O[p, q] = I[q + s] * J[q + s] * F[s]"
    mapping = """[Q1, P, S, Q]
        partitioning:
            O:
                Q: [uniform_shape(10)]
                W: [follow(Q)]
    """
    _, eqn = make_conv(expr, mapping)
    hifiber = "inputs_q1 = Fiber.fromLazy(i_w1.project(trans_fn=lambda w1: w1) & j_w1.project(trans_fn=lambda w1: w1))"

    assert eqn.make_eager_inputs("Q1", ["I", "J"]).gen(0) == hifiber


def test_make_interval_bad_rank():
    expr = "O[p, q] = I[q + s] * F[s]"
    _, eqn = make_conv_part(expr, "[Q1, P, S, Q0]")
    with pytest.raises(ValueError) as excinfo:
        eqn.make_interval("S")

    assert str(excinfo.value) == "Interval not necessary for rank S"


def test_make_interval():
    expr = "O[p, q] = I[q + s] * F[s]"
    _, eqn = make_conv_part(expr, "[Q1, P, S, Q0]")

    hifiber = "if q1_pos == 0:\n" + \
        "    q0_start = 0\n" + \
        "else:\n" + \
        "    q0_start = q1\n" + \
        "if q1_pos + 1 < len(inputs_q1):\n" + \
        "    q0_end = inputs_q1.getCoords()[q1_pos + 1]\n" + \
        "else:\n" + \
        "    q0_end = Q"

    assert eqn.make_interval("Q0").gen(0) == hifiber


def test_make_iter_expr_no_tensors():
    _, eqn = make_basic()
    with pytest.raises(ValueError) as excinfo:
        eqn.make_iter_expr(None, [])

    assert str(excinfo.value) == "Must iterate over at least one tensor"


def test_make_iter_expr():
    graph, eqn = make_basic()

    rank, tensors = graph.peek_concord()
    iter_expr = "b_i & (c_i & d_i)"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_expr_output():
    graph, eqn = make_output()

    rank, tensors = graph.peek_concord()
    iter_expr = "a_i << (b_i & (c_i & d_i))"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_expr_mult_terms():
    graph, eqn = make_mult_terms()

    rank, tensors = graph.peek_concord()
    iter_expr = "a_i << ((b_i & c_i) | ((d_i & e_i) | f_i))"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_expr_take():
    graph, eqn = make_take()

    rank, tensors = graph.peek_concord()
    iter_expr = "z_m << (a_m & c_m)"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_display_coord():
    graph, eqn = make_display("coord", "")

    rank, tensors = graph.peek_concord()
    iter_expr = "b_i & (c_i & d_i)"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_expr_display_pos():
    graph, eqn = make_display("pos", "")

    rank, tensors = graph.peek_concord()
    iter_expr = "enumerate(b_i & (c_i & d_i))"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_expr_display_slip():
    graph, eqn = make_display("pos", "slip")

    rank, tensors = graph.peek_concord()
    iter_expr = "b_i & (c_i & d_i)"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_expr_leader_follower():
    graph, eqn = make_gamma()

    graph.pop_concord()
    iter_expr = "t_k << Fiber.intersection(a_k, b_k, style=\"leader-follower\")"

    assert eqn.make_iter_expr(*graph.peek_concord()).gen() == iter_expr


def test_flattened_output_only_bad():
    mapping = """
        partitioning:
            Z:
                (M, N): [flatten()]
        loop-order:
            Z: [MN, K]
    """
    graph, eqn = make_matmul(mapping)

    with pytest.raises(ValueError) as excinfo:
        eqn.make_iter_expr(*graph.peek_concord())

    assert str(
        excinfo.value) == "Illegal dataflow: cannot iterate over output-only flattened rank MN"


def test_make_iter_expr_output_only():
    program = make_other("A[i] = b", "")
    graph = IterationGraph(program)
    eqn = Equation(program, None)

    rank, tensors = graph.peek_concord()
    iter_expr = "a_i.iterRangeShapeRef(0, I, 1)"

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
    eqn = Equation(program, None)

    rank, tensors = graph.peek_concord()
    iter_expr = "enumerate(a_i.iterRangeShapeRef(0, I, 1))"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_expr_output_only_partition():
    mapping = """
        partitioning:
            A:
                I: [uniform_shape(10), uniform_shape(5)]
    """
    program = make_other("A[i] = b", mapping)

    graph = IterationGraph(program)
    eqn = Equation(program, None)

    rank, tensors = graph.peek_concord()
    iter_expr = "a_i2.iterRangeShapeRef(0, I, I1)"
    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr

    graph.pop_concord()
    rank, tensors = graph.peek_concord()
    iter_expr = "a_i1.iterRangeShapeRef(int(i2 + 1) - 1, int(min(i2 + I1, I) + 1) - 1, I0)"
    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr

    graph.pop_concord()
    rank, tensors = graph.peek_concord()
    iter_expr = "a_i0.iterRangeShapeRef(int(i1 + 1) - 1, int(min(i1 + I0, I) + 1) - 1, 1)"
    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_expr_conv():
    expr = "O[p, q] = I[p + q + s] * F[s]"
    graph, eqn = make_conv(expr, "[P, S, Q]")
    graph.pop_concord()

    rank, tensors = graph.peek_concord()
    iter_expr = "f_s"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr

    graph.pop_concord()
    rank, tensors = graph.peek_concord()
    iter_expr = "o_q << i_w.project(trans_fn=lambda w: w + -1 * p + -1 * s, interval=(0, Q))"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_expr_conv_frac():
    expr = "O[p, q] = I[2 * q + s] * F[s]"
    graph, eqn = make_conv(expr, "[P, S, Q]")
    graph.pop_concord()

    rank, tensors = graph.peek_concord()
    iter_expr = "f_s"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr

    graph.pop_concord()
    rank, tensors = graph.peek_concord()
    iter_expr = "o_q << i_w.project(trans_fn=lambda w: 1 / 2 * w + -1 / 2 * s, interval=(0, Q)).prune(trans_fn=lambda i, c, p: c % 1 == 0)"

    assert eqn.make_iter_expr(rank, tensors).gen() == iter_expr


def test_make_iter_expr_conv_project_output():
    expr = "O[p, q] = I[p + q + s] * F[s]"
    graph, eqn = make_conv(expr, "[P, S, W]")
    graph.pop_concord()
    graph.pop_concord()

    with pytest.raises(ValueError) as excinfo:
        eqn.make_iter_expr(*graph.peek_concord())

    assert str(
        excinfo.value) == "Cannot project into the output tensor. Replace W with Q in the loop order"


def test_make_iter_expr_conv_enum():
    expr = "O[p, q] = I[2 * q + s] * F[s]"
    graph, eqn = make_conv_part(expr, "[Q1, P, S, Q0]")
    hifiber = "enumerate(o_q1 << i_w1.project(trans_fn=lambda w1: 1 / 2 * w1))"

    assert eqn.make_iter_expr(*graph.peek_concord()).gen() == hifiber


def test_make_iter_expr_conv_part():
    expr = "O[p, q] = I[q + s] * F[s]"
    graph, eqn = make_conv_part(expr, "[Q1, P, W0, Q0]")

    graph.pop_concord()
    graph.pop_concord()
    graph.pop_concord()

    hifiber = "o_q0 << f_s.project(trans_fn=lambda s: w0 + -1 * s, interval=(q0_start, q0_end))"

    assert eqn.make_iter_expr(*graph.peek_concord()).gen() == hifiber


def test_make_iter_expr_metrics():
    graph, eqn = make_gamma()
    hifiber = "t_m << a_m"

    assert eqn.make_iter_expr(*graph.peek_concord()).gen() == hifiber


def test_make_payload_no_tensors():
    _, eqn = make_basic()
    with pytest.raises(ValueError) as excinfo:
        eqn.make_payload(None, [])

    assert str(
        excinfo.value) == "Must have at least one tensor to make the payload"


def test_make_payload():
    graph, eqn = make_basic()

    rank, tensors = graph.pop_concord()
    payload = "i, (b_val, (c_val, d_val))"

    assert eqn.make_payload(rank, tensors).gen(False) == payload


def test_make_payload_output():
    graph, eqn = make_output()

    rank, tensors = graph.pop_concord()
    payload = "i, (a_ref, (b_val, (c_val, d_val)))"

    assert eqn.make_payload(rank, tensors).gen(False) == payload


def test_make_payload_mult_terms():
    graph, eqn = make_mult_terms()

    rank, tensors = graph.pop_concord()
    payload = "i, (a_ref, (_, (b_val, c_val), (_, (d_val, e_val), f_val)))"

    assert eqn.make_payload(rank, tensors).gen(False) == payload


def test_make_payload_take():
    graph, eqn = make_take()

    rank, tensors = graph.pop_concord()
    payload = "m, (z_ref, (a_val, c_val))"

    assert eqn.make_payload(rank, tensors).gen(False) == payload


def test_make_payload_display_coord():
    graph, eqn = make_display("coord", "")

    rank, tensors = graph.pop_concord()
    payload = "i, (b_val, (c_val, d_val))"

    assert eqn.make_payload(rank, tensors).gen(False) == payload


def test_make_payload_display_pos():
    graph, eqn = make_display("pos", "")

    rank, tensors = graph.pop_concord()
    payload = "i_pos, (i, (b_val, (c_val, d_val)))"

    assert eqn.make_payload(rank, tensors).gen(False) == payload


def test_make_payload_display_slip():
    graph, eqn = make_display("pos", "slip")

    rank, tensors = graph.pop_concord()
    payload = "i, (b_val, (c_val, d_val))"

    assert eqn.make_payload(rank, tensors).gen(False) == payload


def test_make_payload_output_only():
    program = make_other("A[i] = b", "")
    graph = IterationGraph(program)
    eqn = Equation(program, None)

    rank, tensors = graph.pop_concord()
    iter_expr = "i, a_ref"

    assert eqn.make_payload(rank, tensors).gen(False) == iter_expr


def test_make_payload_flattened():
    mapping = """
        partitioning:
            Z:
                K: [uniform_shape(4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
        loop-order:
            Z: [K1, MK01, N, MK00]
    """
    graph, eqn = make_matmul(mapping)

    assert eqn.make_payload(
        *graph.pop_concord()).gen(parens=False) == "k1, (a_mk01, b_n)"
    assert eqn.make_payload(
        *graph.pop_concord()).gen(parens=False) == "mk01, a_mk00"
    assert eqn.make_payload(
        * graph.pop_concord()).gen(parens=False) == "n, (z_m, b_k0)"
    assert eqn.make_payload(
        * graph.pop_concord()).gen(parens=False) == "(m, k0), a_val"


def test_make_payload_conv_enum():
    expr = "O[p, q] = I[q + s] * F[s]"
    graph, eqn = make_conv_part(expr, "[Q1, P, S, Q0]")
    hifiber = "q1_pos, (q1, (o_p, i_w0))"

    assert eqn.make_payload(*graph.pop_concord()).gen(parens=False) == hifiber


def test_make_payload_metrics():
    graph, eqn = make_gamma()
    hifiber = "m, (t_k, a_k)"

    assert eqn.make_payload(*graph.pop_concord()).gen(parens=False) == hifiber


def test_make_update():
    _, eqn = make_basic()
    stmt = "a_ref += b_val * c_val * d_val"
    assert eqn.make_update().gen(depth=0) == stmt


def test_make_update_vars():
    program = make_other("A[i] = b * c * d", "")
    eqn = Equation(program, None)
    stmt = "a_ref <<= b * c * d"
    assert eqn.make_update().gen(depth=0) == stmt


def test_make_update_mult_terms():
    program = make_other("A[i] = b * B[i] + c * C[i] + d * D[i]", "")
    eqn = Equation(program, None)
    stmt = "a_ref <<= b * b_val + c * c_val + d * d_val"
    assert eqn.make_update().gen(depth=0) == stmt


def test_make_update_take():
    _, eqn = make_take()
    stmt = "z_ref <<= b"
    assert eqn.make_update().gen(depth=0) == stmt
