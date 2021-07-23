import pytest

from es2hfa.ir.program import Program
from es2hfa.parse.einsum import Einsum
from es2hfa.parse.mapping import Mapping
from es2hfa.trans.canvas import Canvas


def create_default():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    """
    return Program(Einsum.from_str(yaml), Mapping.from_str(yaml))


def create_spacetime():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
        loop-order:
            Z: [K, M, N]
        spacetime:
            Z:
                space: [N]
                time: [K.pos, M.coord]
    """
    return Program(Einsum.from_str(yaml), Mapping.from_str(yaml))


def create_partitioned(style):
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
        partitioning:
            Z:
                N: [uniform_shape(6), uniform_shape(3)]
        loop-order:
            Z: [N2, K, N1, M, N0]
        spacetime:
            Z:
                space: [N2.""" + style + """, N1.""" + style + """]
                time: [K.""" + style + """, M.""" + style + """, N0.""" + style + """]"""
    return Program(Einsum.from_str(yaml), Mapping.from_str(yaml))


def create_slip():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
        loop-order:
            Z: [K, M, N]
        spacetime:
            Z:
                space: [N]
                time: [K.pos, M.coord]
                opt: slip
    """
    return Program(Einsum.from_str(yaml), Mapping.from_str(yaml))


def test_create_canvas():
    program = create_spacetime()
    program.add_einsum(0)
    canvas = Canvas(program)

    hfa = "canvas = createCanvas(A_KM, B_KN, Z_MN)"
    assert canvas.create_canvas().gen(0) == hfa


def test_create_canvas_partitioned():
    program = create_partitioned("coord")
    program.add_einsum(0)

    static_parts = program.get_partitioning().get_static_parts()
    for ind in static_parts:
        program.start_partitioning(ind)

    for tensor in program.get_tensors():
        for ind in static_parts:
            program.apply_partitioning(tensor, ind)
        program.apply_curr_loop_order(tensor)

    canvas = Canvas(program)

    hfa = "canvas = createCanvas(A_KM, B_N2KN1N0, Z_N2N1MN0)"
    assert canvas.create_canvas().gen(0) == hfa


def test_add_activity_no_canvas():
    program = create_default()
    program.add_einsum(0)
    canvas = Canvas(program)

    with pytest.raises(ValueError) as excinfo:
        canvas.add_activity()

    assert str(
        excinfo.value) == "Unconfigured canvas. Make sure to first call create_canvas()"


def test_add_activity_no_spacetime():
    program = create_default()
    program.add_einsum(0)

    canvas = Canvas(program)
    canvas.create_canvas()

    with pytest.raises(ValueError) as excinfo:
        canvas.add_activity()

    assert str(excinfo.value) == "SpaceTime information unspecified"


def test_add_activity():
    program = create_spacetime()
    program.add_einsum(0)

    canvas = Canvas(program)
    canvas.create_canvas()

    hfa = "canvas.addActivity((k, m), (k, n), (m, n), spacetime=((n_pos,), (k_pos, m)))"
    assert canvas.add_activity().gen(0) == hfa


def test_add_activity_partitioned_coord():
    program = create_partitioned("coord")
    program.add_einsum(0)

    for ind in program.get_partitioning().get_static_parts():
        program.start_partitioning(ind)

    for tensor in program.get_tensors():
        program.apply_all_partitioning(tensor)
        program.apply_curr_loop_order(tensor)

    canvas = Canvas(program)
    canvas.create_canvas()

    hfa = "canvas.addActivity((k, m), (n2, k, n1, n0), (n2, n1, m, n0), spacetime=((n2, n1 - n2), (k, m, n0 - n1)))"
    assert canvas.add_activity().gen(0) == hfa


def test_add_activity_partitioned_pos():
    program = create_partitioned("pos")
    program.add_einsum(0)

    for ind in program.get_partitioning().get_static_parts():
        program.start_partitioning(ind)

    for tensor in program.get_tensors():
        program.apply_all_partitioning(tensor)
        program.apply_curr_loop_order(tensor)

    canvas = Canvas(program)
    canvas.create_canvas()

    hfa = "canvas.addActivity((k, m), (n2, k, n1, n0), (n2, n1, m, n0), spacetime=((n2_pos, n1_pos), (k_pos, m_pos, n0_pos)))"
    assert canvas.add_activity().gen(0) == hfa


def test_add_activity_slip():
    program = create_slip()
    program.add_einsum(0)

    canvas = Canvas(program)
    canvas.create_canvas()

    hfa = "canvas.addActivity((k, m), (k, n), (m, n), spacetime=((n_pos,), (timestamps[(n_pos,)] - 1,)))"
    assert canvas.add_activity().gen(0) == hfa


def test_display_canvas_no_canvas():
    program = create_default()
    program.add_einsum(0)
    canvas = Canvas(program)

    with pytest.raises(ValueError) as excinfo:
        canvas.display_canvas()

    assert str(
        excinfo.value) == "Unconfigured canvas. Make sure to first call create_canvas()"


def test_display_canvas():
    program = create_spacetime()
    program.add_einsum(0)

    canvas = Canvas(program)
    canvas.create_canvas()

    hfa = "displayCanvas(canvas)"
    assert canvas.display_canvas().gen(0) == hfa


def test_get_space_tuple():
    program = create_spacetime()
    program.add_einsum(0)
    canvas = Canvas(program)

    hfa = "(n_pos,)"
    assert canvas.get_space_tuple().gen() == hfa


def test_get_space_tuple_no_spacetime():
    program = create_default()
    program.add_einsum(0)
    canvas = Canvas(program)

    with pytest.raises(ValueError) as excinfo:
        canvas.get_space_tuple()

    assert str(
        excinfo.value) == "SpaceTime information unspecified"


def test_get_time_tuple():
    program = create_spacetime()
    program.add_einsum(0)
    canvas = Canvas(program)

    hfa = "(k_pos, m)"
    assert canvas.get_time_tuple().gen() == hfa


def test_get_time_tuple_no_spacetime():
    program = create_default()
    program.add_einsum(0)
    canvas = Canvas(program)

    with pytest.raises(ValueError) as excinfo:
        canvas.get_time_tuple()

    assert str(
        excinfo.value) == "SpaceTime information unspecified"


def test_rel_coord_no_spacetime():
    program = create_default()
    program.add_einsum(0)
    canvas = Canvas(program)

    with pytest.raises(ValueError) as excinfo:
        canvas._Canvas__rel_coord("K")
    assert str(excinfo.value) == "SpaceTime information unspecified"
