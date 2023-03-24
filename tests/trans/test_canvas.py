import pytest

from teaal.ir.program import Program
from teaal.parse.einsum import Einsum
from teaal.parse.mapping import Mapping
from teaal.trans.canvas import Canvas


def create_default():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
        expressions:
            - Z[m, n] = A[k, m] * B[k, n]
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
            - Z[m, n] = A[k, m] * B[k, n]
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
            - Z[m, n] = A[k, m] * B[k, n]
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


def create_dyn_partitioned():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
        expressions:
            - Z[m, n] = A[k, m] * B[k, n]
    mapping:
        partitioning:
            Z:
                M: [uniform_shape(20), uniform_occupancy(A.5)]
        spacetime:
            Z:
                space: []
                time: [M2, K, M1, M0, N]
    """
    return Program(Einsum.from_str(yaml), Mapping.from_str(yaml))


def create_slip():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
        expressions:
            - Z[m, n] = A[k, m] * B[k, n]
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


def create_conv():
    yaml = """
    einsum:
        declaration:
            F: [S]
            I: [W]
            O: [Q]
        expressions:
            - O[q] = I[q + s] * F[s]
    mapping:
        loop-order:
            O: [W, Q]
        spacetime:
            O:
                space: []
                time: [W, Q]
    """
    return Program(Einsum.from_str(yaml), Mapping.from_str(yaml))


def test_create_canvas():
    program = create_spacetime()
    program.add_einsum(0)
    canvas = Canvas(program)

    hifiber = "canvas = createCanvas(A_KM, B_KN, Z_MN)"
    assert canvas.create_canvas().gen(0) == hifiber


def test_create_canvas_partitioned():
    program = create_partitioned("coord")
    program.add_einsum(0)

    static_parts = program.get_partitioning().get_static_parts()
    for tensor in program.get_tensors():
        for rank in static_parts:
            program.apply_partitioning(tensor, (rank[0],))
        program.get_loop_order().apply(tensor)

    canvas = Canvas(program)

    hifiber = "canvas = createCanvas(A_KM, B_N2KN1N0, Z_N2N1MN0)"
    assert canvas.create_canvas().gen(0) == hifiber


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

    hifiber = "canvas.addActivity((k, m), (k, n), (m, n), spacetime=((n_pos,), (k_pos, m)))"
    assert canvas.add_activity().gen(0) == hifiber


def test_add_activity_partitioned_coord():
    program = create_partitioned("coord")
    program.add_einsum(0)

    for tensor in program.get_tensors():
        program.apply_all_partitioning(tensor)
        program.get_loop_order().apply(tensor)

    canvas = Canvas(program)
    canvas.create_canvas()

    hifiber = "canvas.addActivity((k, m), (n2, k, n1, n0), (n2, n1, m, n0), spacetime=((n2, n1 - n2), (k, m, n0 - n1)))"
    assert canvas.add_activity().gen(0) == hifiber


def test_add_activity_partitioned_pos():
    program = create_partitioned("pos")
    program.add_einsum(0)

    for tensor in program.get_tensors():
        program.apply_all_partitioning(tensor)
        program.get_loop_order().apply(tensor)

    canvas = Canvas(program)
    canvas.create_canvas()

    hifiber = "canvas.addActivity((k, m), (n2, k, n1, n0), (n2, n1, m, n0), spacetime=((n2_pos, n1_pos), (k_pos, m_pos, n0_pos)))"
    assert canvas.add_activity().gen(0) == hifiber


def test_add_activity_slip():
    program = create_slip()
    program.add_einsum(0)

    canvas = Canvas(program)
    canvas.create_canvas()

    hifiber = "canvas.addActivity((k, m), (k, n), (m, n), spacetime=((n_pos,), (timestamps[(n_pos,)] - 1,)))"
    assert canvas.add_activity().gen(0) == hifiber


def test_add_activity_dyn_part():
    program = create_dyn_partitioned()
    program.add_einsum(0)
    program.apply_all_partitioning(program.get_equation().get_output())
    program.apply_partitioning(program.get_equation().get_tensor("A"), ("M",))

    canvas = Canvas(program)
    canvas.create_canvas()

    hifiber = "canvas.addActivity((k, m2, m0), (k, n), (m2, m1, m0, n), spacetime=((), (m2_pos, k_pos, m1_pos, m0_pos, n_pos)))"
    assert canvas.add_activity().gen(0) == hifiber


def test_add_activity_flatten():
    yaml = """
    einsum:
        declaration:
            A: [M, N]
            Z: [M, N]
        expressions:
            - Z[m, n] = A[m, n]
    mapping:
        partitioning:
            Z:
                (M, N): [flatten()]
        loop-order:
            Z: [MN]
        spacetime:
            Z:
                space: []
                time: [MN]
    """
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)
    part_ir = program.get_partitioning()

    for tensor in program.get_tensors():
        program.apply_all_partitioning(tensor)

    canvas = Canvas(program)
    canvas.create_canvas()

    hifiber = "canvas.addActivity(((m, n),), ((m, n),), spacetime=((), (mn_pos,)))"
    assert canvas.add_activity().gen(0) == hifiber


def test_add_activity_conv():
    program = create_conv()
    program.add_einsum(0)

    canvas = Canvas(program)
    canvas.create_canvas()

    hifiber = "canvas.addActivity((w,), (w + -1 * q,), (q,), spacetime=((), (w_pos, q_pos)))"
    assert canvas.add_activity().gen(0) == hifiber


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

    hifiber = "displayCanvas(canvas)"
    assert canvas.display_canvas().gen(0) == hifiber


def test_get_space_tuple():
    program = create_spacetime()
    program.add_einsum(0)
    canvas = Canvas(program)

    hifiber = "(n_pos,)"
    assert canvas.get_space_tuple().gen() == hifiber


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

    hifiber = "(k_pos, m)"
    assert canvas.get_time_tuple().gen() == hifiber


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
