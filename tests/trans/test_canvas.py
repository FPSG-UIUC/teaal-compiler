import pytest

from es2hfa.ir.program import Program
from es2hfa.parse.input import Input
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
    return Program(Input.from_str(yaml))


def create_displayed():
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
        display:
            Z:
                space: [N]
                time: [K, M]
                style: shape
    """
    return Program(Input.from_str(yaml))


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
        display:
            Z:
                space: [N2, N1]
                time: [K, M, N0]
                style: """ + style
    return Program(Input.from_str(yaml))


def test_create_canvas():
    program = create_displayed()
    program.add_einsum(0)
    canvas = Canvas(program)

    hfa = "canvas = createCanvas(A_KM, B_KN, Z_MN)"
    assert canvas.create_canvas().gen(0) == hfa


def test_create_canvas_partitioned():
    program = create_partitioned("shape")
    program.add_einsum(0)
    for tensor in program.get_tensors():
        program.apply_partitioning(tensor)
        program.apply_loop_order(tensor)

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


def test_add_activity_no_display():
    program = create_default()
    program.add_einsum(0)

    canvas = Canvas(program)
    canvas.create_canvas()

    with pytest.raises(ValueError) as excinfo:
        canvas.add_activity()

    assert str(excinfo.value) == "Display information unspecified"


def test_add_activity():
    program = create_displayed()
    program.add_einsum(0)

    canvas = Canvas(program)
    canvas.create_canvas()

    hfa = "canvas.addActivity((k, m), (k, n), (m, n), spacetime=((n,), (k, m)))"
    assert canvas.add_activity().gen(0) == hfa


def test_add_activity_partitioned_space():
    program = create_partitioned("shape")
    program.add_einsum(0)
    for tensor in program.get_tensors():
        program.apply_partitioning(tensor)
        program.apply_loop_order(tensor)

    canvas = Canvas(program)
    canvas.create_canvas()

    hfa = "canvas.addActivity((k, m), (n2, k, n1, n0), (n2, n1, m, n0), spacetime=((n2, n1 - n2), (k, m, n0 - n1)))"
    assert canvas.add_activity().gen(0) == hfa


def test_add_activity_partitioned_occupancy():
    program = create_partitioned("occupancy")
    program.add_einsum(0)
    for tensor in program.get_tensors():
        program.apply_partitioning(tensor)
        program.apply_loop_order(tensor)

    canvas = Canvas(program)
    canvas.create_canvas()

    hfa = "canvas.addActivity((k, m), (n2, k, n1, n0), (n2, n1, m, n0), spacetime=((n2, n1_pos), (k, m, n0_pos)))"
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
    program = create_displayed()
    program.add_einsum(0)

    canvas = Canvas(program)
    canvas.create_canvas()

    hfa = "displayCanvas(canvas)"
    assert canvas.display_canvas().gen(0) == hfa


def test_displayable_true():
    program = create_displayed()
    program.add_einsum(0)
    canvas = Canvas(program)

    assert canvas.displayable()


def test_displayable_false():
    program = create_default()
    program.add_einsum(0)
    canvas = Canvas(program)

    assert not canvas.displayable()


def test_rel_coord_no_display():
    program = create_default()
    program.add_einsum(0)
    canvas = Canvas(program)

    with pytest.raises(ValueError) as excinfo:
        canvas._Canvas__rel_coord("K")
    assert str(excinfo.value) == "Display information unspecified"
