import pytest

from es2hfa.ir.mapping import Mapping
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
    return Mapping(Input.from_str(yaml))


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
    """
    return Mapping(Input.from_str(yaml))


def create_partitioned():
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
    """
    return Mapping(Input.from_str(yaml))


def test_create_canvas():
    mapping = create_displayed()
    mapping.add_einsum(0)
    canvas = Canvas(mapping)

    hfa = "canvas = createCanvas(A_KM, B_KN, Z_MN)"
    assert canvas.create_canvas().gen(0) == hfa


def test_create_canvas_partitioned():
    mapping = create_partitioned()
    mapping.add_einsum(0)
    for tensor in mapping.get_tensors():
        mapping.apply_partitioning(tensor)
        mapping.apply_loop_order(tensor)

    canvas = Canvas(mapping)

    hfa = "canvas = createCanvas(A_KM, B_N2KN1N0, Z_N2N1MN0)"
    assert canvas.create_canvas().gen(0) == hfa


def test_add_activity_no_canvas():
    mapping = create_default()
    mapping.add_einsum(0)
    canvas = Canvas(mapping)

    with pytest.raises(ValueError) as excinfo:
        canvas.add_activity()

    assert str(
        excinfo.value) == "Unconfigured canvas. Make sure to first call create_canvas()"


def test_add_activity_no_display():
    mapping = create_default()
    mapping.add_einsum(0)

    canvas = Canvas(mapping)
    canvas.create_canvas()

    with pytest.raises(ValueError) as excinfo:
        canvas.add_activity()

    assert str(excinfo.value) == "Display information unspecified"


def test_add_activity():
    mapping = create_displayed()
    mapping.add_einsum(0)

    canvas = Canvas(mapping)
    canvas.create_canvas()

    hfa = "canvas.addActivity((k, m), (k, n), (m, n), spacetime=((n,), (k, m)))"
    assert canvas.add_activity().gen(0) == hfa


def test_add_activity_partitioned():
    mapping = create_partitioned()
    mapping.add_einsum(0)
    for tensor in mapping.get_tensors():
        mapping.apply_partitioning(tensor)
        mapping.apply_loop_order(tensor)

    canvas = Canvas(mapping)
    canvas.create_canvas()

    hfa = "canvas.addActivity((k, m), (n2, k, n1, n0), (n2, n1, m, n0), spacetime=((n2, n1), (k, m, n0)))"
    assert canvas.add_activity().gen(0) == hfa


def test_display_canvas_no_canvas():
    mapping = create_default()
    mapping.add_einsum(0)
    canvas = Canvas(mapping)

    with pytest.raises(ValueError) as excinfo:
        canvas.display_canvas()

    assert str(
        excinfo.value) == "Unconfigured canvas. Make sure to first call create_canvas()"


def test_display_canvas():
    mapping = create_displayed()
    mapping.add_einsum(0)

    canvas = Canvas(mapping)
    canvas.create_canvas()

    hfa = "displayCanvas(canvas)"
    assert canvas.display_canvas().gen(0) == hfa


def test_displayable_true():
    mapping = create_displayed()
    mapping.add_einsum(0)
    canvas = Canvas(mapping)

    assert canvas.displayable()


def test_displayable_false():
    mapping = create_default()
    mapping.add_einsum(0)
    canvas = Canvas(mapping)

    assert not canvas.displayable()
