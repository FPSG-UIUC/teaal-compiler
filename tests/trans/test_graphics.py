from teaal.ir.hardware import Hardware
from teaal.ir.metrics import Metrics
from teaal.ir.program import Program
from teaal.parse import *
from teaal.trans.graphics import Graphics


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
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)
    return Graphics(program, None)


def create_spacetime(opt):
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
                opt: """ + opt
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)
    return Graphics(program, None)

def create_gamma():
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

    return Graphics(program, metrics)

def test_make_body_none():
    graphics = create_default()
    assert graphics.make_body().gen(0) == ""


def test_make_body_no_opt():
    graphics = create_spacetime("")
    graphics.make_header()
    hifiber = "canvas.addActivity((k, m), (k, n), (m, n), spacetime=((n_pos,), (k_pos, m)))"
    assert graphics.make_body().gen(0) == hifiber


def test_make_body_slip():
    graphics = create_spacetime("slip")
    graphics.make_header()
    hifiber = "if (n_pos,) in timestamps.keys():\n" + \
        "    timestamps[(n_pos,)] += 1\n" + \
        "else:\n" + \
        "    timestamps[(n_pos,)] = 1\n" + \
        "canvas.addActivity((k, m), (k, n), (m, n), spacetime=((n_pos,), (timestamps[(n_pos,)] - 1,)))"
    assert graphics.make_body().gen(0) == hifiber

def test_make_body_metrics():
    graphics = create_gamma()
    assert graphics.make_body().gen(0) == ""

def test_make_footer_none():
    graphics = create_default()
    assert graphics.make_footer().gen(0) == ""


def test_make_footer():
    graphics = create_spacetime("")
    graphics.make_header()
    hifiber = "displayCanvas(canvas)"
    assert graphics.make_footer().gen(0) == hifiber

def test_make_footer_metrics():
    graphics = create_gamma()
    assert graphics.make_footer().gen(0) == ""


def test_make_header_none():
    graphics = create_default()
    assert graphics.make_header().gen(0) == ""


def test_make_header_no_opt():
    graphics = create_spacetime("")
    hifiber = "canvas = createCanvas(A_KM, B_KN, Z_MN)"
    assert graphics.make_header().gen(0) == hifiber


def test_make_header_slip():
    graphics = create_spacetime("slip")
    hifiber = "canvas = createCanvas(A_KM, B_KN, Z_MN)\n" + \
        "timestamps = {}"
    assert graphics.make_header().gen(0) == hifiber

def test_make_header_metrics():
    graphics = create_gamma()
    assert graphics.make_header().gen(0) == ""


