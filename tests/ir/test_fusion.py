from teaal.ir.fusion import Fusion
from teaal.ir.hardware import Hardware
from teaal.ir.metrics import Metrics
from teaal.ir.program import Program
from teaal.parse import *

def make_yaml(spacetime, bindings):
    yaml = """
    einsum:
      declaration:
        A: [K, M]
        B: [K, N]
        T: [K, M, N]
        C: [M, N]
        Z: [M, N]
      expressions:
      - T[k, m, n] = A[k, m] * B[k, n]
      - Z[m, n] = T[k, m, n] * C[m, n]
    mapping:
      loop-order:
        T: [M, K, N]
        Z: [M, K, N]
      spacetime:""" + spacetime + """
    format:
      # TODO: allow empty format
      Z:
        default:
          rank-order: [M, N]
          M:
            format: C
          N:
            format: C
            pbits: 32
    architecture:
      configA:
      - name: System
        local:
        - name: FPMul0
          class: compute
          attributes:
            type: mul
        - name: FPMul1
          class: compute
          attributes:
            type: mul
      configB:
      - name: System
        local:
        - name: FPMul
          class: compute
          attributes:
            type: mul
    bindings:""" + bindings

    return yaml

def parse_yamls(yaml):
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)
    program.add_einsum(0)

    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    hardware = Hardware(arch, bindings, program)

    format_ = Format.from_str(yaml)

    return program, hardware, format_

def test_add_einsum_diff_configs():
    spacetime = """
        T:
          space: [N]
          time: [M, K]
        Z:
          space: [N]
          time: [M, K]
    """

    bindings = """
      T:
      - config: configA
        prefix: tmp/T
      Z:
      - config: configB
        prefix: tmp/Z
    """
    yaml = make_yaml(spacetime, bindings)

    program, hardware, format_ = parse_yamls(yaml)
    fusion = Fusion()

    program.add_einsum(0)
    metrics = Metrics(program, hardware, format_)
    fusion.add_einsum(program, metrics)

    program.add_einsum(1)
    metrics = Metrics(program, hardware, format_)
    fusion.add_einsum(program, metrics)

    assert fusion.get_blocks() == [["T"], ["Z"]]

def test_add_einsum_diff_temporal_ranks():
    spacetime = """
        T:
          space: [N]
          time: [M, K]
        Z:
          space: [K]
          time: [M, N]
    """

    bindings = """
      T:
      - config: configA
        prefix: tmp/T
      Z:
      - config: configA
        prefix: tmp/Z
    """
    yaml = make_yaml(spacetime, bindings)

    program, hardware, format_ = parse_yamls(yaml)
    fusion = Fusion()

    program.add_einsum(0)
    metrics = Metrics(program, hardware, format_)
    fusion.add_einsum(program, metrics)

    program.add_einsum(1)
    metrics = Metrics(program, hardware, format_)
    fusion.add_einsum(program, metrics)

    assert fusion.get_blocks() == [["T"], ["Z"]]

def test_add_einsum_diff_temporal_ranks():
    spacetime = """
        T:
          space: [N]
          time: [M, K]
        Z:
          space: [K]
          time: [M, N]
    """

    bindings = """
      T:
      - config: configA
        prefix: tmp/T
      Z:
      - config: configA
        prefix: tmp/Z
    """
    yaml = make_yaml(spacetime, bindings)

    program, hardware, format_ = parse_yamls(yaml)
    fusion = Fusion()

    program.add_einsum(0)
    metrics = Metrics(program, hardware, format_)
    fusion.add_einsum(program, metrics)

    program.add_einsum(1)
    metrics = Metrics(program, hardware, format_)
    fusion.add_einsum(program, metrics)

    assert fusion.get_blocks() == [["T"], ["Z"]]

def test_add_einsum_overlapping_components():
    spacetime = """
        T:
          space: [N]
          time: [M, K]
        Z:
          space: [K]
          time: [M, N]
    """

    bindings = """
      T:
      - config: configA
        prefix: tmp/T
      - component: FPMul0
        bindings:
        - op: mul
      Z:
      - config: configA
        prefix: tmp/Z
      - component: FPMul0
        bindings:
        - op: mul
    """
    yaml = make_yaml(spacetime, bindings)

    program, hardware, format_ = parse_yamls(yaml)
    fusion = Fusion()

    program.add_einsum(0)
    metrics = Metrics(program, hardware, format_)
    fusion.add_einsum(program, metrics)

    program.add_einsum(1)
    metrics = Metrics(program, hardware, format_)
    fusion.add_einsum(program, metrics)

    assert fusion.get_blocks() == [["T"], ["Z"]]

def test_add_einsum_fused():
    spacetime = """
        T:
          space: [N]
          time: [M, K]
        Z:
          space: [N]
          time: [M, K]
    """

    bindings = """
      T:
      - config: configA
        prefix: tmp/T
      - component: FPMul0
        bindings:
        - op: mul
      Z:
      - config: configA
        prefix: tmp/Z
      - component: FPMul1
        bindings:
        - op: mul
    """
    yaml = make_yaml(spacetime, bindings)

    program, hardware, format_ = parse_yamls(yaml)
    fusion = Fusion()

    program.add_einsum(0)
    metrics = Metrics(program, hardware, format_)
    fusion.add_einsum(program, metrics)

    program.add_einsum(1)
    metrics = Metrics(program, hardware, format_)
    fusion.add_einsum(program, metrics)

    assert fusion.get_blocks() == [["T", "Z"]]
