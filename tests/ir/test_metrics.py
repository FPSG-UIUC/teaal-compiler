import pytest

from teaal.ir.component import *
from teaal.ir.hardware import Hardware
from teaal.ir.metrics import Metrics
from teaal.ir.program import Program
from teaal.ir.tensor import Tensor
from teaal.parse import *


def build_gamma_yaml():
    with open("tests/integration/gamma.yaml", "r") as f:
        return f.read()


def parse_yamls(yaml):
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)
    program.add_einsum(0)

    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    format_ = Format.from_str(yaml)

    return program, arch, bindings, format_


def test_get_collected_tensor_info_multiple_formats():
    yaml = """
    einsum:
      declaration:
        A: [M, N]
        Z: [M, N]
      expressions:
      - Z[m, n] = A[m, n]
    architecture:
      accel:
      - name: System
        local:
        - name: Memory
          class: DRAM
        subtree:
        - name: PE
          local:
          - name: Registers
            class: Buffet
    bindings:
      Z:
      - config: accel
      - component: Memory
        bindings:
        - tensor: A
          rank: N
          type: payload
          format: default0
        - tensor: A
          rank: N
          type: payload
          format: default1
      - component: Registers
        bindings:
        - tensor: A
          rank: N
          type: payload
          format: default0
          evict-on: M
        - tensor: A
          rank: N
          type: payload
          format: default1
          evict-on: M
    format:
      A:
        default0:
          rank-order: [M, N]
          M:
            format: U
          N:
            format: U
            pbits: 32
        default1:
          rank-order: [M, N]
          M:
            format: U
          N:
            format: U
            pbits: 32
    """
    program, arch, bindings, format_ = parse_yamls(yaml)
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    with pytest.raises(ValueError) as excinfo:
        metrics.get_collected_tensor_info("A")
    assert str(
        excinfo.value) in {
        "Multiple potential formats {'default0', 'default1'} for tensor A in Einsum Z",
        "Multiple potential formats {'default1', 'default0'} for tensor A in Einsum Z"}


def test_get_collected_tensor_info():
    program, arch, bindings, format_ = parse_yamls(build_gamma_yaml())
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_collected_tensor_info("A") == {(
        "K", "fiber"), ("M", "iter"), ("M", "fiber"), ("K", "iter")}
    assert metrics.get_collected_tensor_info("B") == {(
        "N", "iter"), ("K", "fiber"), ("N", "fiber"), ("K", "iter")}
    assert metrics.get_collected_tensor_info("T") == set()

    program.reset()
    program.add_einsum(1)
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_collected_tensor_info("A") == set()
    assert metrics.get_collected_tensor_info("T") == set()
    assert metrics.get_collected_tensor_info("Z") == {(
        "M", "iter"), ("N", "iter"), ("M", "fiber"), ("N", "fiber")}


def test_get_merger_init_ranks_multiple_bindings():
    yaml = """
    einsum:
      declaration:
        A: [M, N]
        Z: [M, N]
      expressions:
      - Z[m, n] = A[m, n]
    architecture:
      merger:
      - name: mergers
        local:
        - name: Merger0
          class: Merger
          attributes:
            inputs: 2
            comparator_radix: 2
        - name: Merger1
          class: Merger
          attributes:
            inputs: 2
            comparator_radix: 2
    bindings:
      Z:
      - config: merger
      - component: Merger0
        bindings:
        - tensor: A
          init-ranks: [M, N]
          final-ranks: [N, M]
      - component: Merger1
        bindings:
        - tensor: A
          init-ranks: [M, N]
          final-ranks: [N, M]
    format:
      A:
        default:
          rank-order: [N, M]
          N:
            format: U
          M:
            format: U
            pbits: 32
    """
    program, arch, bindings, format_ = parse_yamls(yaml)
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    with pytest.raises(ValueError) as excinfo:
        metrics.get_merger_init_ranks("A", ["N", "M"])
    assert str(
        excinfo.value) == "Multiple bindings for merge of tensor A to final rank order ['N', 'M']"


def test_get_merger_init_ranks():
    program, arch, bindings, format_ = parse_yamls(build_gamma_yaml())
    program.reset()
    program.add_einsum(1)

    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_merger_init_ranks(
        "T", [
            "M", "N", "K"]) == [
        "M", "K", "N"]
    assert metrics.get_merger_init_ranks(
        "T", ["M1", "M0", "K1", "N", "K0"]) is None
    assert metrics.get_merger_init_ranks("Z", ["M", "N"]) is None
