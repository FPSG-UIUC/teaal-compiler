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


def test_used_traffic_paths():
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

    with pytest.raises(ValueError) as excinfo:
        Metrics(program, hardware, format_)
    assert str(
        excinfo.value) in {
        "Multiple potential formats {'default0', 'default1'} for tensor A in Einsum Z",
        "Multiple potential formats {'default1', 'default0'} for tensor A in Einsum Z"}


def test_get_collected_tensor_info():
    program, arch, bindings, format_ = parse_yamls(build_gamma_yaml())
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_collected_tensor_info("A") == {("K", "fiber", False), (
        "M", "iter", False), ("M", "fiber", False), ("K", "iter", False), ("K", "fiber", True)}
    assert metrics.get_collected_tensor_info("B") == {(
        "N", "iter", False), ("K", "fiber", False), ("N", "fiber", False), ("K", "iter", False)}
    assert metrics.get_collected_tensor_info("T") == set()

    program.reset()
    program.add_einsum(1)
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_collected_tensor_info("A") == set()
    assert metrics.get_collected_tensor_info("T") == set()
    assert metrics.get_collected_tensor_info("Z") == {(
        "M", "iter", False), ("N", "iter", False), ("M", "fiber", False), ("N", "fiber", False)}


def test_get_collected_tensor_info_extra_intersection_test():
    yaml = """
    einsum:
      declaration:
        Z: [M, N]
        A: [M]
        B: [M]
        C: [N]
      expressions:
      - Z[m, n] = A[m] * B[m] * C[n]
    architecture:
      accel:
      - name: level0
        local:
        - name: Intersector
          class: Intersector
          attributes:
            type: two-finger
    bindings:
      Z:
      - config: accel
      - component: Intersector
        bindings:
        - rank: M
    format:
      Z:
        default:
          rank-order: [M, N]
          M:
            format: C
          N:
            format: C
            pbits: 64
    """
    program, arch, bindings, format_ = parse_yamls(yaml)
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_collected_tensor_info("A") == {("M", "fiber", True)}
    assert metrics.get_collected_tensor_info("B") == {("M", "fiber", True)}
    assert metrics.get_collected_tensor_info("C") == set()
    assert metrics.get_collected_tensor_info("Z") == set()


def test_get_fiber_trace():
    yaml = """
    einsum:
      declaration:
        Z0: [M]
        Z1: [M]
        Z2: [M]
        Z3: [M]
        Z4: [M]
        A: [M, K]
        B: [M, K]
        C: [M, K]
        D: [M, K]
        E: [M, K]
        F: [M, K]
        G: [M, K]
      expressions:
        - Z0[m] = a
        - Z1[m] = A[m, k]
        - Z2[m] = A[m, k] * B[m, k]
        - Z3[m] = A[m, k] + B[m, k]
        - Z4[m] = A[m, k] * B[m, k] * C[m, k] + D[m, k] + E[m, k] * F[m, k] + G[m, k]
    architecture:
      accel:
      - name: empty
    bindings:
      Z0:
      - config: accel
      Z1:
      - config: accel
      Z2:
      - config: accel
      Z3:
      - config: accel
      Z4:
      - config: accel
    format:
      Z0:
        default:
          rank-order: [M]
          M:
            format: C
            pbits: 64
    """
    program, arch, bindings, format_ = parse_yamls(yaml)
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_fiber_trace("Z0", "M", True) == "iter"
    assert metrics.get_fiber_trace("Z0", "M", False) == "iter"

    program.reset()
    program.add_einsum(1)
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_fiber_trace("Z1", "M", True) == "populate_read_0"
    assert metrics.get_fiber_trace("Z1", "M", False) == "populate_write_0"
    assert metrics.get_fiber_trace("A", "M", True) == "populate_1"
    assert metrics.get_fiber_trace("A", "K", True) == "iter"

    program.reset()
    program.add_einsum(2)
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_fiber_trace("Z2", "M", True) == "populate_read_0"
    assert metrics.get_fiber_trace("Z2", "M", False) == "populate_write_0"
    assert metrics.get_fiber_trace("A", "M", True) == "intersect_2"
    assert metrics.get_fiber_trace("A", "K", True) == "intersect_0"
    assert metrics.get_fiber_trace("B", "M", True) == "intersect_3"
    assert metrics.get_fiber_trace("B", "K", True) == "intersect_1"

    program.reset()
    program.add_einsum(3)
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_fiber_trace("Z3", "M", True) == "populate_read_0"
    assert metrics.get_fiber_trace("Z3", "M", False) == "populate_write_0"
    assert metrics.get_fiber_trace("A", "M", True) == "union_2"
    assert metrics.get_fiber_trace("A", "K", True) == "union_0"
    assert metrics.get_fiber_trace("B", "M", True) == "union_3"
    assert metrics.get_fiber_trace("B", "K", True) == "union_1"

    program.reset()
    program.add_einsum(4)
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_fiber_trace("Z4", "M", True) == "populate_read_0"
    assert metrics.get_fiber_trace("Z4", "M", False) == "populate_write_0"
    assert metrics.get_fiber_trace("A", "M", True) == "intersect_4"
    assert metrics.get_fiber_trace("A", "K", True) == "intersect_2"
    assert metrics.get_fiber_trace("B", "M", True) == "intersect_6"
    assert metrics.get_fiber_trace("B", "K", True) == "intersect_4"
    assert metrics.get_fiber_trace("C", "M", True) == "intersect_7"
    assert metrics.get_fiber_trace("C", "K", True) == "intersect_5"
    assert metrics.get_fiber_trace("D", "M", True) == "union_8"
    assert metrics.get_fiber_trace("D", "K", True) == "union_6"
    assert metrics.get_fiber_trace("E", "M", True) == "intersect_12"
    assert metrics.get_fiber_trace("E", "K", True) == "intersect_10"
    assert metrics.get_fiber_trace("F", "M", True) == "intersect_13"
    assert metrics.get_fiber_trace("F", "K", True) == "intersect_11"
    assert metrics.get_fiber_trace("G", "M", True) == "union_11"
    assert metrics.get_fiber_trace("G", "K", True) == "union_9"


def test_get_fiber_trace_leader_follower_multiple_intersectors():
    yaml = """
    einsum:
      declaration:
        Z: [M]
        A: [M, K]
        B: [M, K]
        C: [M, K]
        D: [M, K]
      expressions:
        - Z[m] = A[m, k] * B[m, k] + C[m, k] * D[m, k]
    architecture:
      accel:
      - name: level0
        local:
        - name: LeaderFollower0
          class: Intersector
          attributes:
            type: leader-follower
        - name: LeaderFollower1
          class: Intersector
          attributes:
            type: leader-follower
    bindings:
      Z:
      - config: accel
      - component: LeaderFollower0
        bindings:
        - rank: M
          leader: A
      - component: LeaderFollower1
        bindings:
        - rank: M
          leader: A
    format:
      Z0:
        default:
          rank-order: [M]
          M:
            format: C
            pbits: 64
    """
    program, arch, bindings, format_ = parse_yamls(yaml)
    hardware = Hardware(arch, bindings, program)

    with pytest.raises(NotImplementedError):
        metrics = Metrics(program, hardware, format_)


def test_get_fiber_trace_leader_follower_multiple_terms():
    yaml = """
    einsum:
      declaration:
        Z: [M]
        A: [M, K]
        B: [M, K]
        C: [M, K]
        D: [M, K]
      expressions:
        - Z[m] = A[m, k] * B[m, k] + C[m, k] * D[m, k]
    architecture:
      accel:
      - name: level0
        local:
        - name: LeaderFollower
          class: Intersector
          attributes:
            type: leader-follower
    bindings:
      Z:
      - config: accel
      - component: LeaderFollower
        bindings:
        - rank: M
          leader: A
        - rank: K
          leader: A
    format:
      Z0:
        default:
          rank-order: [M]
          M:
            format: C
            pbits: 64
    """
    program, arch, bindings, format_ = parse_yamls(yaml)
    hardware = Hardware(arch, bindings, program)

    with pytest.raises(NotImplementedError):
        metrics = Metrics(program, hardware, format_)


def test_get_fiber_trace_leader_follower():
    yaml = """
    einsum:
      declaration:
        Z: [M]
        A: [M, K]
        B: [M, K]
        C: [M, K]
        D: [M, K]
      expressions:
        - Z[m] = A[m, k] * B[m, k] * C[m, k] * D[m, k]
    architecture:
      accel:
      - name: level0
        local:
        - name: LeaderFollower
          class: Intersector
          attributes:
            type: leader-follower
    bindings:
      Z:
      - config: accel
      - component: LeaderFollower
        bindings:
        - rank: M
          leader: A
        - rank: K
          leader: A
    format:
      Z0:
        default:
          rank-order: [M]
          M:
            format: C
            pbits: 64
    """
    program, arch, bindings, format_ = parse_yamls(yaml)
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_fiber_trace("Z", "M", True) == "populate_read_0"
    assert metrics.get_fiber_trace("Z", "M", False) == "populate_write_0"
    assert metrics.get_fiber_trace("A", "M", True) == "intersect_2"
    assert metrics.get_fiber_trace("A", "K", True) == "intersect_0"
    assert metrics.get_fiber_trace("B", "M", True) == "intersect_3"
    assert metrics.get_fiber_trace("B", "K", True) == "intersect_1"
    assert metrics.get_fiber_trace("C", "M", True) == "intersect_4"
    assert metrics.get_fiber_trace("C", "K", True) == "intersect_2"
    assert metrics.get_fiber_trace("D", "M", True) == "intersect_5"
    assert metrics.get_fiber_trace("D", "K", True) == "intersect_3"


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
