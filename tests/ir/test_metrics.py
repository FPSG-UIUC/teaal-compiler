import pytest

from teaal.ir.component import *
from teaal.ir.hardware import Hardware
from teaal.ir.metrics import Metrics
from teaal.ir.program import Program
from teaal.ir.tensor import Tensor
from teaal.parse import *


def build_extensor_yaml():
    with open("tests/integration/extensor.yaml", "r") as f:
        return f.read()


def build_extensor_energy_yaml():
    with open("tests/integration/extensor-energy.yaml", "r") as f:
        return f.read()


def build_gamma_yaml():
    with open("tests/integration/gamma.yaml", "r") as f:
        return f.read()


def build_sigma_yaml():
    with open("tests/integration/sigma.yaml", "r") as f:
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
        prefix: tmp/Z
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


def test_expand_eager():
    program, arch, bindings, format_ = parse_yamls(build_extensor_yaml())
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    bindings = {'Z': [
        {'tensor': 'A', 'rank': 'K1', 'type': 'coord', 'evict-on': 'M2', 'format': 'default', 'style': 'lazy'},
        {'tensor': 'A', 'rank': 'M0', 'type': 'coord', 'evict-on': 'M2', 'format': 'default', 'style': 'eager', 'root': 'M0'},
        {'tensor': 'B', 'rank': 'N1', 'type': 'coord', 'evict-on': 'K2', 'format': 'default', 'style': 'lazy'},
        {'tensor': 'B', 'rank': 'N1', 'type': 'payload', 'evict-on': 'K2', 'format': 'default', 'style': 'lazy'},
        {'tensor': 'B', 'rank': 'K1', 'type': 'coord', 'evict-on': 'K2', 'format': 'default', 'style': 'lazy'},
        {'tensor': 'B', 'rank': 'K1', 'type': 'payload', 'evict-on': 'K2', 'format': 'default', 'style': 'lazy'},
        {'tensor': 'B', 'rank': 'N0', 'type': 'coord', 'evict-on': 'K2', 'format': 'default', 'style': 'eager', 'root': 'N0'},
        {'tensor': 'Z', 'rank': 'M0', 'type': 'coord', 'evict-on': 'M2', 'format': 'default', 'style': 'eager', 'root': 'M0'},
        {'tensor': 'Z', 'evict-on': 'M2', 'style': 'eager', 'format': 'default', 'root': 'M0', 'rank': 'N0', 'type': 'coord'},
        {'tensor': 'Z', 'evict-on': 'M2', 'style': 'eager', 'format': 'default', 'root': 'M0', 'rank': 'N0', 'type': 'payload'},
        {'tensor': 'A', 'evict-on': 'M2', 'style': 'eager', 'format': 'default', 'root': 'M0', 'rank': 'M0', 'type': 'payload'},
        {'tensor': 'A', 'evict-on': 'M2', 'style': 'eager', 'format': 'default', 'root': 'M0', 'rank': 'K0', 'type': 'coord'},
        {'tensor': 'A', 'evict-on': 'M2', 'style': 'eager', 'format': 'default', 'root': 'M0', 'rank': 'K0', 'type': 'payload'},
        {'tensor': 'B', 'evict-on': 'K2', 'style': 'eager', 'format': 'default', 'root': 'N0', 'rank': 'N0', 'type': 'payload'},
        {'tensor': 'B', 'evict-on': 'K2', 'style': 'eager', 'format': 'default', 'root': 'N0', 'rank': 'K0', 'type': 'coord'},
        {'tensor': 'B', 'evict-on': 'K2', 'style': 'eager', 'format': 'default', 'root': 'N0', 'rank': 'K0', 'type': 'payload'}]}

    assert hardware.get_component("LLB").get_bindings()["Z"] == bindings["Z"]


def test_expand_eager_elem():
    yaml = """
    einsum:
      declaration:
        Z: []
        A: [K, M]
        B: [K]
      expressions:
      - Z[] = A[k, m] * B[k]
    architecture:
      accel:
      - name: level0
        local:
        - name: Buffer
          class: Buffet
    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
      - component: Buffer
        bindings:
        - tensor: A
          rank: K
          type: payload
          evict-on: root
          format: default
          style: eager
    format:
      A:
        default:
          rank-order: [K, M]
          K:
            format: C
            pbits: 32
          M:
            format: C
            cbits: 32
            pbits: 32
            layout: interleaved
    """
    program, arch, bindings, format_ = parse_yamls(yaml)
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    bindings = {'Z': [{'tensor': 'A',
                       'rank': 'K',
                       'type': 'payload',
                       'evict-on': 'root',
                       'format': 'default',
                       'style': 'eager',
                       'root': 'K'},
                      {'tensor': 'A',
                       'evict-on': 'root',
                       'style': 'eager',
                       'format': 'default',
                       'root': 'K',
                       'rank': 'M',
                       'type': 'elem'}]}

    assert hardware.get_component("Buffer").get_bindings() == bindings


def test_get_coiter():
    program, arch, bindings, format_ = parse_yamls(build_gamma_yaml())
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_coiter("K") == hardware.get_component("Intersect")


def test_get_coiter_traces_leader_follower():
    program, arch, bindings, format_ = parse_yamls(build_gamma_yaml())
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_coiter_traces("Intersect", "K") == ["intersect_2"]


def test_get_coiter_traces_two_finger_more_than_two():
    yaml = """
    einsum:
      declaration:
        Z: []
        A: [K]
        B: [K]
        C: [K]
      expressions:
      - Z[] = A[k] * B[k] * C[k]
    architecture:
      accel:
      - name: level0
        local:
        - name: Intersect
          class: Intersector
          attributes:
            type: two-finger
    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
      - component: Intersect
        bindings:
        - rank: K
    # TODO: Allow the format to be empty
    format:
      Z:
        default:
          rank-order: []
    """
    program, arch, bindings, format_ = parse_yamls(yaml)
    hardware = Hardware(arch, bindings, program)

    with pytest.raises(NotImplementedError):
        Metrics(program, hardware, format_)


def test_get_coiter_traces_two_finger():
    yaml = """
    einsum:
      declaration:
        Z: []
        A: [K]
        B: [K]
      expressions:
      - Z[] = A[k] * B[k]
    architecture:
      accel:
      - name: level0
        local:
        - name: Intersect
          class: Intersector
          attributes:
            type: two-finger
    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
      - component: Intersect
        bindings:
        - rank: K
    # TODO: Allow the format to be empty
    format:
      Z:
        default:
          rank-order: []
    """
    program, arch, bindings, format_ = parse_yamls(yaml)
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_coiter_traces("Intersect", "K") == [
        "intersect_0", "intersect_1"]


def test_get_collected_iter_info():
    program, arch, bindings, format_ = parse_yamls(build_gamma_yaml())
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_collected_iter_info() == set()

    program, arch, bindings, format_ = parse_yamls(
        build_extensor_energy_yaml())
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_collected_iter_info() == {
        "N2", "K2", "M2", "M1", "N1", "K1", "M0", "N0", "K0"}


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

    assert metrics.get_collected_tensor_info("A") == {(
        "K", "fiber", False), ("M", "iter", False), ("M", "fiber", False), ("K", "iter", False)}
    assert metrics.get_collected_tensor_info("T") == set()
    assert metrics.get_collected_tensor_info("Z") == {(
        "M", "iter", False), ("N", "iter", False), ("M", "fiber", False), ("N", "fiber", False)}


def test_get_collected_tensor_info_eager():
    program, arch, bindings, format_ = parse_yamls(build_extensor_yaml())
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_collected_tensor_info("A") == {('M0', 'M0', False), ('K1', 'fiber', True), (
        'K2', 'fiber', True), ('K1', 'fiber', False), ('K0', 'M0', False), ('K0', 'fiber', True)}
    assert metrics.get_collected_tensor_info("B") == {
        ('N1', 'fiber', False),
        ('K0', 'N0', False),
        ('N1', 'iter', False),
        ('N0', 'N0', False),
        ('K1', 'fiber', True),
        ('K2', 'fiber', True),
        ('K1', 'fiber', False),
        ('K1', 'iter', False),
        ('K0', 'fiber', True)}
    assert metrics.get_collected_tensor_info(
        "Z") == {('N0', 'M0', False), ("M0", "M0", False)}


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
        prefix: tmp/Z
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


def test_get_collected_tensor_info_flattening():
    program, arch, bindings, format_ = parse_yamls(build_sigma_yaml())
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_collected_tensor_info("A") == {("MK00", "MK00", False)}
    assert metrics.get_collected_tensor_info("B") == {("K0", "K0", False)}


def test_get_eager_evict_on():
    program, arch, bindings, format_ = parse_yamls(build_extensor_yaml())
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_eager_evict_on("A", "K2") == []
    assert metrics.get_eager_evict_on("A", "M0") == ["M2"]
    assert metrics.get_eager_evict_on("B", "N0") == ["K2"]


def test_get_eager_evicts():
    program, arch, bindings, format_ = parse_yamls(build_extensor_yaml())
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_eager_evicts("N2") == []
    assert metrics.get_eager_evicts("K2") == [("B", "N0")]
    assert metrics.get_eager_evicts("M2") == [("A", "M0"), ("Z", "M0")]


def test_get_eager_write():
    program, arch, bindings, format_ = parse_yamls(build_gamma_yaml())
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert not metrics.get_eager_write()

    program, arch, bindings, format_ = parse_yamls(build_extensor_yaml())
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_eager_write()


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
        prefix: tmp/Z0
      Z1:
      - config: accel
        prefix: tmp/Z1
      Z2:
      - config: accel
        prefix: tmp/Z2
      Z3:
      - config: accel
        prefix: tmp/Z3
      Z4:
      - config: accel
        prefix: tmp/Z4
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


def test_get_fiber_trace_coord_math():
    yaml = """
    einsum:
      declaration:
        A: [K]
        B: [M]
        Z0: [M]
        Z1: [M]
        Z2: [M]
      expressions:
      - Z0[m] = A[2 * m]
      - Z1[m] = A[2 * m] + B[m]
      - Z2[m] = A[2 * m] * B[m]
    architecture:
      accel:
      - name: empty
    bindings:
      Z0:
      - config: accel
        prefix: tmp/Z0
      Z1:
      - config: accel
        prefix: tmp/Z1
      Z2:
      - config: accel
        prefix: tmp/Z2
    format:
      Z0:
        default:
          rank-order: [M]
          M:
            format: C
    """
    program, arch, bindings, format_ = parse_yamls(yaml)
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_fiber_trace("A", "K", True) == "populate_1"
    assert metrics.get_fiber_trace("Z0", "M", True) == "populate_read_0"
    assert metrics.get_fiber_trace("Z0", "M", False) == "populate_write_0"

    program.reset()
    program.add_einsum(1)
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_fiber_trace("A", "K", True) == "union_2"
    assert metrics.get_fiber_trace("B", "M", True) == "union_3"
    assert metrics.get_fiber_trace("Z1", "M", True) == "populate_read_0"
    assert metrics.get_fiber_trace("Z1", "M", False) == "populate_write_0"

    program.reset()
    program.add_einsum(2)
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_fiber_trace("A", "K", True) == "intersect_2"
    assert metrics.get_fiber_trace("B", "M", True) == "intersect_3"
    assert metrics.get_fiber_trace("Z2", "M", True) == "populate_read_0"
    assert metrics.get_fiber_trace("Z2", "M", False) == "populate_write_0"


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
        prefix: tmp/Z
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
        prefix: tmp/Z
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
        prefix: tmp/Z
      - component: LeaderFollower
        bindings:
        - rank: M
          leader: C
        - rank: K
          leader: B
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
    assert metrics.get_fiber_trace("C", "M", True) == "intersect_2"
    assert metrics.get_fiber_trace("A", "M", True) == "intersect_3"
    assert metrics.get_fiber_trace("B", "M", True) == "intersect_4"
    assert metrics.get_fiber_trace("D", "M", True) == "intersect_5"

    assert metrics.get_fiber_trace("B", "K", True) == "intersect_0"
    assert metrics.get_fiber_trace("A", "K", True) == "intersect_1"
    assert metrics.get_fiber_trace("C", "K", True) == "intersect_2"
    assert metrics.get_fiber_trace("D", "K", True) == "intersect_3"


def test_get_fiber_trace_get_payload():
    program, arch, bindings, format_ = parse_yamls(build_sigma_yaml())
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_fiber_trace("B", "K0", True) == "get_payload_B"


def test_get_format():
    program, arch, bindings, format_ = parse_yamls(build_gamma_yaml())
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_format() == format_


def test_get_hardware():
    program, arch, bindings, format_ = parse_yamls(build_gamma_yaml())
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_hardware() == hardware


def test_get_loop_formats():
    program, arch, bindings, format_ = parse_yamls(build_gamma_yaml())
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_loop_formats() == {"A": "default", "B": "default"}


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
        prefix: tmp/Z
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


def test_get_source_memory_not_memory():
    yaml = """
    einsum:
      declaration:
        Z: [M]
      expressions:
        - Z[m] = a
    architecture:
      accel:
      - name: level0
        local:
        - name: LeaderFollower
          class: Intersector
          attributes:
            type: leader-follower
        - name: DRAM
          class: DRAM
    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
      - component: DRAM
        bindings:
        - tensor: Z
          rank: M
          type: payload
          format: default
    format:
      Z:
        default:
          rank-order: [M]
          M:
            format: C
            pbits: 64
    """
    program, arch, bindings, format_ = parse_yamls(yaml)
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    with pytest.raises(ValueError) as excinfo:
        metrics.get_source_memory("LeaderFollower", "Z", "M", "payload")
    assert str(
        excinfo.value) == "Destination component LeaderFollower not a memory"


def test_get_source_memory():
    yaml = """
    einsum:
      declaration:
        A: [K, M]
        B: [K, M]
        C: [K]
        Z: [M]
      expressions:
      - Z[m] = A[k, m] * B[k, m] * C[k]

    architecture:
      accel:
      - name: level0
        local:
        - name: DRAM
          class: DRAM
          attributes:
            bandwidth: 512

        subtree:
        - name: level1
          local:
          - name: L2Cache
            class: Cache
            attributes:
              width: 64
              depth: 1024
              bandwidth: 2048

          subtree:
          - name: level2
            local:
            - name: L1Cache
              class: Cache
              attributes:
                width: 64
                depth: 128

    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
      - component: DRAM
        bindings:
        - tensor: A
          rank: M
          type: payload
          format: default
        - tensor: A
          rank: K
          type: coord
          format: default
        - tensor: A
          rank: K
          type: payload
          format: default
        - tensor: Z
          rank: M
          type: elem
          format: default
      - component: L2Cache
        bindings:
        - tensor: A
          rank: M
          type: payload
          format: default
        - tensor: A
          rank: K
          type: coord
          format: default
        - tensor: A
          rank: K
          type: payload
          format: default
        - tensor: Z
          rank: M
          type: elem
          format: default
      - component: L1Cache
        bindings:
        - tensor: A
          rank: K
          type: coord
          format: default
        - tensor: A
          rank: K
          type: payload
          format: default
        - tensor: B
          rank: K
          type: payload
          format: default
        - tensor: Z
          rank: M
          type: elem
          format: default

    format:
      A:
        default:
          rank-order: [M, K]
          M:
            format: U
            pbits: 32
          K:
            format: C
            cbits: 32
            pbits: 64
      B:
        default:
          rank-order: [M, K]
          M:
            format: U
          K:
            format: U
            pbits: 64
      Z:
        default:
          rank-order: [M]
          M:
            format: C
            cbits: 32
            pbits: 64
    """
    program, arch, bindings, format_ = parse_yamls(yaml)
    hardware = Hardware(arch, bindings, program)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_source_memory("L2Cache", "C", "K", "payload") is None
    assert metrics.get_source_memory("L1Cache", "B", "M", "payload") is None
    assert metrics.get_source_memory("L2Cache", "B", "K", "payload") is None
    assert metrics.get_source_memory("L1Cache", "B", "K", "payload") is None
    assert metrics.get_source_memory(
        "L2Cache", "A", "M", "payload") == hardware.get_component("DRAM")
    assert metrics.get_source_memory(
        "L1Cache", "Z", "M", "elem") == hardware.get_component("L2Cache")
