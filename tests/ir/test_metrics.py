import pytest

from es2hfa.ir.component import *
from es2hfa.ir.hardware import Hardware
from es2hfa.ir.metrics import Metrics
from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.arch import Architecture
from es2hfa.parse.bindings import Bindings
from es2hfa.parse.einsum import Einsum
from es2hfa.parse.mapping import Mapping


def build_gamma_yaml():
    yaml = """
    einsum:
      declaration:
        A: [K, M]
        B: [K, N]
        T: [K, M, N]
        Z: [M, N]

      expressions:
        - T[k,m,n] = dot(A[k,m], B[k,n], 1)
        - Z[m,n] = sum(K).(T[k,m,n]*A[k,m])

    mapping:
      rank-order:
        A: [M, K]
        B: [K, N]
        T: [M, K, N]
        Z: [M, N]

      loop-order:
        T: [M, K, N]
        Z: [M, N, K]

    architecture:
      subtree:
      - name: System
        attributes:
          clock_frequency: 1000000000

        local:
        - name: MainMemory
          class: DRAM
          attributes:
            datawidth: 8
            bandwidth: 128

        subtree:
        - name: Chip

          local:
          - name: FiberCache # 3MB FiberCache
            class: Cache
            attributes:
              width: 8
              depth: 3145728

          subtree:
          - name: PE[0..31] # 32 PEs

            subtree:
            - name: Stage0

              local:
              - name: RegFile0
                class: Buffet

              - name: Intersection
                class: LeaderFollower

            - name: Stage0to1

              local:
              - name: HighRadixMerger
                class: Merger
                attributes:
                  radix: 64
                  next_latency: 1

            - name: Stage1

              local:
              - name: RegFile1
                class: Buffet

              - name: MAC
                class: compute
    bindings:
    - name: MainMemory
      bindings:
      - tensor: A
        rank: root
      - tensor: B
        rank: root
      - tensor: Z
        rank: root

    - name: FiberCache
      bindings:
      - tensor: B
        rank: K

    - name: RegFile0
      bindings:
      - tensor: A
        rank: M
      - tensor: B
        rank: N
      - tensor: T
        rank: M

    - name: Intersection
      bindings:
      - einsum: T
        rank: K
        leader: A

    - name: HighRadixMerger
      bindings:
      # T[M, K, N] -> T[M, N, K]
      - tensor: T
        init_ranks: [M, K, N]
        swap_depth: 1

    - name: RegFile1
      bindings:
      - tensor: A
        rank: M
      - tensor: T
        rank: M
      - tensor: Z
        rank: N

    - name: MAC
      bindings:
      - einsum: Z
        op: mul
      - einsum: Z
        op: add
    """
    return yaml


def build_metrics():
    yaml = build_gamma_yaml()
    return Metrics(*build_program_hardware(yaml))


def build_program_hardware(yaml):
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)
    program.add_einsum(0)

    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    hardware = Hardware(arch, bindings)

    return program, hardware


def test_not_loaded_on_chip():
    yaml = """
    einsum:
      declaration:
        Z: [M]
      expressions:
        - Z[m] = a

    architecture:
      subtree:
      - name: System
        local:
        - name: Memory
          class: DRAM

        subtree:
        - name: PE
          local:
          - name: MAC
            class: compute

    bindings:
    - name: Memory
      bindings:
      - tensor: Z
        rank: M

    - name: MAC
      bindings:
      - einsum: Z
        op: add
    """
    program, hardware = build_program_hardware(yaml)

    with pytest.raises(ValueError) as excinfo:
        Metrics(program, hardware)
    assert str(excinfo.value) == "Tensor Z never buffered on chip"


def test_not_implemented_root_not_in_dram():
    yaml = """
    einsum:
      declaration:
        Z: [M]
      expressions:
        - Z[m] = a

    architecture:
      subtree:
      - name: System
        local:
        - name: Memory
          class: DRAM

        subtree:
        - name: PE
          local:
          - name: Buffer
            class: Buffet

          - name: MAC
            class: compute

    bindings:
    - name: Memory
      bindings:
      - tensor: Z
        rank: M

    - name: Buffer
      bindings:
      - tensor: Z
        rank: M

    - name: MAC
      bindings:
      - einsum: Z
        op: add
    """
    program, hardware = build_program_hardware(yaml)

    with pytest.raises(NotImplementedError):
        Metrics(program, hardware)


def test_get_on_chip_buffer_not_in_dram():
    metrics = build_metrics()

    with pytest.raises(ValueError) as excinfo:
        metrics.get_on_chip_buffer(Tensor("T", ["M", "K", "N"]))
    assert str(excinfo.value) == "Tensor T not stored in DRAM"


def test_get_on_chip_buffer():
    metrics = build_metrics()
    bindings = Bindings.from_str(build_gamma_yaml())

    attrs = {"width": 8, "depth": 3145728}
    cache = CacheComponent("FiberCache", attrs, bindings.get("FiberCache"))
    regs = BuffetComponent("RegFile0", {}, bindings.get("RegFile0"))

    assert metrics.get_on_chip_buffer(Tensor("A", ["M", "K"])) == regs
    assert metrics.get_on_chip_buffer(Tensor("B", ["K", "N"])) == cache


def test_get_on_chip_rank_not_in_dram():
    metrics = build_metrics()

    with pytest.raises(ValueError) as excinfo:
        metrics.get_on_chip_rank(Tensor("T", ["M", "K", "N"]))
    assert str(excinfo.value) == "Tensor T not stored in DRAM"


def test_get_on_chip_rank():
    metrics = build_metrics()

    assert metrics.get_on_chip_rank(Tensor("A", ["M", "K"])) == "M"
    assert metrics.get_on_chip_rank(Tensor("B", ["K", "N"])) == "K"


def test_in_dram():
    metrics = build_metrics()

    assert metrics.in_dram(Tensor("A", ["M", "K"]))
    assert metrics.in_dram(Tensor("B", ["M", "K"]))
    assert not metrics.in_dram(Tensor("T", ["M", "K", "N"]))


def test_on_chip_stationary():
    metrics = build_metrics()

    assert metrics.on_chip_stationary(Tensor("A", ["M", "K"]))
    assert not metrics.on_chip_stationary(Tensor("B", ["K", "N"]))


def test_on_chip_stationary_root_buffered():
    yaml = """
    einsum:
      declaration:
        Z: [M]
      expressions:
        - Z[m] = a

    architecture:
      subtree:
      - name: System
        local:
        - name: Memory
          class: DRAM

        subtree:
        - name: PE
          local:
          - name: Buffer
            class: Buffet

          - name: MAC
            class: compute

    bindings:
    - name: Memory
      bindings:
      - tensor: Z
        rank: root

    - name: Buffer
      bindings:
      - tensor: Z
        rank: root

    - name: MAC
      bindings:
      - einsum: Z
        op: add
    """
    metrics = Metrics(*build_program_hardware(yaml))

    assert metrics.on_chip_stationary(Tensor("Z", ["M"]))
