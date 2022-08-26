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

    format_ = Format.from_str(yaml)

    return program, hardware, format_


def test_check_configuration_no_dyn_part():
    yaml = """
    einsum:
      declaration:
        A: [M]
        Z: [M]
      expressions:
        - Z[m] = A[m]
    mapping:
      partitioning:
        Z:
          M: [uniform_occupancy(A.10)]

    architecture:
      subtree:
      - name: System
        local:
        - name: Compute
          class: compute

    binding:
    - name: Compute
      bindings:
      - einsum: Z
        op: add
    """
    program, hardware, format_ = build_program_hardware(yaml)

    with pytest.raises(NotImplementedError):
        Metrics(program, hardware, format_)


def test_check_configuration_three_tensors():
    yaml = """
    einsum:
      declaration:
        A: [M]
        B: [M]
        C: [M]
        Z: [M]
      expressions:
        - Z[m] = A[m] * B[m] * C[m]

    architecture:
      subtree:
      - name: System
        local:
        - name: Compute
          class: compute

    binding:
    - name: Compute
      bindings:
      - einsum: Z
        op: add
    """
    program, hardware, format_ = build_program_hardware(yaml)

    with pytest.raises(NotImplementedError):
        Metrics(program, hardware, format_)


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
    program, hardware, format_ = build_program_hardware(yaml)

    with pytest.raises(ValueError) as excinfo:
        Metrics(program, hardware, format_)
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
    program, hardware, format_ = build_program_hardware(yaml)

    with pytest.raises(NotImplementedError):
        Metrics(program, hardware, format_)


def test_get_compute_components():
    metrics = build_metrics()
    bindings = Bindings.from_str(build_gamma_yaml())

    intersect = LeaderFollowerComponent(
        "Intersection", {}, bindings.get("Intersection"))

    assert metrics.get_compute_components() == [intersect]


def test_get_format():
    metrics = build_metrics()
    spec = {
        "M": {
            "format": "U",
            "rhbits": 32,
            "pbits": 32},
        "K": {
            "format": "C",
            "cbits": 32,
            "pbits": 64}}
    assert metrics.get_format(Tensor("A", ["M", "K"])) == spec


def test_get_merger_components():
    yaml = build_gamma_yaml()
    program, hardware, format_ = build_program_hardware(yaml)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_merger_components() == []

    bindings = Bindings.from_str(yaml)
    attrs = {"radix": 64, "next_latency": 1}
    merger = MergerComponent(
        "HighRadixMerger",
        attrs,
        bindings.get("HighRadixMerger"))

    binding = bindings.get("HighRadixMerger")[0].copy()
    binding["final_ranks"] = ["M", "N", "K"]

    program.reset()
    program.add_einsum(1)
    metrics = Metrics(program, hardware, format_)

    assert metrics.get_merger_components() == [(merger, binding)]


def test_get_merger_components_output():
    yaml = """
    einsum:
      declaration:
        Z: [M, N]
      expressions:
        - Z[m, n] = a

    mapping:
      rank-order:
        Z: [M, N]
      loop-order:
        Z: [N, M]

    architecture:
      subtree:
      - name: System
        local:
        - name: Merger
          class: Merger

        - name: Compute
          class: compute

    bindings:
    - name: Merger
      bindings:
      - tensor: Z
        init_ranks: [N, M]
        swap_depth: 0

    - name: Compute
      bindings:
      - einsum: Z
        op: add
    """
    program, hardware, format_ = build_program_hardware(yaml)
    metrics = Metrics(program, hardware, format_)

    bindings = Bindings.from_str(yaml)
    merger = MergerComponent("Merger", {}, bindings.get("Merger"))
    binding = bindings.get("Merger")[0].copy()
    binding["final_ranks"] = ["M", "N"]

    assert metrics.get_merger_components() == [(merger, binding)]


def test_get_merger_components_part_merge():
    yaml = """
    einsum:
      declaration:
        A: [K, M]
        Z: [M]
      expressions:
        - Z[m] = sum(K).A[k, m]

    mapping:
      rank-order:
        A: [M, K]

      partitioning:
        Z:
          M: [uniform_shape(10)]
      loop-order:
        Z: [M1, K, M0]

    architecture:
      subtree:
      - name: System
        local:
        - name: Merger
          class: Merger

        - name: Compute
          class: compute

    bindings:
    - name: Merger
      bindings:
      - tensor: A
        init_ranks: [M1, M0, K]
        swap_depth: 1

    - name: Compute
      bindings:
      - einsum: Z
        op: add
    """
    program, hardware, format_ = build_program_hardware(yaml)
    metrics = Metrics(program, hardware, format_)

    bindings = Bindings.from_str(yaml)
    merger = MergerComponent("Merger", {}, bindings.get("Merger"))
    binding = bindings.get("Merger")[0].copy()
    binding["final_ranks"] = ["M1", "K", "M0"]

    assert metrics.get_merger_components() == [(merger, binding)]


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
