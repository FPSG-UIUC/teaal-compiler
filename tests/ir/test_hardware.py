import pytest

from teaal.ir.component import *
from teaal.ir.hardware import Hardware
from teaal.ir.level import Level
from teaal.parse import *


def test_no_arch():
    arch = Architecture.from_str("")
    bindings = Bindings.from_str("")

    with pytest.raises(ValueError) as excinfo:
        Hardware(arch, bindings)
    assert str(excinfo.value) == "Empty architecture specification"


def test_bad_arch():
    yaml = """
    architecture:
      subtree:
      - name: foo
      - name: bar
    """
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)

    with pytest.raises(ValueError) as excinfo:
        Hardware(arch, bindings)
    assert str(excinfo.value) == "Architecture must have a single root level"


def test_bad_component():
    yaml = """
    architecture:
      subtree:
      - name: System
        local:
        - name: BAD
          class: foo
    """
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)

    with pytest.raises(ValueError) as excinfo:
        Hardware(arch, bindings)
    assert str(excinfo.value) == "Unknown class: foo"


def test_no_binding():
    yaml = """
    architecture:
      subtree:
      - name: System
        local:
        - name: Cache
          class: Cache
    """
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    hardware = Hardware(arch, bindings)

    cache = CacheComponent("Cache", {}, [])
    assert hardware.get_component("Cache") == cache


def test_get_component():
    yaml = """
    architecture:
      subtree:
      - name: Base
        local:
        - name: LLB
          class: Buffet

        - name: FiberCache
          class: Cache
          attributes:
            width: 8
            depth: 3145728

        - name: Compute
          class: Compute

        - name: Memory
          class: DRAM
          attributes:
            datawidth: 8
            bandwidth: 128

        - name: LFIntersect
          class: LeaderFollower

        - name: HighRadixMerger
          class: Merger
          attributes:
            radix: 64
            next_latency: 1

        - name: SAIntersect
          class: SkipAhead

    bindings:
      - name: LLB
        bindings:
        - tensor: A
          rank: K2
        - tensor: B
          rank: K2
        - tensor: Z
          rank: N2

      - name: FiberCache
        bindings:
        - tensor: B
          rank: K

      - name: Compute
        bindings:
        - einsum: Z
          op: mul
        - einsum: Z
          op: add

      - name: Memory
        bindings:
        - tensor: A
          rank: root
        - tensor: B
          rank: root
        - tensor: Z
          rank: root

      - name: LFIntersect
        bindings:
        - einsum: T
          rank: K
          leader: A

      - name: HighRadixMerger
        bindings:
        - tensor: T
          init_ranks: [M, K, N]
          swap_depth: 1

      - name: SAIntersect
        bindings:
        - einsum: Z
          rank: K2
    """
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    hardware = Hardware(arch, bindings)

    def assert_component(type_, name, attrs):
        binding = bindings.get(name)
        component = type_(name, attrs, binding)
        assert hardware.get_component(name) == component

    assert_component(BuffetComponent, "LLB", {})

    attrs = {"width": 8, "depth": 3145728}
    assert_component(CacheComponent, "FiberCache", attrs)

    assert_component(FunctionalComponent, "Compute", {})

    attrs = {"datawidth": 8, "bandwidth": 128}
    assert_component(DRAMComponent, "Memory", attrs)

    assert_component(LeaderFollowerComponent, "LFIntersect", {})

    attrs = {"radix": 64, "next_latency": 1}
    assert_component(MergerComponent, "HighRadixMerger", attrs)

    assert_component(SkipAheadComponent, "SAIntersect", {})


def test_bad_compute_path():
    yaml = """
    architecture:
      subtree:
      - name: System

        subtree:
        - name: Stage0
          local:
          - name: BAD0
            class: compute

        - name: Stage1
          local:
          - name: BAD1
            class: compute

    bindings:
      - name: BAD0
        bindings:
        - einsum: Z
          op: mul
      - name: BAD1
        bindings:
        - einsum: Z
          op: add
    """
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    hardware = Hardware(arch, bindings)

    with pytest.raises(ValueError) as excinfo:
        hardware.get_compute_path("Z")
    assert str(excinfo.value) == "Only one compute path allowed per einsum"


def test_get_compute_path():
    arch = Architecture.from_file("tests/integration/test_arch.yaml")
    bindings = Bindings.from_file("tests/integration/test_bindings.yaml")
    hardware = Hardware(arch, bindings)

    system = hardware.get_tree()
    pe = system.get_subtrees()[0]

    assert hardware.get_compute_path("Z") == [system, pe]
    assert hardware.get_compute_path("T") == []


def test_get_compute_components():
    yaml = """
    architecture:
      subtree:
      - name: System

        local:
        - name: Intersect0
          class: SkipAhead

        subtree:
        - name: PE

          local:
          - name: Intersect1
            class: SkipAhead

          - name: MAC
            class: compute

    bindings:
      - name: Intersect0
        bindings:
        - einsum: Z
          rank: K

      - name: Intersect1
        bindings:
        - einsum: X
          rank: J

      - name: MAC
        bindings:
        - einsum: Z
          op: add
    """
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    hardware = Hardware(arch, bindings)

    intersect = SkipAheadComponent(
        "Intersect0", {}, bindings.get("Intersect0"))
    mac = FunctionalComponent("MAC", {}, bindings.get("MAC"))

    assert hardware.get_compute_components("Z") == [intersect, mac]


def test_get_merger_components():
    yaml = """
    architecture:
      subtree:
      - name: System

        subtree:
        - name: SwapStage0
          local:
          - name: Merger0
            class: Merger
            attributes:
              radix: 64
              next_latency: 1

        - name: ComputeStage
          local:
          - name: Compute
            class: compute

        - name: SwapStage1
          local:
          - name: Merger1
            class: Merger
            attributes:
              radix: 64
              next_latency: 1

    bindings:
      - name: Merger0
        bindings:
        - tensor: T
          init_ranks: [M, K, N]
          swap_depth: 1

      - name: Compute
        bindings:
        - einsum: Z
          op: add

      - name: Merger1
        bindings:
        - tensor: Z
          init_ranks: [N, M]
          swap_depth: 0
    """
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    hardware = Hardware(arch, bindings)

    attrs = {"radix": 64, "next_latency": 1}
    merger0 = MergerComponent("Merger0", attrs, bindings.get("Merger0"))
    merger1 = MergerComponent("Merger1", attrs, bindings.get("Merger1"))

    assert hardware.get_merger_components() == [merger0, merger1]


def test_get_traffic_path_multiple_bindings():
    yaml = """
    architecture:
      subtree:
      - name: BAD

        local:
        - name: Memory0
          class: DRAM

        - name: Memory1
          class: DRAM

        - name: Compute
          class: compute

    bindings:
      - name: Memory0
        bindings:
        - tensor: A
          rank: root

      - name: Memory1
        bindings:
        - tensor: A
          rank: root

      - name: Compute
        bindings:
        - einsum: Z
          op: add
    """
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    hardware = Hardware(arch, bindings)

    with pytest.raises(ValueError) as excinfo:
        hardware.get_traffic_path("Z", "A")
    assert str(excinfo.value) == "Multiple bindings for einsum Z and tensor A"


def test_get_traffic_path():
    yaml = """
    architecture:
      subtree:
      - name: System

        local:
        - name: Memory
          class: DRAM

        subtree:
        - name: Stages
          local:
          - name: Intersection
            class: SkipAhead

          - name: LLB
            class: Buffet

          subtree:
          - name: Stage0
            local:
            - name: S0B
              class: Buffet

            - name: MAC0
              class: compute

          - name: Stage1
            local:
            - name: S1B
              class: Buffet

            - name: MAC1
              class: compute

          - name: Stage2
            local:
            - name: S2B
              class: Buffet

            - name: MAC2
              class: compute

    bindings:
      - name: Memory
        bindings:
        - tensor: A
          rank: root
        - tensor: Z
          rank: root

      - name: S0B
        bindings:
        - tensor: A
          rank: M
        - tensor: Z
          rank: M

      - name: MAC0
        bindings:
        - einsum: A
          op: mul

      - name: S1B
        bindings:
        - tensor: Z
          rank: M

      - name: MAC1
        bindings:
        - einsum: X
          op: add

      - name: S2B
        bindings:
        - tensor: A
          rank: M
        - tensor: Z
          rank: M

      - name: MAC2
        bindings:
        - einsum: Z
          op: add
    """
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    hardware = Hardware(arch, bindings)

    mem = DRAMComponent("Memory", {}, bindings.get("Memory"))
    s0b = BuffetComponent("S0B", {}, bindings.get("S0B"))
    s1b = BuffetComponent("S1B", {}, bindings.get("S1B"))
    s2b = BuffetComponent("S2B", {}, bindings.get("S2B"))

    assert hardware.get_traffic_path("A", "A") == [mem, s0b]
    assert hardware.get_traffic_path("Z", "A") == [mem, s2b]
    assert hardware.get_traffic_path("Z", "Z") == [mem, s2b]
    assert hardware.get_traffic_path("X", "B") == []


def test_get_tree():
    arch = Architecture.from_file("tests/integration/test_arch.yaml")
    bindings = Bindings.from_file("tests/integration/test_bindings.yaml")
    hardware = Hardware(arch, bindings)

    regs = BuffetComponent("Registers", {}, bindings.get("Registers"))
    mac = FunctionalComponent("MAC", {}, bindings.get("MAC"))
    pe = Level("PE", 8, {}, [regs, mac], [])

    mem_attrs = {"datawidth": 8, "bandwidth": 128}
    mem = DRAMComponent("Memory", mem_attrs, bindings.get("Memory"))
    attrs = {"clock_frequency": 10 ** 9}

    tree = Level("System", 1, attrs, [mem], [pe])
    assert hardware.get_tree() == tree
