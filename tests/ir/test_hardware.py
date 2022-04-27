import pytest

from es2hfa.ir.component import *
from es2hfa.ir.hardware import Hardware
from es2hfa.ir.level import Level
from es2hfa.parse.arch import Architecture
from es2hfa.parse.bindings import Bindings


def test_bad_arch():
    yaml = """
    architecture:
      subtree:
      - name: foo
      - name: bar
    """
    arch = Architecture.from_str(yaml)

    with pytest.raises(ValueError) as excinfo:
        Hardware(arch.get_spec(), [])
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

    with pytest.raises(ValueError) as excinfo:
        Hardware(arch.get_spec(), [])
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
    arch = Architecture.from_str(yaml).get_spec()
    bindings = Bindings.from_str(yaml).get_components()
    hardware = Hardware(arch, bindings)

    cache = CacheComponent("Cache", {}, [])
    assert hardware.get_component("Cache") == cache


def test_all_components():
    yaml = """
    architecture:
      subtree:
      - name: Base
        local:
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

        - name: LLB
          class: SRAM
    bindings:
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

      - name: LLB
        bindings:
        - tensor: A
          rank: K2
        - tensor: B
          rank: K2
        - tensor: Z
          rank: N2
    """
    arch = Architecture.from_str(yaml).get_spec()
    bindings = Bindings.from_str(yaml).get_components()
    hardware = Hardware(arch, bindings)

    def assert_component(type_, name, attrs):
        component = type_(name, attrs, bindings[name])
        assert hardware.get_component(name) == component

    attrs = {"width": 8, "depth": 3145728}
    assert_component(CacheComponent, "FiberCache", attrs)

    assert_component(ComputeComponent, "Compute", {})

    attrs = {"datawidth": 8, "bandwidth": 128}
    assert_component(DRAMComponent, "Memory", attrs)

    assert_component(LeaderFollowerComponent, "LFIntersect", {})

    attrs = {"radix": 64, "next_latency": 1}
    assert_component(MergerComponent, "HighRadixMerger", attrs)

    assert_component(SkipAheadComponent, "SAIntersect", {})

    assert_component(SRAMComponent, "LLB", {})


def test_tree():
    arch = Architecture.from_file("tests/integration/test_arch.yaml")
    bindings = Bindings.from_file(
        "tests/integration/test_bindings.yaml").get_components()
    hardware = Hardware(arch.get_spec(), bindings)

    mac = ComputeComponent("MAC", {}, bindings["MAC"])
    pe = Level("PE", 8, {}, [mac], [])

    mem = DRAMComponent(
        "Memory", {
            "datawidth": 8, "bandwidth": 128}, bindings["Memory"])
    attrs = {"clock_frequency": 10 ** 9}

    tree = Level("System", 1, attrs, [mem], [pe])
    assert hardware.get_tree() == tree
