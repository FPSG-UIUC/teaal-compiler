import pytest

from es2hfa.ir.component import *
from es2hfa.ir.hardware import Hardware
from es2hfa.ir.level import Level
from es2hfa.parse.arch import Architecture


def test_bad_arch():
    yaml = """
    architecture:
      subtree:
      - name: foo
      - name: bar
    """
    arch = Architecture.from_str(yaml)

    with pytest.raises(ValueError) as excinfo:
        Hardware(arch.get_spec())
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
        Hardware(arch.get_spec())
    assert str(excinfo.value) == "Unknown class: foo"


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
    """
    arch = Architecture.from_str(yaml)
    hardware = Hardware(arch.get_spec())

    cache = CacheComponent("FiberCache", {"width": 8, "depth": 3145728})
    assert hardware.get_component("FiberCache") == cache

    compute = ComputeComponent("Compute", {})
    assert hardware.get_component("Compute") == compute

    dram = DRAMComponent("Memory", {"datawidth": 8, "bandwidth": 128})
    assert hardware.get_component("Memory") == dram

    leaderfollower = LeaderFollowerComponent("LFIntersect", {})
    assert hardware.get_component("LFIntersect") == leaderfollower

    merger = MergerComponent(
        "HighRadixMerger", {
            "radix": 64, "next_latency": 1})
    assert hardware.get_component("HighRadixMerger") == merger

    skipahead = SkipAheadComponent("SAIntersect", {})
    assert hardware.get_component("SAIntersect") == skipahead

    sram = SRAMComponent("LLB", {})
    assert hardware.get_component("LLB") == sram


def test_tree():
    arch = Architecture.from_file("tests/integration/test_arch.yaml")
    hardware = Hardware(arch.get_spec())

    mac = ComputeComponent("MAC", {})
    pe = Level("PE", 8, {}, [mac], [])

    mem = DRAMComponent("Memory", {"datawidth": 8, "bandwidth": 128})
    attrs = {"clock_frequency": 10 ** 9}

    tree = Level("System", 1, attrs, [mem], [pe])
    assert hardware.get_tree() == tree
