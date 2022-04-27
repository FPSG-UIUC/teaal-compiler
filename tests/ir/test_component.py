from es2hfa.ir.component import *


def test_component_get_name():
    component = Component("Test", {})
    assert component.get_name() == "Test"


def test_component_eq():
    component0 = Component("Test", {"attr0": 5})
    component1 = Component("Test", {"attr0": 5})

    assert component0 == component1
    assert component0 != "foo"


def test_component_repr():
    component = Component("Test", {"attrs0": 5})
    assert repr(component) == "(Component, Test, {'attrs0': 5})"


def test_component_subclass_repr():
    compute = ComputeComponent("MAC", {})
    assert repr(compute) == "(ComputeComponent, MAC, {})"


def test_cache_component():
    cache = CacheComponent("FiberCache", {"width": 8, "depth": 3 * 2 ** 20})

    assert cache.get_depth() == 3 * 2 ** 20
    assert cache.get_width() == 8


def test_compute_component():
    compute = ComputeComponent("MAC", {})


def test_dram_component():
    dram = DRAMComponent("DRAM", {"datawidth": 8, "bandwidth": 128})

    assert dram.get_bandwidth() == 128
    assert dram.get_datawidth() == 8


def test_leader_follower_component():
    leader_follower = LeaderFollowerComponent("Intersection", {})


def test_merger_component():
    merger = MergerComponent(
        "HighRadixMerger", {
            "radix": 64, "next_latency": 1})

    assert merger.get_next_latency() == 1
    assert merger.get_radix() == 64

    merger = MergerComponent("Sort", {"radix": "inf", "next_latency": "N"})

    assert merger.get_next_latency() == "N"
    assert merger.get_radix() == float("inf")


def test_skip_ahead_component():
    skip_ahead = SkipAheadComponent("K2Intersection", {})


def test_sram_component():
    sram = SRAMComponent("LLB", {})
