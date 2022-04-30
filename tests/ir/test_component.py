from es2hfa.ir.component import *


def test_component_get_name():
    component = Component("Test", {}, [])
    assert component.get_name() == "Test"


def test_component_eq():
    component0 = Component("Test", {"attr0": 5}, [])
    component1 = Component("Test", {"attr0": 5}, [])

    assert component0 == component1
    assert component0 != "foo"


def test_component_repr():
    component = Component("Test", {"attrs0": 5}, [])
    assert repr(component) == "(Component, Test, {'attrs0': 5}, {})"


def test_component_subclass_repr():
    bindings = [{"einsum": "Z", "op": "add"}, {"einsum": "Z", "op": "mul"}]
    compute = ComputeComponent("MAC", {}, bindings)

    assert repr(
        compute) == "(ComputeComponent, MAC, {}, {'Z': [{'op': 'add'}, {'op': 'mul'}]})"


def test_compute_component():
    bindings = [{"einsum": "Z", "op": "add"}, {"einsum": "Z", "op": "mul"}]
    compute = ComputeComponent("MAC", {}, bindings)

    assert compute.get_bindings("Z") == [{"op": "add"}, {"op": "mul"}]
    assert compute.get_bindings("T") == []


def test_memory_component():
    memory = MemoryComponent("Memory", {}, [{"tensor": "A", "rank": "M"}])

    assert memory.get_binding("A") == "M"
    assert memory.get_binding("B") is None

    assert repr(memory) == "(MemoryComponent, Memory, {}, {'A': 'M'})"


def test_buffet_component():
    bindings = [{"tensor": "A", "rank": "M"}]
    buffet = BuffetComponent("LLB", {}, bindings)


def test_cache_component():
    attrs = {"width": 8, "depth": 3 * 2 ** 20}
    bindings = [{"tensor": "A", "rank": "M"}]
    cache = CacheComponent("FiberCache", attrs, bindings)

    assert cache.get_depth() == 3 * 2 ** 20
    assert cache.get_width() == 8


def test_dram_component():
    bindings = [{"tensor": "A", "rank": "M"}]
    dram = DRAMComponent("DRAM", {"datawidth": 8, "bandwidth": 128}, bindings)

    assert dram.get_bandwidth() == 128
    assert dram.get_datawidth() == 8


def test_leader_follower_component():
    bindings = [{"einsum": "Z", "rank": "K"}]
    leader_follower = LeaderFollowerComponent("Intersection", {}, bindings)


def test_merger_component():
    attrs = {"radix": 64, "next_latency": 1}
    binding = [{"tensor": "T", "init_ranks": ["M", "K", "N"], "swap_depth": 1}]
    merger = MergerComponent("HighRadixMerger", attrs, binding)

    assert merger.get_bindings(
        "T") == [{"init_ranks": ["M", "K", "N"], "swap_depth": 1}]
    assert merger.get_bindings("A") == []

    assert merger.get_next_latency() == 1
    assert merger.get_radix() == 64

    merger = MergerComponent(
        "Sort", {"radix": "inf", "next_latency": "N"}, binding)

    assert merger.get_next_latency() == "N"
    assert merger.get_radix() == float("inf")


def test_skip_ahead_component():
    bindings = [{"einsum": "Z", "rank": "K2"}]
    skip_ahead = SkipAheadComponent("K2Intersection", {}, bindings)
