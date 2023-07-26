import pytest

from teaal.ir.component import *


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
    assert repr(component) == "(Component, Test, {})"


def test_functional_component():
    bindings = [{"einsum": "Z", "op": "add"}, {"einsum": "T", "op": "mul"}]
    compute = FunctionalComponent("MAC", {}, bindings)

    assert compute.get_bindings("Z") == [{"op": "add"}]
    assert compute.get_bindings("T") == [{"op": "mul"}]

    assert repr(compute) in {
        "(FunctionalComponent, MAC, {'T': [{'op': 'mul'}], 'Z': [{'op': 'add'}]})",
        "(FunctionalComponent, MAC, {'Z': [{'op': 'add'}], 'T': [{'op': 'mul'}]})"}


def test_memory_attr_errs():
    with pytest.raises(ValueError) as excinfo:
        MemoryComponent("Mem", {"bandwidth": "foo"}, [])
    assert str(excinfo.value) == "Bad bandwidth foo for Memory Mem"

    memory = MemoryComponent("Mem", {}, [])
    with pytest.raises(ValueError) as excinfo:
        memory.get_bandwidth()
    assert str(excinfo.value) == "Bandwidth unspecified for component Mem"


def test_memory_component():
    memory = MemoryComponent("Memory", {"bandwidth": 256}, [
                             {"tensor": "A", "rank": "M"}])

    assert memory.get_bandwidth() == 256

    assert memory.get_binding("A") == "M"
    assert memory.get_binding("B") is None

    assert repr(memory) == "(MemoryComponent, Memory, {'A': 'M'}, 256)"


def test_buffer_attr_errs():
    buffer_ = BufferComponent("Buf", {"width": 8}, [])
    with pytest.raises(ValueError) as excinfo:
        buffer_.get_depth()
    assert str(excinfo.value) == "Depth unspecified for component Buf"

    with pytest.raises(ValueError) as excinfo:
        BufferComponent("Buf", {"depth": "foo", "width": 8}, [])
    assert str(excinfo.value) == "Bad depth foo for Buffer Buf"

    buffer_ = BufferComponent("Buf", {"depth": 256}, [])
    with pytest.raises(ValueError) as excinfo:
        buffer_.get_width()
    assert str(excinfo.value) == "Width unspecified for component Buf"

    with pytest.raises(ValueError) as excinfo:
        BufferComponent("Buf", {"depth": 256, "width": "foo"}, [])
    assert str(excinfo.value) == "Bad width foo for Buffer Buf"


def test_buffer_component():
    attrs = {"width": 8, "depth": 3 * 2 ** 20}
    buffer_ = BufferComponent("Buf", attrs, [])

    assert buffer_.get_width() == 8
    assert buffer_.get_depth() == 3 * 2 ** 20

    assert repr(buffer_) == "(BufferComponent, Buf, {}, None, 3145728, 8)"


def test_buffet_component():
    attrs = {"width": 8, "depth": 3 * 2 ** 20}
    bindings = [{"tensor": "A", "rank": "M"}]
    buffet = BuffetComponent("LLB", attrs, bindings)


def test_cache_component():
    attrs = {"width": 8, "depth": 3 * 2 ** 20}
    bindings = [{"tensor": "A", "rank": "M"}]
    cache = CacheComponent("FiberCache", attrs, bindings)


def test_compute_attr_errs():
    with pytest.raises(ValueError) as excinfo:
        ComputeComponent("FU", {}, [])
    assert str(excinfo.value) == "Type unspecified for component FU"

    with pytest.raises(ValueError) as excinfo:
        ComputeComponent("FU", {"type": None}, [])
    assert str(excinfo.value) == "Bad type None for Compute FU"

    with pytest.raises(ValueError) as excinfo:
        ComputeComponent("FU", {"type": "foo"}, [])
    assert str(
        excinfo.value) in {
        "foo is not a valid value for attribute type of class Compute. Choose one of {'mul', 'add'}",
        "foo is not a valid value for attribute type of class Compute. Choose one of {'add', 'mul'}"}


def test_compute_component():
    attrs = {"type": "mul"}
    compute = ComputeComponent("FU", attrs, [])

    assert compute.get_type() == "mul"

    assert repr(compute) == "(ComputeComponent, FU, {}, mul)"


def test_dram_component():
    bindings = [{"tensor": "A", "rank": "M"}]
    dram = DRAMComponent("DRAM", {"datawidth": 8, "bandwidth": 128}, bindings)


def test_leader_follower_component():
    bindings = [{"einsum": "Z", "rank": "K"}]
    leader_follower = LeaderFollowerComponent("Intersection", {}, bindings)


def test_merger_attr_errs():
    attrs = {
        "comparator_radix": 32,
        "outputs": 2,
        "order": "opt",
        "reduce": False}
    with pytest.raises(ValueError) as excinfo:
        MergerComponent("Merger0", attrs, [])
    assert str(excinfo.value) == "Inputs unspecified for component Merger0"

    attrs = {
        "inputs": "foo",
        "comparator_radix": 32,
        "outputs": 2,
        "order": "opt",
        "reduce": False}
    with pytest.raises(ValueError) as excinfo:
        MergerComponent("Merger1", attrs, [])
    assert str(excinfo.value) == "Bad inputs foo for Merger Merger1"

    attrs = {"inputs": 64, "outputs": 2, "order": "opt", "reduce": False}
    with pytest.raises(ValueError) as excinfo:
        MergerComponent("Merger0", attrs, [])
    assert str(
        excinfo.value) == "Comparator radix unspecified for component Merger0"

    attrs = {
        "inputs": 64,
        "comparator_radix": "foo",
        "outputs": 2,
        "order": "opt",
        "reduce": False}
    with pytest.raises(ValueError) as excinfo:
        MergerComponent("Merger1", attrs, [])
    assert str(excinfo.value) == "Bad comparator_radix foo for Merger Merger1"

    attrs = {
        "inputs": 64,
        "comparator_radix": 32,
        "outputs": "foo",
        "order": "opt",
        "reduce": False}
    with pytest.raises(ValueError) as excinfo:
        MergerComponent("Merger1", attrs, [])
    assert str(excinfo.value) == "Bad outputs foo for Merger Merger1"

    attrs = {
        "inputs": 64,
        "comparator_radix": 32,
        "outputs": 2,
        "order": None,
        "reduce": False}
    with pytest.raises(ValueError) as excinfo:
        MergerComponent("Merger1", attrs, [])
    assert str(excinfo.value) == "Bad order None for Merger Merger1"

    attrs = {
        "inputs": 64,
        "comparator_radix": 32,
        "outputs": 2,
        "order": "foo",
        "reduce": False}
    with pytest.raises(ValueError) as excinfo:
        MergerComponent("Merger1", attrs, [])
    assert str(
        excinfo.value) in {
        "foo is not a valid value for attribute order of class Merger. Choose one of {'opt', 'fifo'}",
        "foo is not a valid value for attribute order of class Merger. Choose one of {'fifo', 'opt'}"}

    attrs = {
        "inputs": 64,
        "comparator_radix": 32,
        "outputs": 2,
        "order": "opt",
        "reduce": 2}
    with pytest.raises(ValueError) as excinfo:
        MergerComponent("Merger1", attrs, [])
    assert str(excinfo.value) == "Bad reduce 2 for Merger Merger1"

    attrs = {
        "inputs": 64,
        "comparator_radix": 32,
        "outputs": 2,
        "order": "opt",
        "reduce": True}
    with pytest.raises(NotImplementedError) as excinfo:
        MergerComponent("Merger1", attrs, [])
    assert str(excinfo.value) == "Concurrent merge and reduction not supported"


def test_merger_component():
    attrs = {
        "inputs": 64,
        "comparator_radix": 32,
        "outputs": 2,
        "order": "opt",
        "reduce": False}
    binding = [{"tensor": "T", "init_ranks": ["M", "K", "N"], "swap_depth": 1}]
    merger = MergerComponent("Merger0", attrs, binding)

    bindings = [{"tensor": "T", "init_ranks": [
        "M", "K", "N"], "final_ranks": ["M", "N", "K"], "swap_depth": 1}]
    assert merger.get_bindings() == bindings

    assert merger.get_inputs() == 64
    assert merger.get_comparator_radix() == 32
    assert merger.get_outputs() == 2
    assert merger.get_order() == "opt"
    assert merger.get_reduce() == False

    assert repr(
        merger) == "(MergerComponent, Merger0, [{'tensor': 'T', 'init_ranks': ['M', 'K', 'N'], 'swap_depth': 1, 'final_ranks': ['M', 'N', 'K']}], 64, 32, 2, opt, False)"

    attrs = {"inputs": 200, "comparator_radix": 2}
    merger = MergerComponent("Merger1", attrs, binding)

    assert merger.get_inputs() == 200
    assert merger.get_comparator_radix() == 2
    assert merger.get_outputs() == 1
    assert merger.get_order() == "fifo"
    assert merger.get_reduce() == False


def test_skip_ahead_component():
    bindings = [{"einsum": "Z", "rank": "K2"}]
    skip_ahead = SkipAheadComponent("K2Intersection", {}, bindings)
