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
    component = Component("Test", {"attrs0": 5}, {"Z": [{"foo": "bar"}]})
    assert repr(component) == "(Component, Test, {'Z': [{'foo': 'bar'}]})"


def test_functional_component():
    bindings = {"Z": [{"op": "add"}], "T": [{"op": "mul"}]}
    compute = FunctionalComponent("MAC", {}, bindings)

    assert compute.get_bindings() == bindings

    assert repr(compute) in {
        "(FunctionalComponent, MAC, {'T': [{'op': 'mul'}], 'Z': [{'op': 'add'}]})",
        "(FunctionalComponent, MAC, {'Z': [{'op': 'add'}], 'T': [{'op': 'mul'}]})"}


def test_memory_attr_errs():
    with pytest.raises(ValueError) as excinfo:
        MemoryComponent("Mem", {"bandwidth": "foo"}, {})
    assert str(excinfo.value) == "Bad bandwidth foo for Memory Mem"

    memory = MemoryComponent("Mem", {}, {})
    with pytest.raises(ValueError) as excinfo:
        memory.get_bandwidth()
    assert str(excinfo.value) == "Bandwidth unspecified for component Mem"


def test_memory_binding_errs():
    binding = {"Z": [{"rank": "M", "type": "elem", "format": "default"}]}
    with pytest.raises(ValueError) as excinfo:
        MemoryComponent("Mem", {"bandwidth": 256}, binding)
    assert str(
        excinfo.value) == "Tensor not specified for Einsum Z in binding to Mem"

    binding = {"Z": [{"tensor": "A", "type": "elem", "format": "default"}]}
    with pytest.raises(ValueError) as excinfo:
        MemoryComponent("Mem", {"bandwidth": 256}, binding)
    assert str(
        excinfo.value) == "Rank not specified for tensor A in Einsum Z in binding to Mem"

    binding = {"Z": [{"tensor": "A", "rank": "M", "format": "default"}]}
    with pytest.raises(ValueError) as excinfo:
        MemoryComponent("Mem", {"bandwidth": 256}, binding)
    assert str(
        excinfo.value) == "Type not specified for tensor A in Einsum Z in binding to Mem"

    binding = {"Z": [{"tensor": "A", "rank": "M",
                      "type": "foo", "format": "default"}]}
    with pytest.raises(ValueError) as excinfo:
        MemoryComponent("Mem", {"bandwidth": 256}, binding)
    assert str(
        excinfo.value) in {
        "Type foo for Mem on tensor A in Einsum Z not one of {'coord', 'elem', 'payload'}",
        "Type foo for Mem on tensor A in Einsum Z not one of {'coord', 'payload', 'elem'}",
        "Type foo for Mem on tensor A in Einsum Z not one of {'payload', 'coord', 'elem'}",
        "Type foo for Mem on tensor A in Einsum Z not one of {'payload', 'elem', 'coord'}",
        "Type foo for Mem on tensor A in Einsum Z not one of {'elem', 'coord', 'payload'}",
        "Type foo for Mem on tensor A in Einsum Z not one of {'elem', 'payload', 'coord'}"}

    binding = {"Z": [{"tensor": "A", "rank": "M", "type": "elem"}]}
    with pytest.raises(ValueError) as excinfo:
        MemoryComponent("Mem", {"bandwidth": 256}, binding)
    assert str(
        excinfo.value) == "Format not specified for tensor A in Einsum Z in binding to Mem"

    bindings = {"Z": [{"tensor": "A", "rank": "M", "type": "payload", "format": "default"},
                      {"tensor": "A", "rank": "M", "type": "payload", "format": "default"}]}
    memory = MemoryComponent("Memory", {"bandwidth": 256}, bindings)
    with pytest.raises(ValueError) as excinfo:
        memory.get_binding("Z", "A", "M", "payload", "default")
    assert str(
        excinfo.value) == "Multiple bindings for [('einsum', 'Z'), ('tensor', 'A'), ('rank', 'M'), ('type', 'payload'), ('format', 'default')]"


def test_memory_component():
    bindings = {"Z": [{"tensor": "A", "rank": "M",
                       "type": "payload", "format": "default"}]}
    memory = MemoryComponent("Memory", {"bandwidth": 256}, bindings)

    assert memory.get_bandwidth() == 256

    assert memory.get_binding(
        "Z",
        "A",
        "M",
        "payload",
        "default") == bindings["Z"][0]
    assert memory.get_binding("Z", "B", "M", "payload", "default") is None
    assert memory.get_binding("T", "A", "M", "payload", "default") is None

    assert repr(
        memory) == "(MemoryComponent, Memory, {'Z': [{'tensor': 'A', 'rank': 'M', 'type': 'payload', 'format': 'default'}]}, 256)"


def test_buffer_attr_errs():
    buffer_ = BufferComponent("Buf", {"width": 8}, {})
    with pytest.raises(ValueError) as excinfo:
        buffer_.get_depth()
    assert str(excinfo.value) == "Depth unspecified for component Buf"

    with pytest.raises(ValueError) as excinfo:
        BufferComponent("Buf", {"depth": "foo", "width": 8}, {})
    assert str(excinfo.value) == "Bad depth foo for Buffer Buf"

    buffer_ = BufferComponent("Buf", {"depth": 256}, {})
    with pytest.raises(ValueError) as excinfo:
        buffer_.get_width()
    assert str(excinfo.value) == "Width unspecified for component Buf"

    with pytest.raises(ValueError) as excinfo:
        BufferComponent("Buf", {"depth": 256, "width": "foo"}, {})
    assert str(excinfo.value) == "Bad width foo for Buffer Buf"


def test_buffer_component():
    attrs = {"width": 8, "depth": 3 * 2 ** 20}
    buffer_ = BufferComponent("Buf", attrs, {})

    assert buffer_.get_width() == 8
    assert buffer_.get_depth() == 3 * 2 ** 20

    assert repr(buffer_) == "(BufferComponent, Buf, {}, None, 3145728, 8)"


def test_buffet_binding_errs():
    attrs = {"width": 8, "depth": 3 * 2 ** 20}

    bindings = {"Z": [{"tensor": "A", "rank": "M",
                       "type": "payload", "format": "default", "style": "foo"}]}
    with pytest.raises(ValueError) as excinfo:
        BuffetComponent("LLB", attrs, bindings)
    assert str(
        excinfo.value) == "Evict-on not specified for tensor A in Einsum Z in binding to LLB"

    bindings = {"Z": [{"tensor": "A",
                       "rank": "M",
                       "type": "payload",
                       "format": "default",
                       "style": "foo",
                       "evict-on": "root"}]}
    with pytest.raises(ValueError) as excinfo:
        BuffetComponent("LLB", attrs, bindings)
    assert str(
        excinfo.value) in {
        "Style foo for LLB on tensor A in Einsum Z not one of {'eager', 'lazy'}",
        "Style foo for LLB on tensor A in Einsum Z not one of {'lazy', 'eager'}"}


def test_buffet_component():
    attrs = {"width": 8, "depth": 3 * 2 ** 20}
    bindings = {"Z": [{"tensor": "A",
                       "rank": "M",
                       "type": "payload",
                       "format": "default",
                       "style": "eager",
                       "evict-on": "root"}]}
    buffet = BuffetComponent("LLB", attrs, bindings)

    assert buffet.get_binding(
        "Z",
        "A",
        "M",
        "payload",
        "default") == bindings["Z"][0]

    bindings = {"Z": [{"tensor": "A",
                       "rank": "M",
                       "type": "payload",
                       "format": "default",
                       "evict-on": "root"}]}
    buffet = BuffetComponent("LLB", attrs, bindings)

    bindings_corr = {"tensor": "A",
                     "rank": "M",
                     "type": "payload",
                     "format": "default",
                     "style": "lazy",
                     "evict-on": "root"}

    assert buffet.get_binding(
        "Z",
        "A",
        "M",
        "payload",
        "default") == bindings_corr


def test_cache_component():
    attrs = {"width": 8, "depth": 3 * 2 ** 20}
    bindings = {"Z": [{"tensor": "A", "rank": "M",
                       "type": "payload", "format": "default"}]}
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
    compute = ComputeComponent("FU", attrs, {})

    assert compute.get_type() == "mul"

    assert repr(compute) == "(ComputeComponent, FU, {}, mul)"


def test_dram_component():
    bindings = {"Z": [{"tensor": "A", "rank": "M",
                       "type": "payload", "format": "default"}]}
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

def test_merger_binding_errs():
    attrs = {
        "inputs": 64,
        "comparator_radix": 32,
        "outputs": 2,
        "order": "opt",
        "reduce": False}
    binding = {"Z": [{"init-ranks": ["M", "K", "N"], "final-ranks": ["M", "N", "K"]}]}
    with pytest.raises(ValueError) as excinfo:
        MergerComponent("Merger1", attrs, binding)
    assert str(excinfo.value) == "Tensor not specified for Einsum Z in binding to Merger1"

    binding = {"Z": [{"tensor": "T", "final-ranks": ["M", "N", "K"]}]}
    with pytest.raises(ValueError) as excinfo:
        MergerComponent("Merger1", attrs, binding)
    assert str(excinfo.value) == "Initial ranks not specified for tensor T in Einsum Z in binding to Merger1"

    binding = {"Z": [{"tensor": "T", "init-ranks": ["M", "N", "K"]}]}
    with pytest.raises(ValueError) as excinfo:
        MergerComponent("Merger1", attrs, binding)
    assert str(excinfo.value) == "Final ranks not specified for tensor T in Einsum Z in binding to Merger1"

def test_merger_component():
    attrs = {
        "inputs": 64,
        "comparator_radix": 32,
        "outputs": 2,
        "order": "opt",
        "reduce": False}
    binding = {"Z": [{"tensor": "T", "init-ranks": [
        "M", "K", "N"], "final-ranks": ["M", "N", "K"]}]}
    merger = MergerComponent("Merger0", attrs, binding)

    bindings = {"Z": [{"tensor": "T", "init-ranks": [
        "M", "K", "N"], "final-ranks": ["M", "N", "K"]}]}
    assert merger.get_bindings() == bindings

    assert merger.get_inputs() == 64
    assert merger.get_comparator_radix() == 32
    assert merger.get_outputs() == 2
    assert merger.get_order() == "opt"
    assert merger.get_reduce() == False

    assert merger.get_init_ranks("Z", "T", ["M", "N", "K"]) == ["M", "K", "N"]
    assert merger.get_init_ranks("T", "T", ["M", "K", "N"]) is None
    assert merger.get_init_ranks("Z", "A", ["M", "K"]) is None

    assert repr(
        merger) == "(MergerComponent, Merger0, {'Z': [{'tensor': 'T', 'init-ranks': ['M', 'K', 'N'], 'final-ranks': ['M', 'N', 'K']}]}, 64, 32, 2, opt, False)"

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
