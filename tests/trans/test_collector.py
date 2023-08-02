import pytest

from teaal.ir.hardware import Hardware
from teaal.ir.metrics import Metrics
from teaal.ir.program import Program
from teaal.parse import *
from teaal.trans.collector import Collector


def build_gamma_yaml():
    with open("tests/integration/gamma.yaml", "r") as f:
        return f.read()


def build_collector(yaml, i):
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)

    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    hardware = Hardware(arch, bindings, program)

    format_ = Format.from_str(yaml)

    program.add_einsum(i)
    metrics = Metrics(program, hardware, format_)
    return Collector(program, metrics)


def test_dump_T():
    yaml = build_gamma_yaml()
    collector = build_collector(yaml, 0)

    hifiber = "metrics = {}\n" + \
        "metrics[\"T\"] = {}\n" + \
        "formats = {\"A\": Format(A_MK, {\"rank-order\": [\"M\", \"K\"], \"M\": {\"format\": \"U\", \"pbits\": 32}, \"K\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}}), \"B\": Format(B_KN, {\"rank-order\": [\"K\", \"N\"], \"K\": {\"format\": \"U\", \"pbits\": 32}, \"N\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}})}\n" + \
        "bindings = [{\"tensor\": \"B\", \"rank\": \"K\", \"type\": \"payload\", \"format\": \"default\"}, {\"tensor\": \"B\", \"rank\": \"N\", \"type\": \"coord\", \"format\": \"default\"}, {\"tensor\": \"B\", \"rank\": \"N\", \"type\": \"payload\", \"format\": \"default\"}]\n" + \
        "Traffic.filterTrace(\"tmp/gamma_T-K-intersect_3.csv\", \"tmp/gamma_T-K-iter.csv\", \"tmp/gamma_T-K-intersect_3_payload.csv\")\n" + \
        "Traffic.filterTrace(\"tmp/gamma_T-N-populate_1.csv\", \"tmp/gamma_T-N-iter.csv\", \"tmp/gamma_T-N-populate_1_payload.csv\")\n" + \
        "traces = {(\"B\", \"K\", \"payload\", \"read\"): \"tmp/gamma_T-K-intersect_3_payload.csv\", (\"B\", \"N\", \"coord\", \"read\"): \"tmp/gamma_T-N-populate_1.csv\", (\"B\", \"N\", \"payload\", \"read\"): \"tmp/gamma_T-N-populate_1_payload.csv\"}\n" + \
        "traffic = Traffic.cacheTraffic(\"bindings\", \"formats\", \"traces\", 25165824, 64)\n" + \
        "metrics[\"T\"][\"MainMemory\"] = {}\n" + \
        "metrics[\"T\"][\"MainMemory\"][\"B\"] = {}\n" + \
        "metrics[\"T\"][\"MainMemory\"][\"B\"][\"read\"] = 0\n" + \
        "metrics[\"T\"][\"MainMemory\"][\"B\"][\"read\"] += traffic[\"B\"][\"read\"]\n" + \
        "bindings = [{\"tensor\": \"A\", \"rank\": \"M\", \"type\": \"payload\", \"format\": \"default\", \"evict-on\": \"root\", \"style\": \"lazy\"}, {\"tensor\": \"A\", \"rank\": \"K\", \"type\": \"coord\", \"format\": \"default\", \"evict-on\": \"M\", \"style\": \"lazy\"}, {\"tensor\": \"A\", \"rank\": \"K\", \"type\": \"payload\", \"format\": \"default\", \"evict-on\": \"M\", \"style\": \"lazy\"}]\n" + \
        "Traffic.filterTrace(\"tmp/gamma_T-M-populate_1.csv\", \"tmp/gamma_T-M-iter.csv\", \"tmp/gamma_T-M-populate_1_payload.csv\")\n" + \
        "Traffic.filterTrace(\"tmp/gamma_T-K-intersect_2.csv\", \"tmp/gamma_T-K-iter.csv\", \"tmp/gamma_T-K-intersect_2_payload.csv\")\n" + \
        "traces = {(\"A\", \"M\", \"payload\", \"read\"): \"tmp/gamma_T-M-populate_1_payload.csv\", (\"A\", \"K\", \"coord\", \"read\"): \"tmp/gamma_T-K-intersect_2.csv\", (\"A\", \"K\", \"payload\", \"read\"): \"tmp/gamma_T-K-intersect_2_payload.csv\"}\n" + \
        "traffic = Traffic.buffetTraffic(\"bindings\", \"formats\", \"traces\", float(\"inf\"), 64)\n" + \
        "metrics[\"T\"][\"MainMemory\"][\"A\"] = {}\n" + \
        "metrics[\"T\"][\"MainMemory\"][\"A\"][\"read\"] = 0\n" + \
        "metrics[\"T\"][\"MainMemory\"][\"A\"][\"read\"] += traffic[\"A\"][\"read\"]"

    assert collector.dump().gen(0) == hifiber


def test_dump_Z():
    yaml = build_gamma_yaml()
    collector = build_collector(yaml, 1)

    hifiber = "metrics[\"Z\"] = {}\n" + \
        "formats = {\"Z\": Format(Z_MN, {\"rank-order\": [\"M\", \"N\"], \"M\": {\"format\": \"U\", \"pbits\": 32}, \"N\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}}), \"A\": Format(A_MK, {\"rank-order\": [\"M\", \"K\"], \"M\": {\"format\": \"U\", \"pbits\": 32}, \"K\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}})}\n" + \
        "bindings = [{\"tensor\": \"A\", \"rank\": \"M\", \"type\": \"payload\", \"format\": \"default\", \"evict-on\": \"root\", \"style\": \"lazy\"}, {\"tensor\": \"A\", \"rank\": \"K\", \"type\": \"coord\", \"format\": \"default\", \"evict-on\": \"M\", \"style\": \"lazy\"}, {\"tensor\": \"A\", \"rank\": \"K\", \"type\": \"payload\", \"format\": \"default\", \"evict-on\": \"M\", \"style\": \"lazy\"}]\n" + \
        "Traffic.filterTrace(\"tmp/gamma_Z-M-intersect_3.csv\", \"tmp/gamma_Z-M-iter.csv\", \"tmp/gamma_Z-M-intersect_3_payload.csv\")\n" + \
        "Traffic.filterTrace(\"tmp/gamma_Z-K-intersect_1.csv\", \"tmp/gamma_Z-K-iter.csv\", \"tmp/gamma_Z-K-intersect_1_payload.csv\")\n" + \
        "traces = {(\"A\", \"M\", \"payload\", \"read\"): \"tmp/gamma_Z-M-intersect_3_payload.csv\", (\"A\", \"K\", \"coord\", \"read\"): \"tmp/gamma_Z-K-intersect_1.csv\", (\"A\", \"K\", \"payload\", \"read\"): \"tmp/gamma_Z-K-intersect_1_payload.csv\"}\n" + \
        "traffic = Traffic.buffetTraffic(\"bindings\", \"formats\", \"traces\", float(\"inf\"), 64)\n" + \
        "bindings = [{\"tensor\": \"Z\", \"rank\": \"M\", \"type\": \"payload\", \"format\": \"default\", \"evict-on\": \"root\", \"style\": \"lazy\"}, {\"tensor\": \"Z\", \"rank\": \"N\", \"type\": \"coord\", \"format\": \"default\", \"evict-on\": \"M\", \"style\": \"lazy\"}, {\"tensor\": \"Z\", \"rank\": \"N\", \"type\": \"payload\", \"format\": \"default\", \"evict-on\": \"M\", \"style\": \"lazy\"}]\n" + \
        "Traffic.filterTrace(\"tmp/gamma_Z-M-populate_read_0.csv\", \"tmp/gamma_Z-M-iter.csv\", \"tmp/gamma_Z-M-populate_read_0_payload.csv\")\n" + \
        "Traffic.filterTrace(\"tmp/gamma_Z-M-populate_write_0.csv\", \"tmp/gamma_Z-M-iter.csv\", \"tmp/gamma_Z-M-populate_write_0_payload.csv\")\n" + \
        "Traffic.filterTrace(\"tmp/gamma_Z-N-populate_read_0.csv\", \"tmp/gamma_Z-N-iter.csv\", \"tmp/gamma_Z-N-populate_read_0_payload.csv\")\n" + \
        "Traffic.filterTrace(\"tmp/gamma_Z-N-populate_write_0.csv\", \"tmp/gamma_Z-N-iter.csv\", \"tmp/gamma_Z-N-populate_write_0_payload.csv\")\n" + \
        "traces = {(\"Z\", \"M\", \"payload\", \"read\"): \"tmp/gamma_Z-M-populate_read_0_payload.csv\", (\"Z\", \"M\", \"payload\", \"write\"): \"tmp/gamma_Z-M-populate_write_0_payload.csv\", (\"Z\", \"N\", \"coord\", \"read\"): \"tmp/gamma_Z-N-populate_read_0.csv\", (\"Z\", \"N\", \"coord\", \"write\"): \"tmp/gamma_Z-N-populate_write_0.csv\", (\"Z\", \"N\", \"payload\", \"read\"): \"tmp/gamma_Z-N-populate_read_0_payload.csv\", (\"Z\", \"N\", \"payload\", \"write\"): \"tmp/gamma_Z-N-populate_write_0_payload.csv\"}\n" + \
        "traffic = Traffic.buffetTraffic(\"bindings\", \"formats\", \"traces\", float(\"inf\"), 64)\n" + \
        "metrics[\"Z\"][\"MainMemory\"] = {}\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"] = {}\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"read\"] = 0\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"write\"] = 0\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"read\"] += traffic[\"Z\"][\"read\"]\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"write\"] += traffic[\"Z\"][\"write\"]\n" + \
        "metrics[\"Z\"][\"HighRadixMerger\"] = {}\n" + \
        "metrics[\"Z\"][\"HighRadixMerger\"][\"T_MKN\"] = Compute.numSwaps(T_MKN, 1, 64, 1)"

    # print(collector.dump().gen(0))
    # assert False

    assert collector.dump().gen(0) == hifiber


def test_end():
    hifiber = "Metrics.endCollect()"

    assert Collector.end().gen(0) == hifiber


def test_set_collecting_type_err():
    yaml = build_gamma_yaml()
    collector = build_collector(yaml, 0)

    with pytest.raises(ValueError) as excinfo:
        collector.set_collecting(None, "K", "fiber", False, True)
    assert str(
        excinfo.value) == "Tensor must be specified for trace type fiber"

    with pytest.raises(ValueError) as excinfo:
        collector.set_collecting("A", "K", "iter", False, True)
    assert str(
        excinfo.value) == "Unable to collect iter traces for a specific tensor A"


def test_set_collecting():
    yaml = build_gamma_yaml()
    collector = build_collector(yaml, 0)

    hifiber = "Metrics.trace(\"K\", type_=\"intersect_3\", consumable=False)"
    assert collector.set_collecting(
        "B", "K", "fiber", False, True).gen(0) == hifiber

    hifiber = "Metrics.trace(\"K\", type_=\"iter\", consumable=False)"
    assert collector.set_collecting(
        None, "K", "iter", False, True).gen(0) == hifiber


def test_start():
    yaml = build_gamma_yaml()
    collector = build_collector(yaml, 0)
    hifiber = "Metrics.beginCollect([\"M\", \"K\", \"N\"])"

    assert collector.start().gen(0) == hifiber
