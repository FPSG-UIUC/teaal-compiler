import pytest

from teaal.ir.hardware import Hardware
from teaal.ir.metrics import Metrics
from teaal.ir.program import Program
from teaal.parse import *
from teaal.trans.collector import Collector


def build_extensor_yaml():
    with open("tests/integration/extensor.yaml", "r") as f:
        return f.read()


def build_extensor_energy_yaml():
    with open("tests/integration/extensor-energy.yaml", "r") as f:
        return f.read()


def build_gamma_yaml():
    with open("tests/integration/gamma.yaml", "r") as f:
        return f.read()


def build_outerspace_yaml():
    with open("tests/integration/outerspace.yaml", "r") as f:
        return f.read()


def build_sigma_yaml():
    with open("tests/integration/sigma.yaml", "r") as f:
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


def check_hifiber_lines(gen_lines, corr_lines):
    gen_set = set(gen_lines)
    corr_set = set(corr_lines)

    print("In generated")
    for line in gen_lines:
        if line not in corr_set:
            print(line)

    print("In corr")
    for line in corr_lines:
        if line not in gen_set:
            print(line)

    assert gen_set == corr_set


def test_create_component_unknown():
    yaml = """
    einsum:
      declaration:
        Z: []
        A: [K]
        B: [K]
      expressions:
      - Z[] = A[k] * B[k]
    architecture:
      accel:
      - name: level0
        local:
        - name: DRAM
          class: DRAM
    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
    # TODO: Allow the format to be empty
    format:
      Z:
        default:
          rank-order: []
    """
    collector = build_collector(yaml, 0)

    dram = collector.metrics.get_hardware().get_component("DRAM")
    with pytest.raises(ValueError) as excinfo:
        collector.create_component(dram, "K")
    assert str(
        excinfo.value) == "Unable to create consumable metrics component for DRAM of type DRAMComponent"


def test_create_component():
    yaml = """
    einsum:
      declaration:
        Z: []
        A: [I, J, K]
        B: [I, J, K]
      expressions:
      - Z[] = A[i, j, k] * B[i, j, k]
    architecture:
      accel:
      - name: level0
        local:
        - name: LF
          class: Intersector
          attributes:
            type: leader-follower
        - name: SA
          class: Intersector
          attributes:
            type: skip-ahead
        - name: TF
          class: Intersector
          attributes:
            type: two-finger
    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
      - component: LF
        bindings:
        - rank: I
          leader: A
      - component: SA
        bindings:
        - rank: J
      - component: TF
        bindings:
        - rank: K
    # TODO: Allow the format to be empty
    format:
      Z:
        default:
          rank-order: []
    """
    collector = build_collector(yaml, 0)
    get_comp = collector.metrics.get_hardware().get_component

    assert collector.create_component(get_comp("LF"), "I").gen(
        0) == "LF_I = LeaderFollowerIntersector()"
    assert collector.create_component(get_comp("SA"), "J").gen(
        0) == "SA_J = SkipAheadIntersector()"
    assert collector.create_component(get_comp("TF"), "K").gen(
        0) == "TF_K = TwoFingerIntersector()"


def test_consume_traces_unknown():
    yaml = """
    einsum:
      declaration:
        Z: []
        A: [K]
        B: [K]
      expressions:
      - Z[] = A[k] * B[k]
    architecture:
      accel:
      - name: level0
        local:
        - name: DRAM
          class: DRAM
    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
    # TODO: Allow the format to be empty
    format:
      Z:
        default:
          rank-order: []
    """
    collector = build_collector(yaml, 0)

    with pytest.raises(ValueError) as excinfo:
        collector.consume_traces("DRAM", "K")
    assert str(
        excinfo.value) == "Unable to consume traces for component DRAM of type DRAMComponent"


def test_consume_traces():
    yaml = """
    einsum:
      declaration:
        Z: []
        A: [I, J, K]
        B: [I, J, K]
      expressions:
      - Z[] = A[i, j, k] * B[i, j, k]
    architecture:
      accel:
      - name: level0
        local:
        - name: LF
          class: Intersector
          attributes:
            type: leader-follower
        - name: SA
          class: Intersector
          attributes:
            type: skip-ahead
        - name: TF
          class: Intersector
          attributes:
            type: two-finger
    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
      - component: LF
        bindings:
        - rank: I
          leader: A
      - component: SA
        bindings:
        - rank: J
      - component: TF
        bindings:
        - rank: K
    # TODO: Allow the format to be empty
    format:
      Z:
        default:
          rank-order: []
    """
    collector = build_collector(yaml, 0)

    assert collector.consume_traces("LF", "I").gen(
        0) == "LF_I.addTraces(Metrics.consumeTrace(\"I\", \"intersect_0\"))"
    assert collector.consume_traces("SA", "J").gen(
        0) == "SA_J.addTraces(Metrics.consumeTrace(\"J\", \"intersect_0\"), Metrics.consumeTrace(\"J\", \"intersect_1\"))"
    assert collector.consume_traces("TF", "K").gen(
        0) == "TF_K.addTraces(Metrics.consumeTrace(\"K\", \"intersect_0\"), Metrics.consumeTrace(\"K\", \"intersect_1\"))"


def test_dump_gamma_T():
    yaml = build_gamma_yaml()
    collector = build_collector(yaml, 0)

    hifiber = "metrics = {}\n" + \
        "metrics[\"T\"] = {}\n" + \
        "formats = {\"A\": Format(A_MK, {\"rank-order\": [\"M\", \"K\"], \"M\": {\"format\": \"U\", \"pbits\": 32}, \"K\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}}), \"B\": Format(B_KN, {\"rank-order\": [\"K\", \"N\"], \"K\": {\"format\": \"U\", \"pbits\": 32}, \"N\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}})}\n" + \
        "bindings = [{\"tensor\": \"B\", \"rank\": \"K\", \"type\": \"payload\", \"format\": \"default\"}, {\"tensor\": \"B\", \"rank\": \"N\", \"type\": \"coord\", \"format\": \"default\"}, {\"tensor\": \"B\", \"rank\": \"N\", \"type\": \"payload\", \"format\": \"default\"}]\n" + \
        "Traffic.filterTrace(\"tmp/gamma_T-K-intersect_3.csv\", \"tmp/gamma_T-K-iter.csv\", \"tmp/gamma_T-K-intersect_3_payload.csv\")\n" + \
        "Traffic.filterTrace(\"tmp/gamma_T-N-populate_1.csv\", \"tmp/gamma_T-N-iter.csv\", \"tmp/gamma_T-N-populate_1_payload.csv\")\n" + \
        "traces = {(\"B\", \"K\", \"payload\", \"read\"): \"tmp/gamma_T-K-intersect_3_payload.csv\", (\"B\", \"N\", \"coord\", \"read\"): \"tmp/gamma_T-N-populate_1.csv\", (\"B\", \"N\", \"payload\", \"read\"): \"tmp/gamma_T-N-populate_1_payload.csv\"}\n" + \
        "traffic = Traffic.cacheTraffic(bindings, formats, traces, 25165824, 64)\n" + \
        "metrics[\"T\"][\"MainMemory\"] = {}\n" + \
        "metrics[\"T\"][\"MainMemory\"][\"B\"] = {}\n" + \
        "metrics[\"T\"][\"MainMemory\"][\"B\"][\"read\"] = 0\n" + \
        "metrics[\"T\"][\"MainMemory\"][\"B\"][\"read\"] += traffic[0][\"B\"][\"read\"]\n" + \
        "bindings = [{\"tensor\": \"A\", \"rank\": \"M\", \"type\": \"payload\", \"format\": \"default\", \"evict-on\": \"root\", \"style\": \"lazy\"}, {\"tensor\": \"A\", \"rank\": \"K\", \"type\": \"coord\", \"format\": \"default\", \"evict-on\": \"M\", \"style\": \"lazy\"}, {\"tensor\": \"A\", \"rank\": \"K\", \"type\": \"payload\", \"format\": \"default\", \"evict-on\": \"M\", \"style\": \"lazy\"}]\n" + \
        "Traffic.filterTrace(\"tmp/gamma_T-M-populate_1.csv\", \"tmp/gamma_T-M-iter.csv\", \"tmp/gamma_T-M-populate_1_payload.csv\")\n" + \
        "Traffic.filterTrace(\"tmp/gamma_T-K-intersect_2.csv\", \"tmp/gamma_T-K-iter.csv\", \"tmp/gamma_T-K-intersect_2_payload.csv\")\n" + \
        "traces = {(\"A\", \"M\", \"payload\", \"read\"): \"tmp/gamma_T-M-populate_1_payload.csv\", (\"A\", \"K\", \"coord\", \"read\"): \"tmp/gamma_T-K-intersect_2.csv\", (\"A\", \"K\", \"payload\", \"read\"): \"tmp/gamma_T-K-intersect_2_payload.csv\"}\n" + \
        "traffic = Traffic.buffetTraffic(bindings, formats, traces, float(\"inf\"), 64)\n" + \
        "metrics[\"T\"][\"MainMemory\"][\"A\"] = {}\n" + \
        "metrics[\"T\"][\"MainMemory\"][\"A\"][\"read\"] = 0\n" + \
        "metrics[\"T\"][\"MainMemory\"][\"A\"][\"read\"] += traffic[0][\"A\"][\"read\"]\n" + \
        "metrics[\"T\"][\"MainMemory\"][\"time\"] = (metrics[\"T\"][\"MainMemory\"][\"A\"][\"read\"] + metrics[\"T\"][\"MainMemory\"][\"B\"][\"read\"]) / 1099511627776\n" + \
        "metrics[\"T\"][\"Intersect\"] = 0\n" + \
        "metrics[\"T\"][\"Intersect\"] += Intersect_K.getNumIntersects()\n" + \
        "metrics[\"T\"][\"Intersect\"][\"time\"] = metrics[\"T\"][\"Intersect\"] / 32000000000"

    assert collector.dump().gen(0) == hifiber


def test_dump_gamma_Z():
    yaml = build_gamma_yaml()
    collector = build_collector(yaml, 1)

    hifiber = "metrics[\"Z\"] = {}\n" + \
        "formats = {\"Z\": Format(Z_MN, {\"rank-order\": [\"M\", \"N\"], \"M\": {\"format\": \"U\", \"pbits\": 32}, \"N\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}}), \"A\": Format(A_MK, {\"rank-order\": [\"M\", \"K\"], \"M\": {\"format\": \"U\", \"pbits\": 32}, \"K\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}})}\n" + \
        "bindings = [{\"tensor\": \"A\", \"rank\": \"M\", \"type\": \"payload\", \"format\": \"default\", \"evict-on\": \"root\", \"style\": \"lazy\"}, {\"tensor\": \"A\", \"rank\": \"K\", \"type\": \"coord\", \"format\": \"default\", \"evict-on\": \"M\", \"style\": \"lazy\"}, {\"tensor\": \"A\", \"rank\": \"K\", \"type\": \"payload\", \"format\": \"default\", \"evict-on\": \"M\", \"style\": \"lazy\"}]\n" + \
        "Traffic.filterTrace(\"tmp/gamma_Z-M-intersect_3.csv\", \"tmp/gamma_Z-M-iter.csv\", \"tmp/gamma_Z-M-intersect_3_payload.csv\")\n" + \
        "Traffic.filterTrace(\"tmp/gamma_Z-K-intersect_1.csv\", \"tmp/gamma_Z-K-iter.csv\", \"tmp/gamma_Z-K-intersect_1_payload.csv\")\n" + \
        "traces = {(\"A\", \"M\", \"payload\", \"read\"): \"tmp/gamma_Z-M-intersect_3_payload.csv\", (\"A\", \"K\", \"coord\", \"read\"): \"tmp/gamma_Z-K-intersect_1.csv\", (\"A\", \"K\", \"payload\", \"read\"): \"tmp/gamma_Z-K-intersect_1_payload.csv\"}\n" + \
        "traffic = Traffic.buffetTraffic(bindings, formats, traces, float(\"inf\"), 64)\n" + \
        "bindings = [{\"tensor\": \"Z\", \"rank\": \"M\", \"type\": \"payload\", \"format\": \"default\", \"evict-on\": \"root\", \"style\": \"lazy\"}, {\"tensor\": \"Z\", \"rank\": \"N\", \"type\": \"coord\", \"format\": \"default\", \"evict-on\": \"M\", \"style\": \"lazy\"}, {\"tensor\": \"Z\", \"rank\": \"N\", \"type\": \"payload\", \"format\": \"default\", \"evict-on\": \"M\", \"style\": \"lazy\"}]\n" + \
        "Traffic.filterTrace(\"tmp/gamma_Z-M-populate_read_0.csv\", \"tmp/gamma_Z-M-iter.csv\", \"tmp/gamma_Z-M-populate_read_0_payload.csv\")\n" + \
        "Traffic.filterTrace(\"tmp/gamma_Z-M-populate_write_0.csv\", \"tmp/gamma_Z-M-iter.csv\", \"tmp/gamma_Z-M-populate_write_0_payload.csv\")\n" + \
        "Traffic.filterTrace(\"tmp/gamma_Z-N-populate_read_0.csv\", \"tmp/gamma_Z-N-iter.csv\", \"tmp/gamma_Z-N-populate_read_0_payload.csv\")\n" + \
        "Traffic.filterTrace(\"tmp/gamma_Z-N-populate_write_0.csv\", \"tmp/gamma_Z-N-iter.csv\", \"tmp/gamma_Z-N-populate_write_0_payload.csv\")\n" + \
        "traces = {(\"Z\", \"M\", \"payload\", \"read\"): \"tmp/gamma_Z-M-populate_read_0_payload.csv\", (\"Z\", \"M\", \"payload\", \"write\"): \"tmp/gamma_Z-M-populate_write_0_payload.csv\", (\"Z\", \"N\", \"coord\", \"read\"): \"tmp/gamma_Z-N-populate_read_0.csv\", (\"Z\", \"N\", \"coord\", \"write\"): \"tmp/gamma_Z-N-populate_write_0.csv\", (\"Z\", \"N\", \"payload\", \"read\"): \"tmp/gamma_Z-N-populate_read_0_payload.csv\", (\"Z\", \"N\", \"payload\", \"write\"): \"tmp/gamma_Z-N-populate_write_0_payload.csv\"}\n" + \
        "traffic = Traffic.buffetTraffic(bindings, formats, traces, float(\"inf\"), 64)\n" + \
        "metrics[\"Z\"][\"MainMemory\"] = {}\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"] = {}\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"read\"] = 0\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"write\"] = 0\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"read\"] += traffic[0][\"Z\"][\"read\"]\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"write\"] += traffic[0][\"Z\"][\"write\"]\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"time\"] = (metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"read\"] + metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"write\"]) / 1099511627776\n" + \
        "metrics[\"Z\"][\"HighRadixMerger\"] = {}\n" + \
        "metrics[\"Z\"][\"HighRadixMerger\"][\"T_MKN\"] = Compute.numSwaps(T_MKN, 1, 64, 1)\n" + \
        "metrics[\"Z\"][\"HighRadixMerger\"][\"time\"] = metrics[\"Z\"][\"HighRadixMerger\"][T_MKN] / 32000000000\n" + \
        "metrics[\"Z\"][\"FPMul\"] = {}\n" + \
        "metrics[\"Z\"][\"FPMul\"][\"mul\"] = Metrics.dump()[\"Compute\"][\"payload_mul\"]\n" + \
        "metrics[\"Z\"][\"FPMul\"][\"time\"] = metrics[\"Z\"][\"FPMul\"][\"mul\"] / 32000000000\n" + \
        "metrics[\"Z\"][\"FPAdd\"] = {}\n" + \
        "metrics[\"Z\"][\"FPAdd\"][\"add\"] = Metrics.dump()[\"Compute\"][\"payload_add\"]\n" + \
        "metrics[\"Z\"][\"FPAdd\"][\"time\"] = metrics[\"Z\"][\"FPAdd\"][\"add\"] / 32000000000"

    # print(collector.dump().gen(0))
    # assert False

    assert collector.dump().gen(0) == hifiber


def test_dump_outerspace_Z():
    yaml = build_outerspace_yaml()
    collector = build_collector(yaml, 2)

    hifiber = "metrics[\"Z\"] = {}\n" + \
        "formats = {\"Z\": Format(Z_MN, {\"rank-order\": [\"M\", \"N\"], \"M\": {\"format\": \"U\", \"pbits\": 32}, \"N\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}})}\n" + \
        "bindings = [{\"tensor\": \"Z\", \"rank\": \"M\", \"type\": \"payload\", \"format\": \"default\", \"evict-on\": \"root\", \"style\": \"lazy\"}, {\"tensor\": \"Z\", \"rank\": \"N\", \"type\": \"coord\", \"format\": \"default\", \"evict-on\": \"M\", \"style\": \"lazy\"}, {\"tensor\": \"Z\", \"rank\": \"N\", \"type\": \"payload\", \"format\": \"default\", \"evict-on\": \"M\", \"style\": \"lazy\"}]\n" + \
        "Traffic.filterTrace(\"tmp/outerspace_Z-M-populate_read_0.csv\", \"tmp/outerspace_Z-M-iter.csv\", \"tmp/outerspace_Z-M-populate_read_0_payload.csv\")\n" + \
        "Traffic.filterTrace(\"tmp/outerspace_Z-M-populate_write_0.csv\", \"tmp/outerspace_Z-M-iter.csv\", \"tmp/outerspace_Z-M-populate_write_0_payload.csv\")\n" + \
        "Traffic.filterTrace(\"tmp/outerspace_Z-N-populate_read_0.csv\", \"tmp/outerspace_Z-N-iter.csv\", \"tmp/outerspace_Z-N-populate_read_0_payload.csv\")\n" + \
        "Traffic.filterTrace(\"tmp/outerspace_Z-N-populate_write_0.csv\", \"tmp/outerspace_Z-N-iter.csv\", \"tmp/outerspace_Z-N-populate_write_0_payload.csv\")\n" + \
        "traces = {(\"Z\", \"M\", \"payload\", \"read\"): \"tmp/outerspace_Z-M-populate_read_0_payload.csv\", (\"Z\", \"M\", \"payload\", \"write\"): \"tmp/outerspace_Z-M-populate_write_0_payload.csv\", (\"Z\", \"N\", \"coord\", \"read\"): \"tmp/outerspace_Z-N-populate_read_0.csv\", (\"Z\", \"N\", \"coord\", \"write\"): \"tmp/outerspace_Z-N-populate_write_0.csv\", (\"Z\", \"N\", \"payload\", \"read\"): \"tmp/outerspace_Z-N-populate_read_0_payload.csv\", (\"Z\", \"N\", \"payload\", \"write\"): \"tmp/outerspace_Z-N-populate_write_0_payload.csv\"}\n" + \
        "traffic = Traffic.buffetTraffic(bindings, formats, traces, 8192, 64)\n" + \
        "metrics[\"Z\"][\"MainMemory\"] = {}\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"] = {}\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"read\"] = 0\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"write\"] = 0\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"read\"] += traffic[0][\"Z\"][\"read\"]\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"write\"] += traffic[0][\"Z\"][\"write\"]\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"time\"] = (metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"read\"] + metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"write\"]) / 1099511627776\n" + \
        "metrics[\"Z\"][\"SortHW\"] = {}\n" + \
        "metrics[\"Z\"][\"SortHW\"][\"T1_MKN\"] = Compute.numSwaps(T1_MKN, 1, float(\"inf\"), \"N\")\n" + \
        "metrics[\"Z\"][\"SortHW\"][\"time\"] = metrics[\"Z\"][\"SortHW\"][T1_MKN] / 193500000000\n" + \
        "metrics[\"Z\"][\"FPAdd\"] = {}\n" + \
        "metrics[\"Z\"][\"FPAdd\"][\"add\"] = Metrics.dump()[\"Compute\"][\"payload_add\"]\n" + \
        "metrics[\"Z\"][\"FPAdd\"][\"time\"] = metrics[\"Z\"][\"FPAdd\"][\"add\"] / 193500000000"

    assert collector.dump().gen(0) == hifiber


def test_dump_extensor():
    yaml = build_extensor_yaml()
    collector = build_collector(yaml, 0)

    hifiber = "metrics = {}\n" + \
        "metrics[\"Z\"] = {}\n" + \
        "formats = {\"Z\": Format(Z_N2M2M1N1M0N0, {\"rank-order\": [\"N2\", \"M2\", \"M1\", \"N1\", \"M0\", \"N0\"], \"N2\": {\"format\": \"U\"}, \"M2\": {\"format\": \"U\"}, \"M1\": {\"format\": \"U\"}, \"N1\": {\"format\": \"U\"}, \"M0\": {\"format\": \"U\"}, \"N0\": {\"format\": \"C\", \"cbits\": 64, \"pbits\": 64}}), \"A\": Format(A_K2M2M1K1M0K0, {\"rank-order\": [\"K2\", \"M2\", \"M1\", \"K1\", \"M0\", \"K0\"], \"K2\": {\"format\": \"C\"}, \"M2\": {\"format\": \"C\"}, \"M1\": {\"format\": \"C\"}, \"K1\": {\"format\": \"C\", \"cbits\": 64}, \"M0\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 32}, \"K0\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}}), \"B\": Format(B_N2K2N1K1N0K0, {\"rank-order\": [\"N2\", \"K2\", \"N1\", \"K1\", \"N0\", \"K0\"], \"N2\": {\"format\": \"C\"}, \"K2\": {\"format\": \"C\"}, \"N1\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 32}, \"K1\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 32}, \"N0\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 32}, \"K0\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}})}\n" + \
        "bindings = [{\"tensor\": \"A\", \"rank\": \"K1\", \"type\": \"coord\", \"evict-on\": \"M2\", \"format\": \"default\", \"style\": \"lazy\"}, {\"tensor\": \"A\", \"rank\": \"M0\", \"type\": \"coord\", \"evict-on\": \"M2\", \"format\": \"default\", \"style\": \"eager\", \"root\": \"M0\"}, {\"tensor\": \"B\", \"rank\": \"N1\", \"type\": \"coord\", \"evict-on\": \"K2\", \"format\": \"default\", \"style\": \"lazy\"}, {\"tensor\": \"B\", \"rank\": \"N1\", \"type\": \"payload\", \"evict-on\": \"K2\", \"format\": \"default\", \"style\": \"lazy\"}, {\"tensor\": \"B\", \"rank\": \"K1\", \"type\": \"coord\", \"evict-on\": \"K2\", \"format\": \"default\", \"style\": \"lazy\"}, {\"tensor\": \"B\", \"rank\": \"K1\", \"type\": \"payload\", \"evict-on\": \"K2\", \"format\": \"default\", \"style\": \"lazy\"}, {\"tensor\": \"B\", \"rank\": \"N0\", \"type\": \"coord\", \"evict-on\": \"K2\", \"format\": \"default\", \"style\": \"eager\", \"root\": \"N0\"}, {\"tensor\": \"Z\", \"evict-on\": \"M2\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"M0\", \"rank\": \"N0\", \"type\": \"coord\"}, {\"tensor\": \"Z\", \"evict-on\": \"M2\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"M0\", \"rank\": \"N0\", \"type\": \"payload\"}, {\"tensor\": \"A\", \"evict-on\": \"M2\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"M0\", \"rank\": \"M0\", \"type\": \"payload\"}, {\"tensor\": \"A\", \"evict-on\": \"M2\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"M0\", \"rank\": \"K0\", \"type\": \"coord\"}, {\"tensor\": \"A\", \"evict-on\": \"M2\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"M0\", \"rank\": \"K0\", \"type\": \"payload\"}, {\"tensor\": \"B\", \"evict-on\": \"K2\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"N0\", \"rank\": \"N0\", \"type\": \"payload\"}, {\"tensor\": \"B\", \"evict-on\": \"K2\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"N0\", \"rank\": \"K0\", \"type\": \"coord\"}, {\"tensor\": \"B\", \"evict-on\": \"K2\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"N0\", \"rank\": \"K0\", \"type\": \"payload\"}]\n" + \
        "Traffic.filterTrace(\"tmp/extensor-N1-populate_1.csv\", \"tmp/extensor-N1-iter.csv\", \"tmp/extensor-N1-populate_1_payload.csv\")\n" + \
        "Traffic.filterTrace(\"tmp/extensor-K1-intersect_1.csv\", \"tmp/extensor-K1-iter.csv\", \"tmp/extensor-K1-intersect_1_payload.csv\")\n" + \
        "traces = {(\"A\", \"K1\", \"coord\", \"read\"): \"tmp/extensor-K1-intersect_0.csv\", (\"A\", \"M0\", \"coord\", \"read\"): \"tmp/extensor-M0-eager_a_m0_read.csv\", (\"B\", \"N1\", \"coord\", \"read\"): \"tmp/extensor-N1-populate_1.csv\", (\"B\", \"N1\", \"payload\", \"read\"): \"tmp/extensor-N1-populate_1_payload.csv\", (\"B\", \"K1\", \"coord\", \"read\"): \"tmp/extensor-K1-intersect_1.csv\", (\"B\", \"K1\", \"payload\", \"read\"): \"tmp/extensor-K1-intersect_1_payload.csv\", (\"B\", \"N0\", \"coord\", \"read\"): \"tmp/extensor-N0-eager_b_n0_read.csv\", (\"Z\", \"N0\", \"coord\", \"read\"): \"tmp/extensor-N0-eager_z_m0_read.csv\", (\"Z\", \"N0\", \"coord\", \"write\"): \"tmp/extensor-N0-eager_z_m0_write.csv\", (\"Z\", \"N0\", \"payload\", \"read\"): \"tmp/extensor-N0-eager_z_m0_read.csv\", (\"Z\", \"N0\", \"payload\", \"write\"): \"tmp/extensor-N0-eager_z_m0_write.csv\", (\"A\", \"M0\", \"payload\", \"read\"): \"tmp/extensor-M0-eager_a_m0_read.csv\", (\"A\", \"K0\", \"coord\", \"read\"): \"tmp/extensor-K0-eager_a_m0_read.csv\", (\"A\", \"K0\", \"payload\", \"read\"): \"tmp/extensor-K0-eager_a_m0_read.csv\", (\"B\", \"N0\", \"payload\", \"read\"): \"tmp/extensor-N0-eager_b_n0_read.csv\", (\"B\", \"K0\", \"coord\", \"read\"): \"tmp/extensor-K0-eager_b_n0_read.csv\", (\"B\", \"K0\", \"payload\", \"read\"): \"tmp/extensor-K0-eager_b_n0_read.csv\"}\n" + \
        "traffic = Traffic.buffetTraffic(bindings, formats, traces, 251658240, 64)\n" + \
        "metrics[\"Z\"][\"MainMemory\"] = {}\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"A\"] = {}\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"A\"][\"read\"] = 0\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"A\"][\"read\"] += traffic[0][\"A\"][\"read\"]\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"B\"] = {}\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"B\"][\"read\"] = 0\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"B\"][\"read\"] += traffic[0][\"B\"][\"read\"]\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"] = {}\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"read\"] = 0\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"write\"] = 0\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"read\"] += traffic[0][\"Z\"][\"read\"]\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"write\"] += traffic[0][\"Z\"][\"write\"]\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"time\"] = (metrics[\"Z\"][\"MainMemory\"][\"A\"][\"read\"] + metrics[\"Z\"][\"MainMemory\"][\"B\"][\"read\"] + metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"read\"] + metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"write\"]) / 586314575512\n" + \
        "metrics[\"Z\"][\"FPMul\"] = {}\n" + \
        "metrics[\"Z\"][\"FPMul\"][\"mul\"] = Metrics.dump()[\"Compute\"][\"payload_mul\"]\n" + \
        "metrics[\"Z\"][\"FPMul\"][\"time\"] = metrics[\"Z\"][\"FPMul\"][\"mul\"] / 128000000000\n" + \
        "metrics[\"Z\"][\"FPAdd\"] = {}\n" + \
        "metrics[\"Z\"][\"FPAdd\"][\"add\"] = Metrics.dump()[\"Compute\"][\"payload_add\"]\n" + \
        "metrics[\"Z\"][\"FPAdd\"][\"time\"] = metrics[\"Z\"][\"FPAdd\"][\"add\"] / 128000000000\n" + \
        "metrics[\"Z\"][\"K2Intersect\"] = 0\n" + \
        "metrics[\"Z\"][\"K2Intersect\"] += K2Intersect_K2.getNumIntersects()\n" + \
        "metrics[\"Z\"][\"K2Intersect\"][\"time\"] = metrics[\"Z\"][\"K2Intersect\"] / 1000000000\n" + \
        "metrics[\"Z\"][\"K1Intersect\"] = 0\n" + \
        "metrics[\"Z\"][\"K1Intersect\"] += K1Intersect_K1.getNumIntersects()\n" + \
        "metrics[\"Z\"][\"K1Intersect\"][\"time\"] = metrics[\"Z\"][\"K1Intersect\"] / 1000000000\n" + \
        "metrics[\"Z\"][\"K0Intersection\"] = 0\n" + \
        "metrics[\"Z\"][\"K0Intersection\"] += K0Intersection_K0.getNumIntersects()\n" + \
        "metrics[\"Z\"][\"K0Intersection\"][\"time\"] = metrics[\"Z\"][\"K0Intersection\"] / 128000000000"


    assert collector.dump().gen(0) == hifiber


def test_dump_extensor_energy():
    yaml = build_extensor_energy_yaml()
    collector = build_collector(yaml, 0)

    hifiber = "metrics = {}\n" + \
        "metrics[\"Z\"] = {}\n" + \
        "formats = {\"Z\": Format(Z_N2M2M1N1M0N0, {\"rank-order\": [\"N2\", \"M2\", \"M1\", \"N1\", \"M0\", \"N0\"], \"N2\": {\"format\": \"U\"}, \"M2\": {\"format\": \"U\"}, \"M1\": {\"format\": \"U\"}, \"N1\": {\"format\": \"U\"}, \"M0\": {\"format\": \"U\"}, \"N0\": {\"format\": \"C\", \"cbits\": 64, \"pbits\": 64}}), \"A\": Format(A_K2M2M1K1M0K0, {\"rank-order\": [\"K2\", \"M2\", \"M1\", \"K1\", \"M0\", \"K0\"], \"K2\": {\"format\": \"C\"}, \"M2\": {\"format\": \"C\"}, \"M1\": {\"format\": \"C\"}, \"K1\": {\"format\": \"C\", \"cbits\": 64}, \"M0\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 32}, \"K0\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}}), \"B\": Format(B_N2K2N1K1N0K0, {\"rank-order\": [\"N2\", \"K2\", \"N1\", \"K1\", \"N0\", \"K0\"], \"N2\": {\"format\": \"C\"}, \"K2\": {\"format\": \"C\"}, \"N1\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 32}, \"K1\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 32}, \"N0\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 32}, \"K0\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}})}\n" + \
        "bindings = [{\"tensor\": \"A\", \"rank\": \"K1\", \"type\": \"coord\", \"evict-on\": \"M2\", \"format\": \"default\", \"style\": \"lazy\"}, {\"tensor\": \"A\", \"rank\": \"M0\", \"type\": \"coord\", \"evict-on\": \"M2\", \"format\": \"default\", \"style\": \"eager\", \"root\": \"M0\"}, {\"tensor\": \"B\", \"rank\": \"N1\", \"type\": \"coord\", \"evict-on\": \"K2\", \"format\": \"default\", \"style\": \"lazy\"}, {\"tensor\": \"B\", \"rank\": \"N1\", \"type\": \"payload\", \"evict-on\": \"K2\", \"format\": \"default\", \"style\": \"lazy\"}, {\"tensor\": \"B\", \"rank\": \"K1\", \"type\": \"coord\", \"evict-on\": \"K2\", \"format\": \"default\", \"style\": \"lazy\"}, {\"tensor\": \"B\", \"rank\": \"K1\", \"type\": \"payload\", \"evict-on\": \"K2\", \"format\": \"default\", \"style\": \"lazy\"}, {\"tensor\": \"B\", \"rank\": \"N0\", \"type\": \"coord\", \"evict-on\": \"K2\", \"format\": \"default\", \"style\": \"eager\", \"root\": \"N0\"}, {\"tensor\": \"Z\", \"evict-on\": \"M2\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"M0\", \"rank\": \"N0\", \"type\": \"coord\"}, {\"tensor\": \"Z\", \"evict-on\": \"M2\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"M0\", \"rank\": \"N0\", \"type\": \"payload\"}, {\"tensor\": \"A\", \"evict-on\": \"M2\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"M0\", \"rank\": \"M0\", \"type\": \"payload\"}, {\"tensor\": \"A\", \"evict-on\": \"M2\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"M0\", \"rank\": \"K0\", \"type\": \"coord\"}, {\"tensor\": \"A\", \"evict-on\": \"M2\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"M0\", \"rank\": \"K0\", \"type\": \"payload\"}, {\"tensor\": \"B\", \"evict-on\": \"K2\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"N0\", \"rank\": \"N0\", \"type\": \"payload\"}, {\"tensor\": \"B\", \"evict-on\": \"K2\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"N0\", \"rank\": \"K0\", \"type\": \"coord\"}, {\"tensor\": \"B\", \"evict-on\": \"K2\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"N0\", \"rank\": \"K0\", \"type\": \"payload\"}]\n" + \
        "Traffic.filterTrace(\"tmp/extensor_energy-N1-populate_1.csv\", \"tmp/extensor_energy-N1-iter.csv\", \"tmp/extensor_energy-N1-populate_1_payload.csv\")\n" + \
        "Traffic.filterTrace(\"tmp/extensor_energy-K1-intersect_1.csv\", \"tmp/extensor_energy-K1-iter.csv\", \"tmp/extensor_energy-K1-intersect_1_payload.csv\")\n" + \
        "traces = {(\"A\", \"K1\", \"coord\", \"read\"): \"tmp/extensor_energy-K1-intersect_0.csv\", (\"A\", \"M0\", \"coord\", \"read\"): \"tmp/extensor_energy-M0-eager_a_m0_read.csv\", (\"B\", \"N1\", \"coord\", \"read\"): \"tmp/extensor_energy-N1-populate_1.csv\", (\"B\", \"N1\", \"payload\", \"read\"): \"tmp/extensor_energy-N1-populate_1_payload.csv\", (\"B\", \"K1\", \"coord\", \"read\"): \"tmp/extensor_energy-K1-intersect_1.csv\", (\"B\", \"K1\", \"payload\", \"read\"): \"tmp/extensor_energy-K1-intersect_1_payload.csv\", (\"B\", \"N0\", \"coord\", \"read\"): \"tmp/extensor_energy-N0-eager_b_n0_read.csv\", (\"Z\", \"N0\", \"coord\", \"read\"): \"tmp/extensor_energy-N0-eager_z_m0_read.csv\", (\"Z\", \"N0\", \"coord\", \"write\"): \"tmp/extensor_energy-N0-eager_z_m0_write.csv\", (\"Z\", \"N0\", \"payload\", \"read\"): \"tmp/extensor_energy-N0-eager_z_m0_read.csv\", (\"Z\", \"N0\", \"payload\", \"write\"): \"tmp/extensor_energy-N0-eager_z_m0_write.csv\", (\"A\", \"M0\", \"payload\", \"read\"): \"tmp/extensor_energy-M0-eager_a_m0_read.csv\", (\"A\", \"K0\", \"coord\", \"read\"): \"tmp/extensor_energy-K0-eager_a_m0_read.csv\", (\"A\", \"K0\", \"payload\", \"read\"): \"tmp/extensor_energy-K0-eager_a_m0_read.csv\", (\"B\", \"N0\", \"payload\", \"read\"): \"tmp/extensor_energy-N0-eager_b_n0_read.csv\", (\"B\", \"K0\", \"coord\", \"read\"): \"tmp/extensor_energy-K0-eager_b_n0_read.csv\", (\"B\", \"K0\", \"payload\", \"read\"): \"tmp/extensor_energy-K0-eager_b_n0_read.csv\"}\n" + \
        "traffic = Traffic.buffetTraffic(bindings, formats, traces, 251658240, 64)\n" + \
        "metrics[\"Z\"][\"MainMemory\"] = {}\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"A\"] = {}\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"A\"][\"read\"] = 0\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"A\"][\"read\"] += traffic[0][\"A\"][\"read\"]\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"B\"] = {}\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"B\"][\"read\"] = 0\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"B\"][\"read\"] += traffic[0][\"B\"][\"read\"]\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"] = {}\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"read\"] = 0\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"write\"] = 0\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"read\"] += traffic[0][\"Z\"][\"read\"]\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"write\"] += traffic[0][\"Z\"][\"write\"]\n" + \
        "bindings = [{\"tensor\": \"A\", \"rank\": \"M0\", \"type\": \"coord\", \"evict-on\": \"K1\", \"format\": \"default\", \"style\": \"eager\", \"root\": \"M0\"}, {\"tensor\": \"B\", \"rank\": \"N0\", \"type\": \"coord\", \"evict-on\": \"K1\", \"format\": \"default\", \"style\": \"eager\", \"root\": \"N0\"}, {\"tensor\": \"Z\", \"evict-on\": \"N1\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"M0\", \"rank\": \"N0\", \"type\": \"coord\"}, {\"tensor\": \"Z\", \"evict-on\": \"N1\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"M0\", \"rank\": \"N0\", \"type\": \"payload\"}, {\"tensor\": \"A\", \"evict-on\": \"K1\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"M0\", \"rank\": \"M0\", \"type\": \"payload\"}, {\"tensor\": \"A\", \"evict-on\": \"K1\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"M0\", \"rank\": \"K0\", \"type\": \"coord\"}, {\"tensor\": \"A\", \"evict-on\": \"K1\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"M0\", \"rank\": \"K0\", \"type\": \"payload\"}, {\"tensor\": \"B\", \"evict-on\": \"K1\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"N0\", \"rank\": \"N0\", \"type\": \"payload\"}, {\"tensor\": \"B\", \"evict-on\": \"K1\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"N0\", \"rank\": \"K0\", \"type\": \"coord\"}, {\"tensor\": \"B\", \"evict-on\": \"K1\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"N0\", \"rank\": \"K0\", \"type\": \"payload\"}]\n" + \
        "traces = {(\"A\", \"M0\", \"coord\", \"read\"): \"tmp/extensor_energy-M0-eager_a_m0_read.csv\", (\"B\", \"N0\", \"coord\", \"read\"): \"tmp/extensor_energy-N0-eager_b_n0_read.csv\", (\"Z\", \"N0\", \"coord\", \"read\"): \"tmp/extensor_energy-N0-eager_z_m0_read.csv\", (\"Z\", \"N0\", \"coord\", \"write\"): \"tmp/extensor_energy-N0-eager_z_m0_write.csv\", (\"Z\", \"N0\", \"payload\", \"read\"): \"tmp/extensor_energy-N0-eager_z_m0_read.csv\", (\"Z\", \"N0\", \"payload\", \"write\"): \"tmp/extensor_energy-N0-eager_z_m0_write.csv\", (\"A\", \"M0\", \"payload\", \"read\"): \"tmp/extensor_energy-M0-eager_a_m0_read.csv\", (\"A\", \"K0\", \"coord\", \"read\"): \"tmp/extensor_energy-K0-eager_a_m0_read.csv\", (\"A\", \"K0\", \"payload\", \"read\"): \"tmp/extensor_energy-K0-eager_a_m0_read.csv\", (\"B\", \"N0\", \"payload\", \"read\"): \"tmp/extensor_energy-N0-eager_b_n0_read.csv\", (\"B\", \"K0\", \"coord\", \"read\"): \"tmp/extensor_energy-K0-eager_b_n0_read.csv\", (\"B\", \"K0\", \"payload\", \"read\"): \"tmp/extensor_energy-K0-eager_b_n0_read.csv\"}\n" + \
        "traffic = Traffic.buffetTraffic(bindings, formats, traces, 524288, 64)\n" + \
        "metrics[\"Z\"][\"LLB\"] = {}\n" + \
        "metrics[\"Z\"][\"LLB\"][\"A\"] = {}\n" + \
        "metrics[\"Z\"][\"LLB\"][\"A\"][\"read\"] = 0\n" + \
        "metrics[\"Z\"][\"LLB\"][\"A\"][\"read\"] += traffic[0][\"A\"][\"read\"]\n" + \
        "metrics[\"Z\"][\"LLB\"][\"B\"] = {}\n" + \
        "metrics[\"Z\"][\"LLB\"][\"B\"][\"read\"] = 0\n" + \
        "metrics[\"Z\"][\"LLB\"][\"B\"][\"read\"] += traffic[0][\"B\"][\"read\"]\n" + \
        "metrics[\"Z\"][\"LLB\"][\"Z\"] = {}\n" + \
        "metrics[\"Z\"][\"LLB\"][\"Z\"][\"read\"] = 0\n" + \
        "metrics[\"Z\"][\"LLB\"][\"Z\"][\"write\"] = 0\n" + \
        "metrics[\"Z\"][\"LLB\"][\"Z\"][\"read\"] += traffic[0][\"Z\"][\"read\"]\n" + \
        "metrics[\"Z\"][\"LLB\"][\"Z\"][\"write\"] += traffic[0][\"Z\"][\"write\"]\n" + \
        "metrics[\"Z\"][\"MainMemory\"][\"time\"] = (metrics[\"Z\"][\"MainMemory\"][\"A\"][\"read\"] + metrics[\"Z\"][\"MainMemory\"][\"B\"][\"read\"] + metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"read\"] + metrics[\"Z\"][\"MainMemory\"][\"Z\"][\"write\"]) / 586314575512\n" + \
        "metrics[\"Z\"][\"LLB\"][\"time\"] = (metrics[\"Z\"][\"LLB\"][\"A\"][\"read\"] + metrics[\"Z\"][\"LLB\"][\"B\"][\"read\"] + metrics[\"Z\"][\"LLB\"][\"Z\"][\"read\"] + metrics[\"Z\"][\"LLB\"][\"Z\"][\"write\"]) / 9223372036854775807\n" + \
        "metrics[\"Z\"][\"FPMul\"] = {}\n" + \
        "metrics[\"Z\"][\"FPMul\"][\"mul\"] = Metrics.dump()[\"Compute\"][\"payload_mul\"]\n" + \
        "metrics[\"Z\"][\"FPMul\"][\"time\"] = metrics[\"Z\"][\"FPMul\"][\"mul\"] / 128000000000\n" + \
        "metrics[\"Z\"][\"FPAdd\"] = {}\n" + \
        "metrics[\"Z\"][\"FPAdd\"][\"add\"] = Metrics.dump()[\"Compute\"][\"payload_add\"]\n" + \
        "metrics[\"Z\"][\"FPAdd\"][\"time\"] = metrics[\"Z\"][\"FPAdd\"][\"add\"] / 128000000000\n" + \
        "metrics[\"Z\"][\"K2Intersect\"] = 0\n" + \
        "metrics[\"Z\"][\"K2Intersect\"] += K2Intersect_K2.getNumIntersects()\n" + \
        "metrics[\"Z\"][\"K2Intersect\"][\"time\"] = metrics[\"Z\"][\"K2Intersect\"] / 1000000000\n" + \
        "metrics[\"Z\"][\"K1Intersect\"] = 0\n" + \
        "metrics[\"Z\"][\"K1Intersect\"] += K1Intersect_K1.getNumIntersects()\n" + \
        "metrics[\"Z\"][\"K1Intersect\"][\"time\"] = metrics[\"Z\"][\"K1Intersect\"] / 1000000000\n" + \
        "metrics[\"Z\"][\"K0Intersection\"] = 0\n" + \
        "metrics[\"Z\"][\"K0Intersection\"] += K0Intersection_K0.getNumIntersects()\n" + \
        "metrics[\"Z\"][\"K0Intersection\"][\"time\"] = metrics[\"Z\"][\"K0Intersection\"] / 128000000000\n" + \
        "metrics[\"Z\"][\"TopSequencer\"] = {}\n" + \
        "metrics[\"Z\"][\"TopSequencer\"][\"N2\"] = Compute.numIters(\"tmp/extensor_energy-N2-iter.csv\")\n" + \
        "metrics[\"Z\"][\"TopSequencer\"][\"K2\"] = Compute.numIters(\"tmp/extensor_energy-K2-iter.csv\")\n" + \
        "metrics[\"Z\"][\"TopSequencer\"][\"M2\"] = Compute.numIters(\"tmp/extensor_energy-M2-iter.csv\")\n" + \
        "metrics[\"Z\"][\"TopSequencer\"][\"time\"] = (metrics[\"Z\"][\"TopSequencer\"][\"N2\"] + metrics[\"Z\"][\"TopSequencer\"][\"K2\"] + metrics[\"Z\"][\"TopSequencer\"][\"M2\"]) / 1000000000\n" + \
        "metrics[\"Z\"][\"MiddleSequencer\"] = {}\n" + \
        "metrics[\"Z\"][\"MiddleSequencer\"][\"M1\"] = Compute.numIters(\"tmp/extensor_energy-M1-iter.csv\")\n" + \
        "metrics[\"Z\"][\"MiddleSequencer\"][\"N1\"] = Compute.numIters(\"tmp/extensor_energy-N1-iter.csv\")\n" + \
        "metrics[\"Z\"][\"MiddleSequencer\"][\"K1\"] = Compute.numIters(\"tmp/extensor_energy-K1-iter.csv\")\n" + \
        "metrics[\"Z\"][\"MiddleSequencer\"][\"time\"] = (metrics[\"Z\"][\"MiddleSequencer\"][\"M1\"] + metrics[\"Z\"][\"MiddleSequencer\"][\"N1\"] + metrics[\"Z\"][\"MiddleSequencer\"][\"K1\"]) / 1000000000\n" + \
        "metrics[\"Z\"][\"BottomSequencer\"] = {}\n" + \
        "metrics[\"Z\"][\"BottomSequencer\"][\"M0\"] = Compute.numIters(\"tmp/extensor_energy-M0-iter.csv\")\n" + \
        "metrics[\"Z\"][\"BottomSequencer\"][\"N0\"] = Compute.numIters(\"tmp/extensor_energy-N0-iter.csv\")\n" + \
        "metrics[\"Z\"][\"BottomSequencer\"][\"K0\"] = Compute.numIters(\"tmp/extensor_energy-K0-iter.csv\")\n" + \
        "metrics[\"Z\"][\"BottomSequencer\"][\"time\"] = (metrics[\"Z\"][\"BottomSequencer\"][\"M0\"] + metrics[\"Z\"][\"BottomSequencer\"][\"N0\"] + metrics[\"Z\"][\"BottomSequencer\"][\"K0\"]) / 128000000000"


    assert collector.dump().gen(0) == hifiber


def test_dump_sigma():
    yaml = build_sigma_yaml()
    collector = build_collector(yaml, 0)

    hifiber = "metrics = {}\n" + \
        "metrics[\"Z\"] = {}\n" + \
        "formats = {\"A\": Format(Tensor(rank_ids=[\"K1\", \"MK01\", \"MK00\"], shape=[K, M * K, M * K]), {\"rank-order\": [\"K1\", \"MK01\", \"MK00\"], \"K1\": {\"format\": \"U\"}, \"MK01\": {\"format\": \"U\"}, \"MK00\": {\"format\": \"C\", \"pbits\": 32}}), \"B\": Format(B_K1NK0, {\"rank-order\": [\"K1\", \"N\", \"K0\"], \"K1\": {\"format\": \"U\"}, \"N\": {\"format\": \"U\"}, \"K0\": {\"format\": \"U\", \"pbits\": 32}})}\n" + \
        "bindings = [{\"tensor\": \"A\", \"rank\": \"MK00\", \"type\": \"payload\", \"evict-on\": \"root\", \"format\": \"flattened\", \"style\": \"eager\", \"root\": \"MK00\"}, {\"tensor\": \"B\", \"rank\": \"K0\", \"type\": \"payload\", \"evict-on\": \"root\", \"format\": \"partitioned\", \"style\": \"eager\", \"root\": \"K0\"}]\n" + \
        "traces = {(\"A\", \"MK00\", \"payload\", \"read\"): \"tmp/sigma-MK00-eager_a_mk00_read.csv\", (\"B\", \"K0\", \"payload\", \"read\"): \"tmp/sigma-K0-eager_b_k0_read.csv\"}\n" + \
        "traffic = Traffic.buffetTraffic(bindings, formats, traces, 268435456, 32, {\"K0\": \"MK00\"})\n" + \
        "bindings = [{\"tensor\": \"A\", \"rank\": \"MK00\", \"format\": \"flattened\", \"type\": \"payload\", \"evict-on\": \"MK01\", \"style\": \"eager\", \"root\": \"MK00\"}, {\"tensor\": \"B\", \"rank\": \"K0\", \"format\": \"partitioned\", \"type\": \"payload\", \"evict-on\": \"N\", \"style\": \"eager\", \"root\": \"K0\"}]\n" + \
        "traces = {(\"A\", \"MK00\", \"payload\", \"read\"): \"tmp/sigma-MK00-eager_a_mk00_read.csv\", (\"B\", \"K0\", \"payload\", \"read\"): \"tmp/sigma-K0-eager_b_k0_read.csv\"}\n" + \
        "traffic = Traffic.buffetTraffic(bindings, formats, traces, 1048576, 4096, {\"K0\": \"MK00\"})\n" + \
        "metrics[\"Z\"][\"DataSRAMBanks\"] = {}\n" + \
        "metrics[\"Z\"][\"DataSRAMBanks\"][\"A\"] = {}\n" + \
        "metrics[\"Z\"][\"DataSRAMBanks\"][\"A\"][\"read\"] = 0\n" + \
        "metrics[\"Z\"][\"DataSRAMBanks\"][\"A\"][\"read\"] += traffic[0][\"A\"][\"read\"]\n" + \
        "metrics[\"Z\"][\"DataSRAMBanks\"][\"B\"] = {}\n" + \
        "metrics[\"Z\"][\"DataSRAMBanks\"][\"B\"][\"read\"] = 0\n" + \
        "metrics[\"Z\"][\"DataSRAMBanks\"][\"B\"][\"read\"] += traffic[0][\"B\"][\"read\"]\n" + \
        "metrics[\"Z\"][\"DataSRAMBanks\"][\"time\"] = (metrics[\"Z\"][\"DataSRAMBanks\"][\"A\"][\"read\"] + metrics[\"Z\"][\"DataSRAMBanks\"][\"B\"][\"read\"]) / 8246337208320\n" + \
        "metrics[\"Z\"][\"Multiplier\"] = {}\n" + \
        "metrics[\"Z\"][\"Multiplier\"][\"mul\"] = Metrics.dump()[\"Compute\"][\"payload_mul\"]\n" + \
        "metrics[\"Z\"][\"Multiplier\"][\"time\"] = metrics[\"Z\"][\"Multiplier\"][\"mul\"] / 8192000000000"

    assert collector.dump().gen(0) == hifiber


def test_dump_new_flattened_tensor_for_format():
    yaml = """
    einsum:
      declaration:
        Z: [K, M]
        A: [K, M]
      expressions:
        - Z[k, m] = A[k, m]
    mapping:
      partitioning:
        Z:
          (K, M): [flatten()]
    architecture:
      accel:
      - name: level0
        local:
        - name: Buffer
          class: Buffet
          attributes:
            width: 64
            depth: 1024
    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
      - component: Buffer
        bindings:
        - tensor: A
          rank: KM
          type: payload
          evict-on: root
          format: default
    format:
      A:
        default:
          rank-order: [KM]
          KM:
            format: C
            pbits: 32
    """
    collector = build_collector(yaml, 0)

    hifiber = "metrics = {}\n" + \
        "metrics[\"Z\"] = {}\n" + \
        "formats = {\"A\": Format(Tensor(rank_ids=[\"KM\"], shape=[K * M]), {\"rank-order\": [\"KM\"], \"KM\": {\"format\": \"C\", \"pbits\": 32}})}\n" + \
        "bindings = [{\"tensor\": \"A\", \"rank\": \"KM\", \"type\": \"payload\", \"evict-on\": \"root\", \"format\": \"default\", \"style\": \"lazy\"}]\n" + \
        "Traffic.filterTrace(\"tmp/Z-KM-populate_1.csv\", \"tmp/Z-KM-iter.csv\", \"tmp/Z-KM-populate_1_payload.csv\")\n" + \
        "traces = {(\"A\", \"KM\", \"payload\", \"read\"): \"tmp/Z-KM-populate_1_payload.csv\"}\n" + \
        "traffic = Traffic.buffetTraffic(bindings, formats, traces, 65536, 64)"

    assert collector.dump().gen(0) == hifiber


def test_dump_skip_zero_bits():
    yaml = """
    einsum:
      declaration:
        Z: [K, M]
        A: [K, M]
      expressions:
        - Z[k, m] = A[k, m]
    architecture:
      accel:
      - name: level0
        local:
        - name: Buffer
          class: Buffet
          attributes:
            width: 64
            depth: 1024
    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
      - component: Buffer
        bindings:
        - tensor: Z
          rank: K
          type: payload
          style: eager
          evict-on: root
          format: default
        - tensor: A
          rank: K
          type: coord
          style: eager
          evict-on: root
          format: default
    format:
      Z:
        default:
          rank-order: [K, M]
          K:
            format: C
            cbits: 0
          M:
            format: C
            pbits: 32
      A:
        default:
          rank-order: [K, M]
          K:
            format: C
            pbits: 0
          M:
            format: C
            pbits: 32
    """
    collector = build_collector(yaml, 0)

    hifiber = "metrics = {}\n" + \
        "metrics[\"Z\"] = {}\n" + \
        "formats = {\"Z\": Format(Z_KM, {\"rank-order\": [\"K\", \"M\"], \"K\": {\"format\": \"C\", \"cbits\": 0}, \"M\": {\"format\": \"C\", \"pbits\": 32}}), \"A\": Format(A_KM, {\"rank-order\": [\"K\", \"M\"], \"K\": {\"format\": \"C\", \"pbits\": 0}, \"M\": {\"format\": \"C\", \"pbits\": 32}})}\n" + \
        "bindings = [{\"tensor\": \"Z\", \"evict-on\": \"root\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"K\", \"rank\": \"M\", \"type\": \"payload\"}, {\"tensor\": \"A\", \"evict-on\": \"root\", \"style\": \"eager\", \"format\": \"default\", \"root\": \"K\", \"rank\": \"M\", \"type\": \"payload\"}]\n" + \
        "traces = {(\"Z\", \"M\", \"payload\", \"read\"): \"tmp/Z-M-eager_z_k_read.csv\", (\"Z\", \"M\", \"payload\", \"write\"): \"tmp/Z-M-eager_z_k_write.csv\", (\"A\", \"M\", \"payload\", \"read\"): \"tmp/Z-M-eager_a_k_read.csv\"}\n" + \
        "traffic = Traffic.buffetTraffic(bindings, formats, traces, 65536, 64)"

    assert collector.dump().gen(0) == hifiber


def test_end():
    hifiber = "Metrics.endCollect()"

    assert Collector.end().gen(0) == hifiber


def test_make_body_none():
    yaml = build_extensor_yaml()
    collector = build_collector(yaml, 0)

    hifiber = ""

    assert collector.make_body().gen(0) == hifiber


def test_make_body_iter_num():
    yaml = """
    einsum:
      declaration:
        Z: [K, M]
        A: [K, M]
      expressions:
        - Z[k, m] = A[k, m]
    architecture:
      accel:
      - name: level0
        local:
        - name: Buffer
          class: Buffet
    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
      - component: Buffer
        bindings:
        - tensor: Z
          rank: K
          type: payload
          style: eager
          evict-on: root
          format: default
    format:
      Z:
        default:
          rank-order: [K, M]
          K:
            format: C
            pbits: 32
          M:
            format: C
            cbits: 32
            pbits: 64
    """
    collector = build_collector(yaml, 0)

    hifiber = "m_iter_num = Metrics.getIter().copy()"

    assert collector.make_body().gen(0) == hifiber


def test_make_loop_footer_unconfigured():
    yaml = build_gamma_yaml()
    collector = build_collector(yaml, 0)

    with pytest.raises(ValueError) as excinfo:
        collector.make_loop_footer("K")
    assert str(
        excinfo.value) == "Unconfigured collector. Make sure to first call start()"


def test_make_loop_footer():
    yaml = build_gamma_yaml()
    collector = build_collector(yaml, 0)
    collector.start()

    hifiber = "Intersect_K.addTraces(Metrics.consumeTrace(\"K\", \"intersect_2\"))"
    assert collector.make_loop_footer("K").gen(0) == hifiber

    yaml = build_extensor_yaml()
    collector = build_collector(yaml, 0)
    collector.start()

    program = collector.program
    part_ir = program.get_partitioning()
    for tensor in program.get_equation().get_tensors():
        tensor.update_ranks(
            part_ir.partition_ranks(
                tensor.get_ranks(),
                part_ir.get_all_parts(),
                True,
                True))
        program.get_loop_order().apply(tensor)

    assert collector.make_loop_footer("M2").gen(0) == ""

    hifiber = "n0_iter_num = Metrics.getIter().copy()\n" + \
        "K0Intersection_K0.addTraces(Metrics.consumeTrace(\"K0\", \"intersect_0\"), Metrics.consumeTrace(\"K0\", \"intersect_1\"))"

    assert collector.make_loop_footer("K0").gen(0) == hifiber

    hifiber = "K1Intersect_K1.addTraces(Metrics.consumeTrace(\"K1\", \"intersect_0\"), Metrics.consumeTrace(\"K1\", \"intersect_1\"))\n" + \
        "z_k1.trace(\"eager_z_k1_write\", iteration_num=n0_iter_num)"

    assert collector.make_loop_footer("K1").gen(0) == hifiber


def test_make_loop_header_unconfigured():
    yaml = build_gamma_yaml()
    collector = build_collector(yaml, 0)

    with pytest.raises(ValueError) as excinfo:
        collector.make_loop_header("K")
    assert str(
        excinfo.value) == "Unconfigured collector. Make sure to first call start()"


def test_make_loop_header():
    yaml = build_extensor_yaml()
    collector = build_collector(yaml, 0)
    collector.start()

    assert collector.make_loop_header("N2").gen(0) == ""

    hifiber = "eager_a_m0_read = set()\n" + \
        "eager_z_m0_read = set()"

    assert collector.make_loop_header("M1").gen(0) == hifiber

    hifiber_option1 = "if () not in eager_z_m0_read:\n" + \
        "    eager_z_m0_read.add(())\n" + \
        "    z_m0.trace(\"eager_z_m0_read\")\n" + \
        "if () not in eager_a_m0_read:\n" + \
        "    eager_a_m0_read.add(())\n" + \
        "    a_m0.trace(\"eager_a_m0_read\")"

    hifiber_option2 = "if () not in eager_a_m0_read:\n" + \
        "    eager_a_m0_read.add(())\n" + \
        "    a_m0.trace(\"eager_a_m0_read\")\n" + \
        "if () not in eager_z_m0_read:\n" + \
        "    eager_z_m0_read.add(())\n" + \
        "    z_m0.trace(\"eager_z_m0_read\")"

    assert collector.make_loop_header("M0").gen(
        0) in {hifiber_option1, hifiber_option2}

# TODO: Actually compute the exectution time in TeAAL
# TODO: Multiply buffer size by number of instances


def test_make_loop_header_eager_root():
    yaml = """
    einsum:
      declaration:
        A: [K]
        Z: [M]
      expressions:
      - Z[M] = A[k]
    mapping:
      loop-order:
        Z: [K, M]
    format:
      Z:
        default:
          rank-order: [M]
          M:
            format: U
            pbits: 32
    architecture:
      Accelerator:
      - name: System
        attributes:
          clock_frequency: 1000
        local:
        - name: MainMemory
          class: DRAM
          attributes:
            bandwidth: 2048
        subtree:
        - name: Chip
          local:
          - name: Buffer
            class: Buffet
            attributes:
              width: 32
              depth: 128
    bindings:
      Z:
      - config: Accelerator
        prefix: tmp/eager_root
      - component: MainMemory
        bindings:
        - tensor: Z
          rank: M
          type: payload
          format: default
      - component: Buffer
        bindings:
        - tensor: Z
          rank: M
          type: payload
          format: default
          style: eager
          evict-on: root
    """
    collector = build_collector(yaml, 0)
    collector.start()

    hifiber = "z_k.trace(\"eager_z_k_write\", iteration_num=m_iter_num)"
    assert collector.make_loop_footer("K").gen(0) == hifiber


def test_register_ranks():
    yaml = build_gamma_yaml()
    collector = build_collector(yaml, 0)

    hifiber = "Metrics.registerRank(\"M\")\n" + \
        "Metrics.registerRank(\"K\")\n" + \
        "Metrics.registerRank(\"N\")"

    assert collector.register_ranks().gen(0) == hifiber


def test_set_collecting_type_err():
    yaml = build_gamma_yaml()
    collector = build_collector(yaml, 0)

    with pytest.raises(ValueError) as excinfo:
        collector.set_collecting(None, "K", "fiber", False, True)
    assert str(
        excinfo.value) == "Tensor must be specified for trace type fiber"


def test_set_collecting():
    yaml = build_gamma_yaml()
    collector = build_collector(yaml, 0)

    hifiber = "Metrics.trace(\"K\", type_=\"intersect_3\", consumable=False)"
    assert collector.set_collecting(
        "B", "K", "fiber", False, True).gen(0) == hifiber

    hifiber = "Metrics.trace(\"K\", type_=\"iter\", consumable=False)"
    assert collector.set_collecting(
        None, "K", "iter", False, True).gen(0) == hifiber


def test_set_collecting_eager():
    yaml = build_extensor_yaml()
    collector = build_collector(yaml, 0)

    program = collector.program
    part_ir = program.get_partitioning()
    for tensor in program.get_equation().get_tensors():
        tensor.update_ranks(
            part_ir.partition_ranks(
                tensor.get_ranks(),
                part_ir.get_all_parts(),
                True,
                True))
        program.get_loop_order().apply(tensor)

    hifiber = "Metrics.trace(\"N0\", type_=\"eager_a_m0_read\", consumable=False)"
    assert collector.set_collecting(
        "A", "N0", "M0", False, True).gen(0) == hifiber

    hifiber = "n0_iter_num = None\n" + \
        "Metrics.trace(\"N0\", type_=\"eager_z_m0_write\", consumable=False)"
    assert collector.set_collecting(
        "Z", "N0", "M0", False, False).gen(0) == hifiber


def test_start():
    yaml = build_gamma_yaml()
    collector = build_collector(yaml, 0)

    generated = collector.start().gen(0).split("\n")

    corr = ["Metrics.beginCollect(\"tmp/gamma_T\")"]
    check_hifiber_lines(generated[:1], corr)

    corr = ["Intersect_K = LeaderFollowerIntersector()"]
    check_hifiber_lines(generated[1:2], corr)

    corr = ["Metrics.trace(\"K\", type_=\"iter\", consumable=False)",
            "Metrics.trace(\"K\", type_=\"intersect_2\", consumable=True)",
            "Metrics.trace(\"M\", type_=\"iter\", consumable=False)",
            "Metrics.trace(\"M\", type_=\"populate_1\", consumable=False)",
            "Metrics.trace(\"K\", type_=\"intersect_2\", consumable=False)",
            "Metrics.trace(\"K\", type_=\"intersect_3\", consumable=False)",
            "Metrics.trace(\"N\", type_=\"iter\", consumable=False)",
            "Metrics.trace(\"N\", type_=\"populate_1\", consumable=False)"]
    check_hifiber_lines(generated[2:], corr)


def test_start_sequencer():
    yaml = build_extensor_energy_yaml()
    collector = build_collector(yaml, 0)

    generated = collector.start().gen(0).split("\n")

    corr = ['Metrics.beginCollect("tmp/extensor_energy")']
    check_hifiber_lines(generated[:1], corr)

    corr = ["K2Intersect_K2 = SkipAheadIntersector()",
            "K1Intersect_K1 = SkipAheadIntersector()",
            "K0Intersection_K0 = SkipAheadIntersector()"]
    check_hifiber_lines(generated[1:4], corr)

    corr = [
        "Metrics.trace(\"N2\", type_=\"iter\", consumable=False)",
        "Metrics.trace(\"K2\", type_=\"iter\", consumable=False)",
        "Metrics.trace(\"M2\", type_=\"iter\", consumable=False)",
        "Metrics.trace(\"M1\", type_=\"iter\", consumable=False)",
        "Metrics.trace(\"N1\", type_=\"iter\", consumable=False)",
        "Metrics.trace(\"K1\", type_=\"iter\", consumable=False)",
        "Metrics.trace(\"M0\", type_=\"iter\", consumable=False)",
        "Metrics.trace(\"N0\", type_=\"iter\", consumable=False)",
        "Metrics.trace(\"K0\", type_=\"iter\", consumable=False)",
        "Metrics.trace(\"N0\", type_=\"eager_z_m0_read\", consumable=False)",
        "n0_iter_num = None",
        "Metrics.trace(\"N0\", type_=\"eager_z_m0_write\", consumable=False)",
        "Metrics.trace(\"M0\", type_=\"eager_z_m0_read\", consumable=False)",
        "n0_iter_num = None",
        "Metrics.trace(\"M0\", type_=\"eager_z_m0_write\", consumable=False)",
        "Metrics.trace(\"K0\", type_=\"eager_a_m0_read\", consumable=False)",
        "Metrics.trace(\"K1\", type_=\"intersect_0\", consumable=True)",
        "Metrics.trace(\"M0\", type_=\"eager_a_m0_read\", consumable=False)",
        "Metrics.trace(\"K0\", type_=\"intersect_0\", consumable=True)",
        "Metrics.trace(\"K1\", type_=\"intersect_0\", consumable=False)",
        "Metrics.trace(\"K2\", type_=\"intersect_0\", consumable=True)",
        "Metrics.trace(\"N1\", type_=\"populate_1\", consumable=False)",
        "Metrics.trace(\"K1\", type_=\"intersect_1\", consumable=True)",
        "Metrics.trace(\"N0\", type_=\"eager_b_n0_read\", consumable=False)",
        "Metrics.trace(\"K0\", type_=\"intersect_1\", consumable=True)",
        "Metrics.trace(\"K1\", type_=\"intersect_1\", consumable=False)",
        "Metrics.trace(\"K2\", type_=\"intersect_1\", consumable=True)",
        "Metrics.trace(\"K0\", type_=\"eager_b_n0_read\", consumable=False)"]
    check_hifiber_lines(generated[4:32], corr)

    corr = ["Metrics.registerRank(\"N2\")",
            "Metrics.registerRank(\"K2\")",
            "Metrics.registerRank(\"M2\")",
            "Metrics.registerRank(\"M1\")",
            "Metrics.registerRank(\"N1\")",
            "Metrics.registerRank(\"K1\")",
            "Metrics.registerRank(\"M0\")",
            "Metrics.registerRank(\"N0\")",
            "Metrics.registerRank(\"K0\")"]
    check_hifiber_lines(generated[32:], corr)


def test_start_flattening():
    yaml = build_sigma_yaml()
    collector = build_collector(yaml, 0)

    generated = collector.start().gen(0).split("\n")

    corr = ["Metrics.beginCollect(\"tmp/sigma\")",
            "Metrics.associateShape(\"MK01\", (M, K))",
            "Metrics.matchRanks(\"MK00\", \"M\")",
            "Metrics.matchRanks(\"MK00\", \"K0\")",
            "Metrics.associateShape(\"MK00\", (M, K))"]
    check_hifiber_lines(generated[:5], corr)

    corr = [
        "Metrics.trace(\"MK00\", type_=\"eager_a_mk00_read\", consumable=False)",
        "Metrics.trace(\"K0\", type_=\"eager_b_k0_read\", consumable=False)"]
    check_hifiber_lines(generated[5:7], corr)

    corr = ["Metrics.registerRank(\"K1\")",
            "Metrics.registerRank(\"MK01\")",
            "Metrics.registerRank(\"N\")",
            "Metrics.registerRank(\"MK00\")"]
    check_hifiber_lines(generated[7:], corr)


def test_trace_tree():
    yaml = build_extensor_yaml()
    collector = build_collector(yaml, 0)

    program = collector.program
    part_ir = program.get_partitioning()
    for tensor in program.get_equation().get_tensors():
        tensor.update_ranks(
            part_ir.partition_ranks(
                tensor.get_ranks(),
                part_ir.get_all_parts(),
                True,
                True))
        program.get_loop_order().apply(tensor)

    hifiber = "if (m1, k1) not in eager_a_m0_read:\n" + \
        "    eager_a_m0_read.add((m1, k1))\n" + \
        "    a_m0.trace(\"eager_a_m0_read\")"
    assert collector.trace_tree("A", "M0", True).gen(0) == hifiber

    hifiber = "z_m0.trace(\"eager_z_m0_write\", iteration_num=n0_iter_num)"
    assert collector.trace_tree("Z", "M0", False).gen(0) == hifiber
