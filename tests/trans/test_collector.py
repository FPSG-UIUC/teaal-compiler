from es2hfa.ir.hardware import Hardware
from es2hfa.ir.metrics import Metrics
from es2hfa.ir.program import Program
from es2hfa.parse.arch import Architecture
from es2hfa.parse.bindings import Bindings
from es2hfa.parse.einsum import Einsum
from es2hfa.parse.mapping import Mapping
from es2hfa.trans.collector import Collector


def build_gamma_yaml():
    with open("tests/integration/gamma.yaml", "r") as f:
        return f.read()


def build_collector():
    yaml = build_gamma_yaml()
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)

    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    hardware = Hardware(arch, bindings)

    program.add_einsum(0)
    metrics = Metrics(program, hardware)
    return Collector(program, metrics)


def test_end():
    hfa = "Metrics.endCollect()"

    assert Collector.end().gen(0) == hfa


def test_set_collecting():
    collector = build_collector()
    hfa = "B_KN.setCollecting(\"K\", True)"

    assert collector.set_collecting("B", "K").gen(0) == hfa


def test_start():
    collector = build_collector()
    hfa = "Metrics.beginCollect([\"M\", \"K\", \"N\"])"

    assert collector.start().gen(0) == hfa
