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
