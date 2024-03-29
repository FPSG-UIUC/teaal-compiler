from teaal.ir.component import *
from teaal.ir.level import Level


def build_local():
    attrs = {"datawidth": 8, "bandwidth": 128}
    bindings = {"Z": [{"tensor": "A", "rank": "M",
                       "format": "default", "type": "payload"}]}
    return DRAMComponent("DRAM", 1, attrs, bindings)


def build_level():
    name = "System"
    num = 1
    attrs = {"clock_frequency": 10 ** 9}
    local = [build_local()]
    subtrees = [build_subtree()]

    return Level(name, num, attrs, local, subtrees)


def build_subtree():
    return Level("PE", 8, {}, [FunctionalComponent("MAC", 8, {}, {})], [])


def test_get_attr():
    level = build_level()

    assert level.get_attr("clock_frequency") == 10 ** 9
    assert level.get_attr("foo") is None


def test_get_local():
    assert build_level().get_local() == [build_local()]


def test_get_name():
    assert build_level().get_name() == "System"


def test_get_num():
    assert build_level().get_num() == 1


def test_get_subtrees():
    assert build_level().get_subtrees() == [build_subtree()]


def test_eq():
    assert build_level() == build_level()
    assert build_level() != build_subtree()
    assert build_level() != "foo"


def test_repr():
    level = build_level()
    repr_ = "(Level, System, 1, {'clock_frequency': 1000000000}, [(DRAMComponent, DRAM, 1, {'Z': [{'tensor': 'A', 'rank': 'M', 'format': 'default', 'type': 'payload'}]}, 128)], [(Level, PE, 8, {}, [(FunctionalComponent, MAC, 8, {})], [])])"

    assert repr(level) == repr_
