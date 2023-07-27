import pytest

from teaal.parse.arch import Architecture


def build_local(name, class_, attrs):
    return {"name": name, "class": class_, "attributes": attrs}


def build_subtree(name, num, attrs, local, subtrees):
    return {
        "name": name,
        "num": num,
        "attributes": attrs,
        "local": local,
        "subtree": subtrees}


def test_bad_architecture():
    yaml = """
    architecture:
    - foo
    """

    with pytest.raises(ValueError) as excinfo:
        Architecture.from_str(yaml)
    assert str(
        excinfo.value) == "Bad architecture spec: {'architecture': ['foo']}"


def test_unnamed_subtree():
    yaml = """
    architecture:
      Config0:
      - attributes:
          foo: 1
    """

    with pytest.raises(ValueError) as excinfo:
        Architecture.from_str(yaml)
    assert str(excinfo.value) == "Unnamed subtree: {'attributes': {'foo': 1}}"


def test_unnamed_local():
    yaml = """
    architecture:
      Config0:
      - name: System
        local:
        - class: DRAM
    """

    with pytest.raises(ValueError) as excinfo:
        Architecture.from_str(yaml)
    assert str(excinfo.value) == "Unnamed local: {'class': 'DRAM'}"


def test_unclassed_local():
    yaml = """
    architecture:
      Config0:
      - name: System
        local:
        - name: Memory
    """

    with pytest.raises(ValueError) as excinfo:
        Architecture.from_str(yaml)
    assert str(excinfo.value) == "Unclassed local: {'name': 'Memory'}"


def test_empty():
    assert Architecture.from_str("").get_spec() is None


def test_no_arch():
    yaml = """
    foo:
      - bar
      - baz
    """
    assert Architecture.from_str(yaml).get_spec() is None


def test_unspecified():
    yaml = """
    architecture:
      subtree:
      - name: System
    """

    arch = Architecture.from_str(yaml)
    spec = {
        "architecture": {
            "subtree": [
                build_subtree(
                    "System",
                    1,
                    {},
                    [],
                    [])]}}

    assert arch.get_spec() == spec


def test_all_spec():
    regs = build_local("Registers", "Buffet", {})
    mac = build_local("MAC", "compute", {})
    subtree0 = build_subtree("PE", 8, {}, [regs, mac], [])

    mac0 = build_local("MAC0", "compute", {})
    mac1 = build_local("MAC1", "compute", {})
    subtree1 = build_subtree("PE", 8, {}, [regs, mac0, mac1], [])

    mem = build_local("Memory", "DRAM", {"datawidth": 8, "bandwidth": 128})
    attrs = {"clock_frequency": 10 ** 9}
    tree0 = build_subtree("System", 1, attrs, [mem], [subtree0])
    tree1 = build_subtree("System", 1, attrs, [mem], [subtree1])

    arch = Architecture.from_file("tests/integration/test_arch.yaml")
    spec = {"architecture": {"Config0": [tree0], "Config1": [tree1]}}

    assert arch.get_spec() == spec
