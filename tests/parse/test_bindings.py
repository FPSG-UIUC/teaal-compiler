from teaal.parse.bindings import Bindings


def test_empty():
    bindings = Bindings.from_str("")
    assert bindings.get("BAD") == []


def test_no_bindings():
    yaml = """
    foo:
      - bar
      - baz
    """
    assert Bindings.from_str(yaml).get("BAD") == []


def test_defined():
    bindings = Bindings.from_file("tests/integration/test_bindings.yaml")
    mem = [{"tensor": "A", "rank": "root"}, {"tensor": "Z", "rank": "root"}]
    regs = [{"tensor": "A", "rank": "M"}, {"tensor": "Z", "rank": "M"}]
    mac = [{"einsum": "Z", "op": "add"}]

    assert bindings.get("Memory") == mem
    assert bindings.get("Registers") == regs
    assert bindings.get("MAC") == mac
    assert bindings.get("BAD") == []
