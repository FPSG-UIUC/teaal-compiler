from teaal.parse.bindings import Bindings


def test_empty():
    bindings = Bindings.from_str("")
    assert bindings.get("BAD") == {}


def test_no_bindings():
    yaml = """
    foo:
      - bar
      - baz
    """
    assert Bindings.from_str(yaml).get("BAD") == {}


def test_defined():
    bindings = Bindings.from_file("tests/integration/test_bindings.yaml")
    mem = {"Z": [{"tensor": "A", "rank": "M", "format": "A_default", "type": "payload"}, {
        "tensor": "Z", "rank": "M", "type": "payload", "format": "Z_default"}]}
    regs = {"Z": [{"tensor": "A",
                   "rank": "M",
                   "format": "A_default",
                   "type": "payload",
                   "style": "eager",
                   "evict-on": "M"},
                  {"tensor": "Z",
                   "rank": "M",
                   "format": "Z_default",
                   "rank": "M",
                   "type": "payload",
                   "evict-on": "root"}]}
    mac = {"Z": [{"op": "add"}]}

    assert bindings.get_config("Z") == "Config0"
    assert bindings.get("Memory") == mem
    assert bindings.get("Registers") == regs
    assert bindings.get("MAC") == mac
    assert bindings.get("BAD") == {}
