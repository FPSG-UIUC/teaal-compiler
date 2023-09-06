import pytest

from teaal.parse.bindings import Bindings


def test_empty():
    bindings = Bindings.from_str("")
    assert bindings.get_component("BAD") == {}


def test_no_bindings():
    yaml = """
    foo:
      - bar
      - baz
    """
    assert Bindings.from_str(yaml).get_component("BAD") == {}


def test_no_config():
    yaml = """
    bindings:
      Z:
      - component: foo
        bindings:
        - tensor: bar
    """
    with pytest.raises(ValueError) as excinfo:
        Bindings.from_str(yaml)
    assert str(
        excinfo.value) == "Accelerator config and prefix missing for Einsum Z"


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
    assert bindings.get_prefix("Z") == "tmp/Z"
    assert not bindings.get_energy("Z")

    assert bindings.get_component("Memory") == mem
    assert bindings.get_component("Registers") == regs
    assert bindings.get_component("MAC") == mac
    assert bindings.get_component("BAD") == {}

    assert bindings.get_bindings() == {
        "Z": {
            "Memory": mem["Z"],
            "Registers": regs["Z"],
            "MAC": mac["Z"]}}

def test_get_energy():
  yaml = """
  bindings:
    Z:
    - config: default
      prefix: tmp/Z
      energy: True
  """
  bindings = Bindings.from_str(yaml)
  assert bindings.get_energy("Z")
