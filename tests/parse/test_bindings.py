from es2hfa.parse.bindings import Bindings


def test_empty():
    bindings = Bindings.from_str("")
    assert bindings.get_components() == {}


def test_no_bindings():
    yaml = """
    foo:
      - bar
      - baz
    """
    assert Bindings.from_str(yaml).get_components() == {}


def test_defined():
    bindings = Bindings.from_file("tests/integration/test_bindings.yaml")
    components = {"Memory": [{"tensor": "A", "rank": "root"}, {
        "tensor": "Z", "rank": "root"}], "MAC": [{"einsum": "Z", "ops": ["add"]}]}

    assert bindings.get_components() == components
