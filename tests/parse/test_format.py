import pytest

from teaal.parse.format import Format


def build_format():
    yaml = """
    format:
      A:
        M:
          format: U
          rhbits: 32
          pbits: 32
        K:
          format: C
          cbits: 32
          pbits: 64
    """
    return Format.from_str(yaml)


def test_no_spec():
    Format.from_str("")


def test_no_format():
    Format.from_file("tests/integration/test_arch.yaml")


def test_missing_tensor():
    format_ = build_format()

    with pytest.raises(ValueError) as excinfo:
        format_.get_spec("B")
    assert str(
        excinfo.value) == "Format unspecified for tensor B"


def test_format():
    format_ = build_format()
    spec = {
        "M": {
            "format": "U",
            "rhbits": 32,
            "pbits": 32},
        "K": {
            "format": "C",
            "cbits": 32,
            "pbits": 64}}

    assert format_.get_spec("A") == spec
