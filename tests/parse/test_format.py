import pytest

from teaal.parse.format import Format


def build_format():
    yaml = """
    format:
      A:
        init:
          rank-order: [M, K]
          M:
            format: U
            rhbits: 32
            pbits: 32
          K:
            format: C
            cbits: 32
            pbits: 64
        loop:
          rank-order: [K, M]
          K:
            format: C
          M:
            format: C
            cbits: 32
            pbits: 64
    """
    return Format.from_str(yaml)


def test_no_spec():
    Format.from_str("")


def test_no_format():
    Format.from_file("tests/integration/test_arch.yaml")


def test_missing_rank_order():
    yaml = """
    format:
      A:
        BAD:
          M:
            format: C
            pbits: 32
    """

    with pytest.raises(ValueError) as excinfo:
        Format.from_str(yaml)
    assert str(
        excinfo.value) == "Rank order not specified for tensor A in format BAD"


def test_format():
    format_ = build_format()
    spec = {
        "init": {
            "rank-order": ["M", "K"],
            "M": {
                "format": "U",
                "rhbits": 32,
                "pbits": 32},
            "K": {
                "format": "C",
                "cbits": 32,
                "pbits": 64}},
        "loop": {
            "rank-order": ["K", "M"],
            "K": {"format": "C"},
            "M": {
                "format": "C",
                "cbits": 32,
                "pbits": 64}}}

    assert format_.get_spec("A") == spec
    assert format_.get_spec("B") == {}
