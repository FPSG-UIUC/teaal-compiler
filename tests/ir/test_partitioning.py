import pytest

from es2hfa.ir.partitioning import Partitioning
from es2hfa.parse.mapping import Mapping


def parse_partitioning(parts):
    yaml = """
    mapping:
        partitioning:
            Z:
    """ + parts
    return Mapping.from_str(yaml).get_partitioning()


def test_mixed_partitioning():
    parts = """
                K: [uniform_occupancy(A.5), uniform_shape(4)]
    """
    dict_ = parse_partitioning(parts)

    with pytest.raises(ValueError) as excinfo:
        Partitioning(dict_["Z"])
    assert str(excinfo.value) \
        == "Dimension K cannot be partitioned both statically and dynamically"


def test_get_all_partitioning():
    all_parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    dict_ = parse_partitioning(all_parts)["Z"]
    partitioning = Partitioning(dict_)
    assert partitioning.get_all_parts() == dict_


def test_get_dynamic_partitioning():
    all_parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = Partitioning(parse_partitioning(all_parts)["Z"])

    dyn_parts = """
                M: [uniform_occupancy(A.6)]
    """
    dyn = parse_partitioning(dyn_parts)["Z"]

    assert partitioning.get_dyn_parts() == dyn


def test_get_static_partitioning():
    all_parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = Partitioning(parse_partitioning(all_parts)["Z"])

    static_parts = """
                K: [uniform_shape(4)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    static = parse_partitioning(static_parts)["Z"]

    assert partitioning.get_static_parts() == static


def test_skip_empty_partitioning():
    all_parts = """
                K: []
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = Partitioning(parse_partitioning(all_parts)["Z"])

    assert "K" not in partitioning.get_all_parts()
