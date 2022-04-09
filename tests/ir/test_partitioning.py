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


def test_nway_after_dyn():
    all_parts = """
                M: [uniform_occupancy(A.6), nway_shape(20)]
    """
    dict_ = parse_partitioning(all_parts)["Z"]
    with pytest.raises(ValueError) as excinfo:
        partitioning = Partitioning(dict_, ["M", "N", "K"])

    assert str(
        excinfo.value) == "N-way partitioning after dynamic partitioning on rank M"


def test_get_all_partitioning():
    all_parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    dict_ = parse_partitioning(all_parts)["Z"]
    partitioning = Partitioning(dict_, ["M", "N", "K"])
    assert partitioning.get_all_parts() == dict_


def test_mixed_partitioning():
    parts = """
                K:
                    - uniform_shape(500)
                    - uniform_shape(250)
                    - uniform_occupancy(A.100)
                    - uniform_occupancy(A.50)
                    - uniform_shape(10)
                    - uniform_occupancy(A.6)
                    - uniform_shape(4)
                    - uniform_shape(2)
    """
    dict_ = parse_partitioning(parts)["Z"]
    partitioning = Partitioning(dict_, ["M", "N", "K"])

    dict_["K6I"] = dict_["K"][-6:]
    dict_["K5I"] = dict_["K"][-5:]
    dict_["K3I"] = dict_["K"][-3:]
    assert partitioning.get_all_parts() == dict_


def test_get_dyn_rank():
    all_parts = """
                M: [uniform_occupancy(A.6)]
    """
    partitioning = Partitioning(
        parse_partitioning(all_parts)["Z"], ["M", "N", "K"])

    assert partitioning.get_dyn_rank("m") == "m0"
    assert partitioning.get_dyn_rank("m1") == "m1"
    assert partitioning.get_dyn_rank("n") == "n"


def test_get_dynamic_partitioning():
    all_parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = Partitioning(
        parse_partitioning(all_parts)["Z"], [
            "M", "N", "K"])

    dyn_parts = """
                M: [uniform_occupancy(A.6)]
    """
    dyn = parse_partitioning(dyn_parts)["Z"]

    assert partitioning.get_dyn_parts() == dyn


def test_get_final_rank_id():
    all_parts = """
                M: [uniform_occupancy(A.6), uniform_occupancy(A.3)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = Partitioning(
        parse_partitioning(all_parts)["Z"], [
            "M", "N", "K"])

    assert partitioning.get_final_rank_id("N") == "N2"
    assert partitioning.get_final_rank_id("N2") == "N2"
    assert partitioning.get_final_rank_id("M1I") == "M1"
    assert partitioning.get_final_rank_id("K") == "K"


def test_get_intermediates():
    all_parts = """
                K:
                    - uniform_shape(500)
                    - uniform_shape(250)
                    - uniform_occupancy(A.100)
                    - uniform_occupancy(A.50)
                    - uniform_shape(10)
                    - uniform_occupancy(A.6)
                    - uniform_shape(4)
                    - uniform_shape(2)
    """
    partitioning = Partitioning(
        parse_partitioning(all_parts)["Z"], ["M", "N", "K"])

    assert partitioning.get_intermediates("K") == ["K6I", "K5I", "K3I"]


def test_get_leader():
    parts = """
                M: [uniform_occupancy(A.6)]
    """
    partitioning = Partitioning(parse_partitioning(parts)["Z"], ["M"])

    part = partitioning.get_all_parts()["M"][0]
    assert partitioning.get_leader(part) == "A"


def test_get_leader_bad_style():
    parts = """
                M: [uniform_shape(6)]
    """
    partitioning = Partitioning(parse_partitioning(parts)["Z"], ["M"])

    part = partitioning.get_all_parts()["M"][0]
    with pytest.raises(ValueError) as excinfo:
        partitioning.get_leader(part)

    assert str(excinfo.value) == "Style uniform_shape has no leader"


def test_get_root_name():
    parts = """
                M: [uniform_shape(6)]
    """
    partitioning = Partitioning(parse_partitioning(parts)["Z"], ["M", "K"])
    assert partitioning.get_root_name("M1") == "M"
    assert partitioning.get_root_name("K") == "K"


def test_get_static_partitioning():
    all_parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = Partitioning(
        parse_partitioning(all_parts)["Z"], [
            "M", "N", "K"])

    static_parts = """
                K: [uniform_shape(4)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    static = parse_partitioning(static_parts)["Z"]

    assert partitioning.get_static_parts() == static


def test_get_tensor_spec():
    all_parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = Partitioning(
        parse_partitioning(all_parts)["Z"], [
            "M", "N", "K"])

    used_parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
    """
    used = parse_partitioning(used_parts)["Z"]

    tensor_ranks = ["J", "K", "M", "N"]

    assert partitioning.get_tensor_spec(tensor_ranks, {"K", "M"}) == used


def test_partition_names():
    all_parts = """
                M: [uniform_occupancy(A.6), uniform_occupancy(A.3)]
                N: [uniform_shape(2), nway_shape(7)]
                K:
                    - uniform_shape(500)
                    - uniform_shape(250)
                    - uniform_occupancy(A.100)
                    - uniform_occupancy(A.50)
                    - uniform_shape(10)
                    - uniform_occupancy(A.6)
                    - uniform_shape(4)
                    - uniform_shape(2)
    """
    partitioning = Partitioning(
        parse_partitioning(all_parts)["Z"], [
            "M", "N", "K"])

    assert partitioning.partition_names("M", True) == ["M0", "M1", "M2"]
    assert partitioning.partition_names("M", False) == ["M1I", "M2"]

    assert partitioning.partition_names("N", False) == ["N0", "N1", "N2"]

    assert partitioning.partition_names("K", False) == ["K6I", "K7", "K8"]
    assert partitioning.partition_names("K3I", False) == [
        "K0", "K1", "K2", "K3"]


def test_skip_empty_partitioning():
    all_parts = """
                K: []
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = Partitioning(
        parse_partitioning(all_parts)["Z"], [
            "M", "N", "K"])

    assert "K" not in partitioning.get_all_parts()


def test_eq():
    all_parts = """
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning1 = Partitioning(
        parse_partitioning(all_parts)["Z"], [
            "M", "N", "K"])

    partitioning2 = Partitioning(
        parse_partitioning(all_parts)["Z"], [
            "M", "N", "K"])

    assert partitioning1 == partitioning2


def test_neq_dyn_parts():
    parts1 = """
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning1 = Partitioning(
        parse_partitioning(parts1)["Z"], [
            "M", "N", "K"])

    parts2 = """
                M: [uniform_occupancy(A.5)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning2 = Partitioning(
        parse_partitioning(parts2)["Z"], [
            "M", "N", "K"])

    assert partitioning1 != partitioning2


def test_neq_static_parts():
    parts1 = """
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(3), nway_shape(7)]
    """
    partitioning1 = Partitioning(
        parse_partitioning(parts1)["Z"], [
            "M", "N", "K"])

    parts2 = """
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning2 = Partitioning(
        parse_partitioning(parts2)["Z"], [
            "M", "N", "K"])

    assert partitioning1 != partitioning2


def test_neq_obj():
    all_parts = """
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning1 = Partitioning(
        parse_partitioning(all_parts)["Z"], [
            "M", "N", "K"])

    assert partitioning1 != "foo"
