import pytest
from sympy import symbols

from teaal.ir.partitioning import Partitioning
from teaal.ir.tensor import Tensor
from teaal.parse.mapping import Mapping


def parse_partitioning(parts):
    yaml = """
    mapping:
        partitioning:
            Z:
    """ + parts
    return Mapping.from_str(yaml).get_partitioning()


def build_part_dict(parts):
    parsed = parse_partitioning(parts)
    return {tuple(str(child) for child in key.children)
                  : val for key, val in parsed["Z"].items()}


def build_partitioning(parts):
    dict_ = parse_partitioning(parts)["Z"]

    j, k, m, n = symbols("j k m n")
    eqn_exprs = {j: j, k: k, m: m, n: n}

    return Partitioning(dict_, ["J", "M", "N", "K"], eqn_exprs)


def build_partitioning_conv(parts):
    dict_ = parse_partitioning(parts)["Z"]

    p, q, s, w = symbols("p q s w")
    eqn_exprs = {p: p, q: q, s: s, w: q + s}

    return Partitioning(dict_, ["P", "Q", "S", "W"], eqn_exprs)


def test_nway_after_dyn():
    all_parts = """
                M: [uniform_occupancy(A.6), nway_shape(20)]
    """
    dict_ = parse_partitioning(all_parts)["Z"]

    k, m, n = symbols("k m n")
    eqn_exprs = {k: k, m: m, n: n}

    with pytest.raises(ValueError) as excinfo:
        partitioning = Partitioning(dict_, ["M", "N", "K"], eqn_exprs)

    assert str(
        excinfo.value) == "N-way partitioning after dynamic partitioning on rank(s) ('M',)"


def test_check_flatten_single_rank_ops():
    all_parts = """
                (M, K): [nway_shape(5), uniform_shape(20)]
    """
    dict_ = parse_partitioning(all_parts)["Z"]

    k, m, n = symbols("k m n")
    eqn_exprs = {k: k, m: m, n: n}

    with pytest.raises(ValueError) as excinfo:
        partitioning = Partitioning(dict_, ["M", "N", "K"], eqn_exprs)

    assert str(
        excinfo.value) == "Operations ['nway_shape', 'uniform_shape'] can only be applied to one rank; not ('M', 'K')"


def test_check_flatten_alone():
    all_parts = """
                (M, K): [flatten(), uniform_shape(20)]
    """
    dict_ = parse_partitioning(all_parts)["Z"]

    k, m, n = symbols("k m n")
    eqn_exprs = {k: k, m: m, n: n}

    with pytest.raises(ValueError) as excinfo:
        partitioning = Partitioning(dict_, ["M", "N", "K"], eqn_exprs)

    assert str(
        excinfo.value) == "flatten() combined with other operators on rank(s) ('M', 'K')"


def test_check_flatten_multiple_ranks():
    all_parts = """
                K: [flatten()]
    """
    dict_ = parse_partitioning(all_parts)["Z"]

    k, m, n = symbols("k m n")
    eqn_exprs = {k: k, m: m, n: n}

    with pytest.raises(ValueError) as excinfo:
        partitioning = Partitioning(dict_, ["M", "N", "K"], eqn_exprs)

    assert str(
        excinfo.value) == "flatten() must combine at least two ranks; only ('K',) specified"


def test_check_flatten_index_math():
    all_parts = """
                (B, Q): [flatten()]
    """
    dict_ = parse_partitioning(all_parts)["Z"]

    b, q, s, w = symbols("b q s w")
    eqn_exprs = {b: b, q: q, s: s, w: q + s}

    with pytest.raises(ValueError) as excinfo:
        partitioning = Partitioning(dict_, ["B", "Q", "S", "W"], eqn_exprs)

    assert str(
        excinfo.value) == "Cannot flatten rank Q because it is used in index math"


def test_check_flatten_multiple_partitionings():
    all_parts = """
                K: [uniform_shape(10)]
                (M, K): [flatten()]
    """
    dict_ = parse_partitioning(all_parts)["Z"]

    k, m, n = symbols("k m n")
    eqn_exprs = {k: k, m: m, n: n}

    with pytest.raises(ValueError) as excinfo:
        partitioning = Partitioning(dict_, ["M", "N", "K"], eqn_exprs)

    assert str(
        excinfo.value) == "Cannot flatten rank K because it will also be independently partitioned"


def test_check_flatten_flattened_rank():
    all_parts = """
                (K, N): [flatten()]
                (M, KN): [flatten()]
    """
    dict_ = parse_partitioning(all_parts)["Z"]

    k, m, n = symbols("k m n")
    eqn_exprs = {k: k, m: m, n: n}

    with pytest.raises(ValueError) as excinfo:
        partitioning = Partitioning(dict_, ["M", "N", "K"], eqn_exprs)

    assert str(
        excinfo.value) == "Cannot flatten rank KN because it is a flattened rank"


def test_check_flatten_not_bottom_rank():
    all_parts = """
                K: [uniform_shape(10)]
                (M, K1): [flatten()]
    """
    dict_ = parse_partitioning(all_parts)["Z"]

    k, m, n = symbols("k m n")
    eqn_exprs = {k: k, m: m, n: n}

    with pytest.raises(ValueError) as excinfo:
        partitioning = Partitioning(dict_, ["M", "N", "K"], eqn_exprs)

    assert str(
        excinfo.value) == "Cannot flatten rank K1 because it will have multiple partitionings"


def test_only_occupancy_after_flattening():
    all_parts = """
                K: [uniform_occupancy(A.4)]
                (M, K0): [flatten()]
                MK0: [uniform_shape(5)]
    """
    dict_ = parse_partitioning(all_parts)["Z"]

    k, m, n = symbols("k m n")
    eqn_exprs = {k: k, m: m, n: n}

    with pytest.raises(ValueError) as excinfo:
        partitioning = Partitioning(dict_, ["M", "N", "K"], eqn_exprs)

    assert str(
        excinfo.value) == "Shape-based partitioning found on rank MK0 after flattening"


def test_multiple_partitionings_on_same_rank():
    all_parts = """
                Q: [uniform_shape(4)]
                S: [uniform_occupancy(A.6)]
    """
    dict_ = parse_partitioning(all_parts)["Z"]

    q, s, w = symbols("q s w")
    eqn_exprs = {q: q, s: s, w: q + s}

    with pytest.raises(ValueError) as excinfo:
        partitioning = Partitioning(dict_, ["Q", "S", "W"], eqn_exprs)

    assert str(
        excinfo.value) == "Cannot partition W with multiple specifications. " + \
        "Partitioning specified by S, Q" or \
        str(excinfo.value) == "Cannot partition W with multiple specifications. " + \
        "Partitioning specified by Q, S"


def test_get_all_partitioning():
    all_parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = build_partitioning(all_parts)
    corr = {("K",), ("M",), ("N",)}
    assert partitioning.get_all_parts() == corr


def test_get_all_partitioning_flatten():
    all_parts = """
                K: [uniform_occupancy(A.4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
    """
    partitioning = build_partitioning(all_parts)
    corr = {("K",), ("M", "K0"), ("MK0",)}
    assert partitioning.get_all_parts() == corr


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
    partitioning = build_partitioning(parts)

    corr = {("K",), ("K3I",), ("K5I",), ("K6I",)}
    assert partitioning.get_all_parts() == corr


def test_get_dyn_rank_flattening():
    all_parts = """
                K: [uniform_shape(4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
    """
    partitioning = build_partitioning(all_parts)

    with pytest.raises(ValueError) as excinfo:
        partitioning.get_dyn_rank("MK0")

    assert str(
        excinfo.value) == "Should never be used for flattened ranks, used on rank MK0"


def test_get_dyn_rank():
    all_parts = """
                M: [uniform_occupancy(A.6)]
    """
    partitioning = build_partitioning(all_parts)

    assert partitioning.get_dyn_rank("M") == "M0"
    assert partitioning.get_dyn_rank("M1") == "M1"
    assert partitioning.get_dyn_rank("N") == "N"


def test_get_dyn_rank_conv():
    all_parts = """
                Q: [uniform_occupancy(I.6)]
    """
    partitioning = build_partitioning_conv(all_parts)

    assert partitioning.get_dyn_rank("Q") == "Q0"
    assert partitioning.get_dyn_rank("W") == "W0"
    assert partitioning.get_dyn_rank("S") == "S"


def test_get_dynamic_partitioning():
    all_parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = build_partitioning(all_parts)

    dyn = {("M",)}
    assert partitioning.get_dyn_parts() == dyn


def test_get_dynamic_partitioning_flattening():
    all_parts = """
                K: [uniform_shape(4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
    """
    partitioning = build_partitioning(all_parts)
    assert partitioning.get_dyn_parts() == {("MK0",)}


def test_get_dynamic_partitioning_flattening_dyn():
    all_parts = """
                K: [uniform_occupancy(A.4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
    """
    partitioning = build_partitioning(all_parts)
    assert partitioning.get_dyn_parts() == {("K",), ("M", "K0"), ("MK0",)}


def test_get_final_rank_id():
    all_parts = """
                M: [uniform_occupancy(A.6), uniform_occupancy(A.3)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = build_partitioning(all_parts)

    assert partitioning.get_final_rank_id(Tensor("B", ["K", "N"]), "N") == "N2"
    assert partitioning.get_final_rank_id(
        Tensor("B", ["K", "N"]), "N2") == "N2"
    assert partitioning.get_final_rank_id(
        Tensor("A", ["K", "M"]), "M1I") == "M1"
    assert partitioning.get_final_rank_id(Tensor("B", ["K", "N"]), "K") == "K"


def test_final_rank_id_flattening():
    all_parts = """
                K: [uniform_shape(4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
    """
    partitioning = build_partitioning(all_parts)

    assert partitioning.get_final_rank_id(
        Tensor("A", ["K", "M"]), "MK00") == "MK00"
    assert partitioning.get_final_rank_id(
        Tensor("A", ["K", "M"]), "MK01") == "MK01"
    assert partitioning.get_final_rank_id(
        Tensor("A", ["K", "M"]), "MK0") == "MK01"
    assert partitioning.get_final_rank_id(
        Tensor("A", ["K", "M"]), "M") == "MK01"
    assert partitioning.get_final_rank_id(
        Tensor("Z", ["M", "N"]), "M") == "MK00"
    assert partitioning.get_final_rank_id(
        Tensor("A", ["K", "M"]), "K0") == "MK01"
    assert partitioning.get_final_rank_id(
        Tensor("B", ["K", "N"]), "K0") == "MK00"
    assert partitioning.get_final_rank_id(
        Tensor("B", ["K", "N"]), "K1") == "K1"
    assert partitioning.get_final_rank_id(Tensor("B", ["K", "N"]), "N") == "N"


def test_get_final_rank_id_conv():
    all_parts = """
                Q: [uniform_occupancy(A.6), uniform_occupancy(A.3)]
    """
    partitioning = build_partitioning_conv(all_parts)

    assert partitioning.get_final_rank_id(Tensor("I", ["W"]), "W") == "Q2"
    assert partitioning.get_final_rank_id(Tensor("I", ["W"]), "Q2") == "Q2"
    assert partitioning.get_final_rank_id(Tensor("I", ["W"]), "W1I") == "Q1"
    assert partitioning.get_final_rank_id(Tensor("I", ["W"]), "W0") == "W0"
    assert partitioning.get_final_rank_id(Tensor("F", ["S"]), "S") == "S"


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
    partitioning = build_partitioning(all_parts)

    assert partitioning.get_intermediates(
        Tensor(
            "A", [
                "K", "M"]), "K") == [
        "K6I", "K5I", "K3I"]


def test_get_intermediates_flattening():
    all_parts = """
                K: [uniform_occupancy(A.12), uniform_occupancy(A.6)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
    """
    partitioning = build_partitioning(all_parts)

    assert partitioning.get_intermediates(
        Tensor(
            "A", [
                "K", "M"]), "K") == [
        "K1I", "K0", "MK0"]
    assert partitioning.get_intermediates(
        Tensor("B", ["K", "N"]), "K") == ["K1I"]
    assert partitioning.get_intermediates(Tensor("Z", ["M", "N"]), "M") == []


def test_get_intermediates_conv():
    all_parts = """
                Q:
                    - uniform_shape(500)
                    - uniform_shape(250)
                    - uniform_occupancy(A.100)
                    - uniform_occupancy(A.50)
                    - uniform_shape(10)
                    - uniform_occupancy(A.6)
                    - uniform_shape(4)
                    - uniform_shape(2)
    """
    partitioning = build_partitioning_conv(all_parts)

    assert partitioning.get_intermediates(Tensor("I", ["W"]), "W") == [
        "W6I", "W5I", "W3I"]


def test_get_leader():
    parts = """
                M: [uniform_occupancy(A.6)]
    """
    partitioning = build_partitioning(parts)
    assert partitioning.get_leader("M", "M1") == "A"


def test_get_leader_bad_style():
    parts = """
                M: [uniform_shape(6)]
    """
    partitioning = build_partitioning(parts)
    with pytest.raises(ValueError) as excinfo:
        partitioning.get_leader("M", "M1")

    assert str(excinfo.value) == "Style uniform_shape has no leader"


def test_get_offset():
    parts = """
                M: [uniform_shape(6), uniform_shape(3)]
                N: [uniform_shape(5)]
    """
    partitioning = build_partitioning(parts)

    assert partitioning.get_offset("M0") == "M1"
    assert partitioning.get_offset("M1") == "M2"
    assert partitioning.get_offset("M2") is None
    assert partitioning.get_offset("K") is None


def test_get_root_name():
    parts = """
                M: [uniform_shape(6)]
    """
    partitioning = build_partitioning(parts)

    assert partitioning.get_root_name("M1") == "M"
    assert partitioning.get_root_name("K") == "K"


def test_get_root_name_conv():
    parts = """
                Q: [uniform_shape(20), uniform_occupancy(I.5)]
    """
    partitioning = build_partitioning_conv(parts)

    assert partitioning.get_root_name("W0") == "W"
    assert partitioning.get_root_name("W1I") == "W"
    assert partitioning.get_root_name("Q1") == "Q"


def test_get_static_partitioning():
    all_parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = build_partitioning(all_parts)
    static = {("K",), ("N",)}

    assert partitioning.get_static_parts() == static


def test_get_static_partitioning_flattening():
    all_parts = """
                K: [uniform_shape(4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
    """
    partitioning = build_partitioning(all_parts)
    assert partitioning.get_static_parts() == {("K",), ("M", "K0")}


def test_get_static_partitioning_flattening_dyn():
    all_parts = """
                K: [uniform_occupancy(A.4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
    """
    partitioning = build_partitioning(all_parts)
    assert partitioning.get_static_parts() == set()


def test_get_step_dyn_part():
    parts = """
                M: [uniform_occupancy(Z.5)]
    """
    partitioning = build_partitioning(parts)

    with pytest.raises(ValueError) as excinfo:
        partitioning.get_step("M0")

    assert str(
        excinfo.value) == "No static step for dynamically partitioned rank M0"


def test_get_step():
    parts = """
                M: [uniform_shape(6), uniform_shape(3)]
                N: [uniform_shape(5)]
    """
    partitioning = build_partitioning(parts)

    assert partitioning.get_step("M0") is None
    assert partitioning.get_step("M1") == "M0"
    assert partitioning.get_step("M2") == "M1"
    assert partitioning.get_step("K") is None


def test_get_tensor_spec():
    all_parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = build_partitioning(all_parts)

    used_parts = """
                K: [uniform_shape(4)]
                M: [uniform_occupancy(A.6)]
    """
    used = build_part_dict(used_parts)

    tensor_ranks = ["J", "K", "M", "N"]

    assert partitioning.get_tensor_spec(tensor_ranks, {("K",), ("M",)}) == used


def test_get_tensor_spec_multiple_dyn():
    all_parts = """
                M: [uniform_occupancy(A.6), uniform_occupancy(A.3)]
    """
    partitioning = build_partitioning(all_parts)

    used_parts = """
                M1I: [uniform_occupancy(A.3)]
    """
    used = build_part_dict(used_parts)

    tensor_ranks = ["J", "K", "M1I", "N"]

    assert partitioning.get_tensor_spec(tensor_ranks, {("M1I",)}) == used


def test_get_tensor_spec_flattening():
    all_parts = """
                K: [uniform_shape(4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
    """
    partitioning = build_partitioning(all_parts)

    tensor_ranks = ["M", "K"]
    part_ranks = {("K",), ("M", "K0"), ("MK0",)}

    used_parts = """
                K: [uniform_shape(4)]
    """
    used = build_part_dict(used_parts)
    assert partitioning.get_tensor_spec(tensor_ranks, part_ranks) == used

    tensor_ranks = ["M", "K1", "K0"]
    used_parts = """
                (M, K0): [flatten()]
    """
    used = build_part_dict(used_parts)
    assert partitioning.get_tensor_spec(tensor_ranks, part_ranks) == used

    tensor_ranks = ["K1", "MK0"]

    used_parts = """
                MK0: [uniform_occupancy(A.5)]
    """
    used = build_part_dict(used_parts)
    assert partitioning.get_tensor_spec(tensor_ranks, part_ranks) == used


def test_get_tensor_spec_conv():
    all_parts = """
                P: [uniform_shape(4)]
                Q: [uniform_shape(5)]
    """
    partitioning = build_partitioning_conv(all_parts)

    used_parts = """
                Q: [uniform_shape(5)]
    """
    used = build_part_dict(used_parts)

    tensor_ranks = ["W"]

    assert partitioning.get_tensor_spec(tensor_ranks, {("P",), ("Q",)}) == used


def test_tensor_spec_bad_conv():
    all_parts = """
                P: [uniform_shape(4)]
                Q: [uniform_shape(5)]
    """
    partitioning = build_partitioning_conv(all_parts)
    tensor_ranks = ["W", "Q"]

    with pytest.raises(ValueError) as excinfo:
        partitioning.get_tensor_spec(tensor_ranks, {("Q",)})

    assert str(
        excinfo.value) == "Partitioning rank Q maps to tensor ranks ['W', 'Q']"


def test_get_valid_partitionings():
    all_parts = """
                K: [uniform_shape(4)]
                M: [uniform_shape(2), nway_shape(7)]
                N: [uniform_occupancy(A.6), uniform_occupancy(A.3)]
    """
    partitioning = build_partitioning(all_parts)

    assert set(partitioning.get_valid_parts(["K", "N"], False)) == {
        ("K",), ("N",), ("N1I",)}
    assert set(partitioning.get_valid_parts(["K", "N"], True)) == {
        ("K",), ("N",), ("N1I",)}


def test_get_valid_partitionings_flattening():
    all_parts = """
                K: [uniform_shape(4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
    """
    partitioning = build_partitioning(all_parts)

    assert set(partitioning.get_valid_parts(["K", "M"], False)) == {("K",)}
    assert set(partitioning.get_valid_parts(["K", "M"], True)) == {
        ("K",), ("M", "K0"), ("MK0",)}


def test_get_valid_partitionings_conv():
    all_parts = """
                P: [uniform_shape(4)]
                Q: [uniform_shape(5)]
    """
    partitioning = build_partitioning_conv(all_parts)

    assert set(partitioning.get_valid_parts(["W"], False)) == {("W",)}
    assert set(partitioning.get_valid_parts(["W"], True)) == {("W",)}


def test_is_flattened():
    all_parts = """
                K: [uniform_shape(4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.6), uniform_occupancy(A.2)]
    """
    partitioning = build_partitioning(all_parts)

    assert not partitioning.is_flattened("K")
    assert not partitioning.is_flattened("M")
    assert not partitioning.is_flattened("N")
    assert not partitioning.is_flattened("K0")
    assert partitioning.is_flattened("MK0")
    assert partitioning.is_flattened("MK02")
    assert partitioning.is_flattened("MK01I")


def test_partition_names_empty():
    all_parts = """
                M: [uniform_occupancy(A.6), uniform_occupancy(A.3)]
    """
    partitioning = build_partitioning(all_parts)

    with pytest.raises(ValueError) as excinfo:
        partitioning.partition_names((), True)

    assert str(
        excinfo.value) == "At least one rank required"


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
    partitioning = build_partitioning(all_parts)

    assert partitioning.partition_names(("J",), True) == ["J"]
    assert partitioning.partition_names(("J",), False) == ["J"]

    assert partitioning.partition_names(("M",), True) == ["M0", "M1", "M2"]
    assert partitioning.partition_names(("M",), False) == ["M1I", "M2"]

    assert partitioning.partition_names(("N",), False) == ["N0", "N1", "N2"]

    assert partitioning.partition_names(("K",), False) == ["K6I", "K7", "K8"]
    assert partitioning.partition_names(("K3I",), False) == [
        "K0", "K1", "K2", "K3"]


def test_partition_names_flattening():
    all_parts = """
                K: [uniform_shape(4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
    """
    partitioning = build_partitioning(all_parts)

    assert partitioning.partition_names(("M",), True) == ["M"]
    assert partitioning.partition_names(("K",), True) == ["K0", "K1"]
    assert partitioning.partition_names(("M", "K0"), True) == ["MK00", "MK01"]
    assert partitioning.partition_names(("M", "K0"), False) == ["MK0"]


def test_partition_names_conv():
    all_parts = """
                Q:
                    - uniform_shape(500)
                    - uniform_shape(250)
                    - uniform_occupancy(A.100)
                    - uniform_occupancy(A.50)
                    - uniform_shape(10)
                    - uniform_occupancy(A.6)
                    - uniform_shape(4)
                    - uniform_shape(2)
    """
    partitioning = build_partitioning_conv(all_parts)
    all_head = ["Q" + str(i + 1) for i in range(8)]

    assert partitioning.partition_names(("W",), True) == ["W0"] + all_head
    assert partitioning.partition_names(("W",), False) == ["W6I", "Q7", "Q8"]
    assert partitioning.partition_names(("W3I",), False) == [
        "W0"] + all_head[:3]


def test_partition_rank():
    all_parts = """
                P: [uniform_shape(500)]
                Q:
                    - uniform_shape(500)
                    - uniform_shape(250)
                    - uniform_occupancy(A.100)
                    - uniform_occupancy(A.50)
                    - uniform_shape(10)
                    - uniform_occupancy(A.6)
                    - uniform_shape(4)
                    - uniform_shape(2)
    """
    partitioning = build_partitioning_conv(all_parts)

    assert partitioning.partition_rank("S") is None
    assert partitioning.partition_rank("P") == "P"
    assert partitioning.partition_rank("W") == "Q"
    assert partitioning.partition_rank("W6I") == "Q6I"


def test_partition_rank_flattening():
    all_parts = """
                K: [uniform_shape(4)]
                (M, K0): [flatten()]
    """
    partitioning = build_partitioning(all_parts)
    assert partitioning.partition_rank("MK0") is None

    all_parts = """
                K: [uniform_shape(4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
    """
    partitioning = build_partitioning(all_parts)
    assert partitioning.partition_rank("MK0") == "MK0"


def test_partition_ranks_all():
    parts = """
                M: [uniform_shape(3)]
                K: [uniform_occupancy(A.4), uniform_occupancy(A.2)]
    """
    partitioning = build_partitioning(parts)

    ranks = ["M", "N", "K"]
    new_ranks = partitioning.partition_ranks(ranks, ranks, True, False)

    assert new_ranks == ["M1", "M0", "N", "K2", "K1", "K0"]


def test_partition_ranks_dyn():
    parts = """
                M: [uniform_occupancy(A.4), uniform_occupancy(A.2)]
                K: [uniform_shape(6), uniform_shape(3)]
    """
    ranks = ["M", "N", "K"]
    partitioning = build_partitioning(parts)

    new_ranks = partitioning.partition_ranks(ranks, ranks, False, False)
    assert new_ranks == ["M2", "M1I", "N", "K2", "K1", "K0"]

    new_ranks = partitioning.partition_ranks(new_ranks, {"M1I"}, False, False)
    assert new_ranks == ["M2", "M1", "M0", "N", "K2", "K1", "K0"]


def test_partition_ranks_flattening():
    all_parts = """
                K: [uniform_shape(4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
    """
    partitioning = build_partitioning(all_parts)
    ranks = ["M", "K", "N"]

    new_ranks = partitioning.partition_ranks(
        ranks, sorted(ranks), False, False)
    assert new_ranks == ["M", "K1", "K0", "N"]

    assert partitioning.partition_ranks(
        new_ranks, sorted(new_ranks), False, False) == new_ranks

    new_ranks = partitioning.partition_ranks(
        new_ranks, sorted(new_ranks), False, True)
    assert new_ranks == ["K1", "N", "MK0"]

    test_ranks = ["K1", "M", "K0", "N"]
    new_ranks = partitioning.partition_ranks(
        test_ranks, sorted(test_ranks), False, False)
    assert new_ranks == ["K1", "MK0", "N"]

    new_ranks = partitioning.partition_ranks(
        new_ranks, sorted(new_ranks), False, False)
    assert new_ranks == ["K1", "MK01", "MK00", "N"]


def test_partition_ranks_flattening_all():
    all_parts = """
                K: [uniform_shape(4)]
                (K0, M): [flatten()]
                K0M: [uniform_occupancy(A.5)]
    """
    partitioning = build_partitioning(all_parts)
    ranks = ["K", "M", "N"]

    new_ranks = partitioning.partition_ranks(
        ranks, sorted(ranks), False, False)
    assert new_ranks == ["K1", "K0", "M", "N"]

    new_ranks = partitioning.partition_ranks(ranks, sorted(ranks), True, False)
    assert new_ranks == ["K1", "K0M1", "K0M0", "N"]

    ranks = ["M", "K", "N"]
    new_ranks = partitioning.partition_ranks(ranks, sorted(ranks), True, True)
    assert new_ranks == ["K1", "N", "K0M1", "K0M0"]


def test_partition_ranks_conv():
    parts = """
                Q: [uniform_occupancy(A.4), uniform_occupancy(A.2)]
    """
    partitioning = build_partitioning_conv(parts)
    ranks = ["W"]

    new_ranks = partitioning.partition_ranks(ranks, ["W"], True, False)
    assert new_ranks == ["Q2", "Q1", "W0"]

    new_ranks = partitioning.partition_ranks(ranks, ["W"], False, False)
    assert new_ranks == ["Q2", "W1I"]

    new_ranks = partitioning.partition_ranks(new_ranks, ["W1I"], False, False)
    assert new_ranks == ["Q2", "Q1", "W0"]


def test_skip_empty_partitioning():
    all_parts = """
                K: []
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning = build_partitioning(all_parts)

    assert "K" not in partitioning.get_all_parts()


def test_eq():
    all_parts = """
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning1 = build_partitioning(all_parts)

    partitioning2 = build_partitioning(all_parts)

    assert partitioning1 == partitioning2


def test_neq_dyn_parts():
    parts1 = """
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning1 = build_partitioning(parts1)

    parts2 = """
                M: [uniform_occupancy(A.5)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning2 = build_partitioning(parts2)

    assert partitioning1 != partitioning2


def test_neq_static_parts():
    parts1 = """
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(3), nway_shape(7)]
    """
    partitioning1 = build_partitioning(parts1)

    parts2 = """
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning2 = build_partitioning(parts2)

    assert partitioning1 != partitioning2


def test_neq_obj():
    all_parts = """
                M: [uniform_occupancy(A.6)]
                N: [uniform_shape(2), nway_shape(7)]
    """
    partitioning1 = build_partitioning(all_parts)

    assert partitioning1 != "foo"
