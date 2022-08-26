from teaal.parse.mapping import Mapping
from teaal.parse.spacetime import SpaceTimeParser
from tests.utils.parse_tree import make_uniform_shape


def test_empty():
    mapping = Mapping.from_str("")

    assert mapping.get_loop_orders() == {}
    assert mapping.get_partitioning() == {}
    assert mapping.get_rank_orders() == {}
    assert mapping.get_rank_orders() == {}
    assert mapping.get_spacetime() == {}


def test_eq():
    mapping = Mapping.from_file("tests/integration/test_input.yaml")
    assert mapping != "foo"


def test_from():
    yaml = """
    mapping:
        rank-order:
            A: [M, K]
            C: [N, M]
            Z: [N, M]
        loop-order:
            T1: [K, N, M]
            Z: [M2, N2, M1, N1, M0, N0]
        partitioning:
            Z:
                M: [uniform_shape(4), uniform_shape(2)]
                N: [uniform_shape(6), uniform_shape(3)]
        spacetime:
            T1:
                space: [N]
                time: [K.pos, M.coord]
    """
    from_file = Mapping.from_file("tests/integration/test_input.yaml")
    from_str = Mapping.from_str(yaml)
    assert from_file == from_str


def test_just_mapping():
    mapping = Mapping.from_str("mapping:")

    assert mapping.get_loop_orders() == {}
    assert mapping.get_partitioning() == {}
    assert mapping.get_rank_orders() == {}
    assert mapping.get_rank_orders() == {}
    assert mapping.get_spacetime() == {}


def test_loop_orders():
    mapping = Mapping.from_file("tests/integration/test_input.yaml")
    assert mapping.get_loop_orders() == {
        "T1": [
            "K", "N", "M"], "Z": [
            "M2", "N2", "M1", "N1", "M0", "N0"]}


def test_loop_orders_missing():
    mapping = Mapping.from_file(
        "tests/integration/test_input_no_loop_order.yaml")
    assert mapping.get_loop_orders() == {}


def test_no_mapping():
    mapping = Mapping.from_file("tests/integration/test_input_no_mapping.yaml")

    assert mapping.get_spacetime() == {}
    assert mapping.get_loop_orders() == {}
    assert mapping.get_partitioning() == {}
    assert mapping.get_rank_orders() == {}


def test_partitioning():
    mapping = Mapping.from_file("tests/integration/test_input.yaml")
    partitioning = {"Z": {"M": make_uniform_shape(
        [4, 2]), "N": make_uniform_shape([6, 3])}}

    assert mapping.get_partitioning() == partitioning


def test_partitioning_missing():
    mapping = Mapping.from_file(
        "tests/integration/test_input_no_partitioning.yaml")
    assert mapping.get_partitioning() == {}


def test_rank_orders():
    mapping = Mapping.from_file("tests/integration/test_input.yaml")

    tensors = ["A[M, K]", "C[N, M]", "Z[N, M]"]
    tensors = {"A": ["M", "K"], "C": ["N", "M"], "Z": ["N", "M"]}

    assert mapping.get_rank_orders() == tensors


def test_rank_orders_missing():
    mapping = Mapping.from_file(
        "tests/integration/test_input_no_rank_order.yaml")
    tensors = {}
    assert mapping.get_rank_orders() == tensors


def test_spacetime():
    yaml = """
    mapping:
        spacetime:
            T1:
                space: [M0]
                time: [M1.pos, M2.coord]
                opt: slip
    """
    mapping = Mapping.from_str(yaml)
    spacetime = {
        "T1": {
            "space": [
                SpaceTimeParser.parse("M0")],
            "time": [
                SpaceTimeParser.parse("M1.pos"),
                SpaceTimeParser.parse("M2.coord")],
            "opt": "slip"}}

    assert mapping.get_spacetime() == spacetime


def test_spacetime_missing():
    mapping = Mapping.from_file(
        "tests/integration/test_input_no_spacetime.yaml")
    assert mapping.get_spacetime() == {}
