from es2hfa.parse.mapping import Mapping
from es2hfa.parse.display import DisplayParser
from tests.utils.parse_tree import make_uniform_shape


def test_display():
    mapping = Mapping.from_file("tests/integration/test_input.yaml")
    display = {
        "T1": {
            "space": [
                DisplayParser.parse("N")], "time": [
                DisplayParser.parse("K.pos"), DisplayParser.parse("M.coord")]}}

    assert mapping.get_display() == display


def test_display_missing():
    mapping = Mapping.from_file("tests/integration/test_input_no_display.yaml")
    assert mapping.get_display() == {}


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
        display:
            T1:
                space: [N]
                time: [K.pos, M.coord]
    """
    from_file = Mapping.from_file("tests/integration/test_input.yaml")
    from_str = Mapping.from_str(yaml)
    assert from_file == from_str


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

    assert mapping.get_display() == {}
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
