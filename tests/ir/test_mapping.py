import pytest

from es2hfa.ir.display import Display
from es2hfa.ir.mapping import Mapping
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.input import Input
from tests.utils.parse_tree import make_uniform_shape


def create_default():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
            C: [M, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    """
    return Mapping(Input.from_str(yaml))


def create_loop_ordered():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
        loop-order:
            Z: [K, N, M]
    """
    return Mapping(Input.from_str(yaml))


def create_partitioned():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
        partitioning:
            Z:
                K: [uniform_shape(6), uniform_shape(3)]
                M: [uniform_shape(5)]
    """
    return Mapping(Input.from_str(yaml))


def create_rank_ordered():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
        rank-order:
            Z: [N, M]
            A: [M, K]
    """
    return Mapping(Input.from_str(yaml))


def create_displayed(time, style):
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
        display:
            Z:
                space: [N]
                time: """ + time + """
                """ + style
    return Mapping(Input.from_str(yaml))


def test_missing_decl():
    yaml = """
    einsum:
        declaration:
            A: []
        expressions:
            - A[] = b
    mapping:
        rank-order:
            A: []
            B: []
    """
    input_ = Input.from_str(yaml)
    with pytest.raises(ValueError) as excinfo:
        Mapping(input_)
    assert str(excinfo.value) == "Undeclared tensor: B"


def test_add_einsum_missing_decl():
    yaml = """
    einsum:
        declaration:
            A: []
            B: []
        expressions:
            - A[] = B[] + C[]
    """
    mapping = Mapping(Input.from_str(yaml))

    with pytest.raises(ValueError) as excinfo:
        mapping.add_einsum(0)
    assert str(excinfo.value) == "Undeclared tensor: C"


def test_apply_loop_order_unconfigured():
    mapping = create_default()

    with pytest.raises(ValueError) as excinfo:
        mapping.apply_loop_order(Tensor("A", ["K", "M"]))
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"


def test_apply_loop_order_default():
    mapping = create_default()
    mapping.add_einsum(0)
    mapping.apply_loop_order(mapping.get_tensors()[2])

    assert mapping.get_tensors()[2] == Tensor("B", ["N", "K"])


def test_apply_loop_order_ordered():
    mapping = create_loop_ordered()
    mapping.add_einsum(0)
    mapping.apply_loop_order(mapping.get_tensors()[0])

    Z = Tensor("Z", ["N", "M"])
    Z.set_is_output(True)
    assert mapping.get_tensors()[0] == Z


def test_apply_partitioning_unconfigured():
    mapping = create_default()

    with pytest.raises(ValueError) as excinfo:
        mapping.apply_partitioning(Tensor("A", ["K", "M"]))
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"


def test_apply_partitioning_default():
    mapping = create_default()
    mapping.add_einsum(0)
    mapping.apply_partitioning(mapping.get_tensors()[2])

    assert mapping.get_tensors()[2] == Tensor("B", ["K", "N"])


def test_apply_partitiong_mapped():
    mapping = create_partitioned()
    mapping.add_einsum(0)
    mapping.apply_partitioning(mapping.get_tensors()[2])

    assert mapping.get_tensors()[2] == Tensor("B", ["K2", "K1", "K0", "N"])


def test_get_display_unconfigured():
    mapping = create_default()

    with pytest.raises(ValueError) as excinfo:
        mapping.get_display()
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"


def test_get_display_unspecified():
    mapping = create_default()
    mapping.add_einsum(0)
    assert mapping.get_display() is None


def test_get_display_specified():
    mapping = create_displayed("[K, M]", "style: shape")
    mapping.add_einsum(0)

    yaml = {"space": ["N"], "time": ["M", "K"], "style": "shape"}
    display = Display(yaml, mapping.get_loop_order(),
                      mapping.get_output().root_name())
    assert mapping.get_display() == display


def test_get_loop_order_unconfigured():
    mapping = create_default()

    with pytest.raises(ValueError) as excinfo:
        mapping.get_loop_order()
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"


def test_get_loop_order_default():
    mapping = create_default()
    mapping.add_einsum(0)

    assert mapping.get_loop_order() == ["M", "N", "K"]


def test_get_loop_order_ordered():
    mapping = create_loop_ordered()
    mapping.add_einsum(0)

    assert mapping.get_loop_order() == ["K", "N", "M"]


def test_get_loop_order_default_partitioned():
    mapping = create_partitioned()
    mapping.add_einsum(0)
    assert mapping.get_loop_order() == ["M1", "M0", "N", "K2", "K1", "K0"]


def test_get_output_unconfigured():
    mapping = create_default()

    with pytest.raises(ValueError) as excinfo:
        mapping.get_output()
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"


def test_get_output():
    mapping = create_default()
    mapping.add_einsum(0)

    result = Tensor("Z", ["M", "N"])
    result.set_is_output(True)

    assert mapping.get_output() == result


def test_get_partitioning_unconfigured():
    mapping = create_partitioned()

    with pytest.raises(ValueError) as excinfo:
        mapping.get_partitioning(Tensor("A", ["K", "M"]))
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"


def test_get_partitioning_default():
    mapping = create_default()
    mapping.add_einsum(0)

    assert mapping.get_partitioning(Tensor("A", ["K", "M"])) == {}


def test_get_partitioning_mapped():
    mapping = create_partitioned()
    mapping.add_einsum(0)

    assert mapping.get_partitioning(Tensor("B", ["K", "N"])) == {
        "K": make_uniform_shape([6, 3])}


def test_get_tensors_unconfigured():
    mapping = create_default()

    with pytest.raises(ValueError) as excinfo:
        mapping.get_tensors()
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"


def test_get_tensors():
    mapping = create_default()
    mapping.add_einsum(0)

    Z = Tensor("Z", ["M", "N"])
    Z.set_is_output(True)
    A = Tensor("A", ["K", "M"])
    B = Tensor("B", ["K", "N"])

    assert mapping.get_tensors() == [Z, A, B]


def test_get_tensors_ordered():
    mapping = create_rank_ordered()
    mapping.add_einsum(0)

    Z = Tensor("Z", ["N", "M"])
    Z.set_is_output(True)
    A = Tensor("A", ["M", "K"])
    B = Tensor("B", ["K", "N"])

    assert mapping.get_tensors() == [Z, A, B]


def test_reset():
    mapping = create_rank_ordered()
    mapping.add_einsum(0)

    for tensor in mapping.get_tensors():
        mapping.apply_loop_order(tensor)

    mapping.reset()

    with pytest.raises(ValueError) as excinfo:
        mapping.get_display()
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"

    with pytest.raises(ValueError) as excinfo:
        mapping.get_loop_order()
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"

    with pytest.raises(ValueError) as excinfo:
        mapping.get_partitioning(Tensor("A", ["M", "K"]))
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"

    with pytest.raises(ValueError) as excinfo:
        mapping.get_tensors()
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"

    mapping.add_einsum(0)

    Z = Tensor("Z", ["N", "M"])
    Z.set_is_output(True)
    A = Tensor("A", ["M", "K"])
    B = Tensor("B", ["K", "N"])

    assert mapping.get_tensors() == [Z, A, B]


def test_default_loop_order_unconfigured():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
            C: [M, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    """
    input_ = Input.from_str(yaml)
    mapping = Mapping(input_)
    with pytest.raises(ValueError) as excinfo:
        mapping._Mapping__default_loop_order(input_.get_expressions()[0])
    assert str(excinfo.value) == "Must configure partitioning before loop order"
