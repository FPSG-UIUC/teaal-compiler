import pytest

from es2hfa.ir.mapping import Mapping
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.einsum import EinsumParser
from es2hfa.parse.tensor import TensorParser
from tests.utils.parse_tree import make_uniform_shape


def test_missing_decl():
    with pytest.raises(ValueError) as excinfo:
        Mapping([TensorParser.parse("A[]")], {"A": [], "B": []})
    assert str(excinfo.value) == "Undeclared tensor: B"


def test_add_einsum_bad_tree():
    tensors = [TensorParser.parse("A[]"), TensorParser.parse("B[]")]
    mapping = Mapping(tensors, {})
    tree = TensorParser.parse("A[]")

    with pytest.raises(ValueError) as excinfo:
        mapping.add_einsum(tree, {}, {})
    assert str(excinfo.value) == "Input parse tree must be an einsum"


def test_add_einsum_missing_decl():
    tensors = [TensorParser.parse("A[]"), TensorParser.parse("B[]")]
    mapping = Mapping(tensors, {})
    tree = EinsumParser.parse("A[] = B[] + C[]")

    with pytest.raises(ValueError) as excinfo:
        mapping.add_einsum(tree, {}, {})
    assert str(excinfo.value) == "Undeclared tensor: C"


def test_apply_loop_order_unconfigured():
    tensors = [TensorParser.parse("A[I, J]")]
    mapping = Mapping(tensors, {})

    with pytest.raises(ValueError) as excinfo:
        mapping.apply_loop_order(Tensor.from_tree(tensors[0]))
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"


def test_apply_loop_order_default():
    tensors = ["A[I, J]", "B[I, K]", "C[K, J]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, {})

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[k, j])")
    mapping.add_einsum(tree, {}, {})
    mapping.apply_loop_order(mapping.get_tensors()[2])

    C = Tensor.from_tree(TensorParser.parse("C[J, K]"))
    assert mapping.get_tensors()[2] == C


def test_apply_loop_order_ordered():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, {})

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, {"A": ["K", "J", "I"]}, {})
    mapping.apply_loop_order(mapping.get_tensors()[2])

    C = Tensor.from_tree(TensorParser.parse("C[K, J]"))
    assert mapping.get_tensors()[2] == C


def test_apply_partitioning_unconfigured():
    tensors = [TensorParser.parse("A[I, J]")]
    mapping = Mapping(tensors, {})

    with pytest.raises(ValueError) as excinfo:
        mapping.apply_partitioning(Tensor.from_tree(tensors[0]))
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"


def test_apply_partitioning_default():
    tensors = ["A[I, J]", "B[I, K]", "C[K, J]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, {})

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[k, j])")
    mapping.add_einsum(tree, {}, {})
    mapping.apply_partitioning(mapping.get_tensors()[2])

    C = Tensor.from_tree(TensorParser.parse("C[K, J]"))
    assert mapping.get_tensors()[2] == C


def test_apply_partitiong_mapped():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, {})

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(
        tree, {}, {"A": {"I": make_uniform_shape([4]), "K": make_uniform_shape([6, 3])}})
    mapping.apply_partitioning(mapping.get_tensors()[2])

    C = Tensor.from_tree(TensorParser.parse("C[J, K2, K1, K0]"))
    assert mapping.get_tensors()[2] == C


def test_get_loop_order_unconfigured():
    tensors = [TensorParser.parse("A[]"), TensorParser.parse("B[]")]
    mapping = Mapping(tensors, {})

    with pytest.raises(ValueError) as excinfo:
        mapping.get_loop_order()
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"


def test_get_loop_order_default():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, {})

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, {}, {})

    assert mapping.get_loop_order() == ["I", "J", "K"]


def test_get_loop_order_ordered():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, {})

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, {"A": ["J", "K", "I"]}, {})

    assert mapping.get_loop_order() == ["J", "K", "I"]


def test_get_loop_order_default_partitioned():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, {})

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(
        tree, {}, {"A": {"I": make_uniform_shape([4]), "K": make_uniform_shape([6, 3])}})

    assert mapping.get_loop_order() == ["I1", "I0", "J", "K2", "K1", "K0"]


def test_get_output_unconfigured():
    tensors = [TensorParser.parse("A[]"), TensorParser.parse("B[]")]
    mapping = Mapping(tensors, {})

    with pytest.raises(ValueError) as excinfo:
        mapping.get_output()
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"


def test_get_output():
    tensors = ["B[I, K]", "A[I, J]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, {})

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, {}, {})

    result = Tensor.from_tree(tensors[1])
    result.set_is_output(True)

    assert mapping.get_output() == result


def test_get_partitioning_unconfigured():
    tensors = [TensorParser.parse("A[]"), TensorParser.parse("B[]")]
    mapping = Mapping(tensors, {})

    with pytest.raises(ValueError) as excinfo:
        mapping.get_partitioning(Tensor.from_tree(tensors[0]))
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"


def test_get_partitioning_default():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, {})

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, {}, {})

    assert mapping.get_partitioning(Tensor.from_tree(tensors[2])) == {}


def test_get_partitioning_mapped():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, {})

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(
        tree, {}, {"A": {"I": make_uniform_shape([4]), "K": make_uniform_shape([6, 3])}})

    assert mapping.get_partitioning(Tensor.from_tree(tensors[2])) == {
        "K": make_uniform_shape([6, 3])}


def test_get_tensors_unconfigured():
    tensors = [TensorParser.parse("A[]"), TensorParser.parse("B[]")]
    mapping = Mapping(tensors, {})

    with pytest.raises(ValueError) as excinfo:
        mapping.get_tensors()
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"


def test_get_tensors():
    tensors = ["A[]", "B[]", "C[]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, {})

    tree = EinsumParser.parse("A[] = C[]")
    mapping.add_einsum(tree, {}, {})

    A = Tensor.from_tree(TensorParser.parse("A[]"))
    A.set_is_output(True)
    C = Tensor.from_tree(TensorParser.parse("C[]"))

    assert mapping.get_tensors() == [A, C]


def test_get_tensors_ordered():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    orders = {"A": ["J", "I"], "C": ["K", "J"]}
    mapping = Mapping(tensors, orders)

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, {}, {})

    results = ["A[J, I]", "B[I, K]", "C[K, J]"]
    results = [Tensor.from_tree(TensorParser.parse(result))
               for result in results]
    results[0].set_is_output(True)

    assert mapping.get_tensors() == results


def test_reset():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    orders = {"A": ["J", "I"], "C": ["K", "J"]}
    mapping = Mapping(tensors, orders)

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, {"A": ["K", "I", "J"]}, {})

    for tensor in mapping.get_tensors():
        mapping.apply_loop_order(tensor)

    mapping.reset()

    with pytest.raises(ValueError) as excinfo:
        mapping.get_loop_order()
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"

    with pytest.raises(ValueError) as excinfo:
        mapping.get_partitioning(Tensor("A", orders["A"]))
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"
    with pytest.raises(ValueError) as excinfo:
        mapping.get_tensors()
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"

    mapping.add_einsum(tree, {}, {})

    results = ["A[J, I]", "B[I, K]", "C[K, J]"]
    results = [Tensor.from_tree(TensorParser.parse(result))
               for result in results]
    results[0].set_is_output(True)

    assert mapping.get_tensors() == results


def test_default_loop_order_unconfigured():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, {})

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")

    with pytest.raises(ValueError) as excinfo:
        mapping._Mapping__default_loop_order(tree)
    assert str(excinfo.value) == "Must configure partitioning before loop order"
