import pytest

from es2hfa.ir.mapping import Mapping
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.einsum import EinsumParser
from es2hfa.parse.tensor import TensorParser


def test_missing_decl():
    tensors = [TensorParser.parse("A[]"), TensorParser.parse("B[]")]
    with pytest.raises(ValueError) as excinfo:
        Mapping(tensors[:1], tensors)
    assert str(excinfo.value) == "Undeclared tensor: B"


def test_add_einsum_bad_tree():
    tensors = [TensorParser.parse("A[]"), TensorParser.parse("B[]")]
    mapping = Mapping(tensors, [])
    tree = TensorParser.parse("A[]")

    with pytest.raises(ValueError) as excinfo:
        mapping.add_einsum(tree, None)
    assert str(excinfo.value) == "Input parse tree must be an einsum"


def test_add_einsum_missing_decl():
    tensors = [TensorParser.parse("A[]"), TensorParser.parse("B[]")]
    mapping = Mapping(tensors, [])
    tree = EinsumParser.parse("A[] = B[] + C[]")

    with pytest.raises(ValueError) as excinfo:
        mapping.add_einsum(tree, None)
    assert str(excinfo.value) == "Undeclared tensor: C"


def test_apply_loop_order_unconfigured():
    tensors = [TensorParser.parse("A[I, J]")]
    mapping = Mapping(tensors, [])

    with pytest.raises(ValueError) as excinfo:
        mapping.apply_loop_order("A")
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"


def test_apply_loop_order_default():
    tensors = ["A[I, J]", "B[I, K]", "C[K, J]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[k, j])")
    mapping.add_einsum(tree, [])
    mapping.apply_loop_order(mapping.get_tensors()[2])

    C = Tensor(TensorParser.parse("C[J, K]"))
    assert mapping.get_tensors()[2] == C


def test_apply_loop_order_ordered():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, ["K", "J", "I"])
    mapping.apply_loop_order(mapping.get_tensors()[2])

    C = Tensor(TensorParser.parse("C[K, J]"))
    assert mapping.get_tensors()[2] == C


def test_get_loop_order_unconfigured():
    tensors = [TensorParser.parse("A[]"), TensorParser.parse("B[]")]
    mapping = Mapping(tensors, [])

    with pytest.raises(ValueError) as excinfo:
        mapping.get_loop_order()
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"


def test_get_loop_order_default():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, [])

    assert mapping.get_loop_order() == ["I", "J", "K"]


def test_get_loop_order_ordered():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, ["J", "K", "I"])

    assert mapping.get_loop_order() == ["J", "K", "I"]


def test_get_tensors_unconfigured():
    tensors = [TensorParser.parse("A[]"), TensorParser.parse("B[]")]
    mapping = Mapping(tensors, [])

    with pytest.raises(ValueError) as excinfo:
        mapping.get_tensors()
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"


def test_get_tensors():
    tensors = ["A[]", "B[]", "C[]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[] = C[]")
    mapping.add_einsum(tree, [])

    A = Tensor(TensorParser.parse("A[]"))
    A.set_is_output(True)
    C = Tensor(TensorParser.parse("C[]"))

    assert mapping.get_tensors() == [A, C]


def test_get_tensors_ordered():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    orders = ["A[J, I]", "C[K, J]"]
    orders = [TensorParser.parse(order) for order in orders]
    mapping = Mapping(tensors, orders)

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, [])

    results = ["A[J, I]", "B[I, K]", "C[K, J]"]
    results = [Tensor(TensorParser.parse(result)) for result in results]
    results[0].set_is_output(True)

    assert mapping.get_tensors() == results


def test_reset():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    orders = ["A[J, I]", "C[K, J]"]
    orders = [TensorParser.parse(order) for order in orders]
    mapping = Mapping(tensors, orders)

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, ["K", "I", "J"])

    for tensor in mapping.get_tensors():
        mapping.apply_loop_order(tensor)

    mapping.reset()

    with pytest.raises(ValueError) as excinfo:
        mapping.get_loop_order()
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"

    with pytest.raises(ValueError) as excinfo:
        mapping.get_tensors()
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"

    mapping.add_einsum(tree, [])

    results = ["A[J, I]", "B[I, K]", "C[K, J]"]
    results = [Tensor(TensorParser.parse(result)) for result in results]
    results[0].set_is_output(True)

    assert mapping.get_tensors() == results
