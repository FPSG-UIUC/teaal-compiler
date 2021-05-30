import pytest

from es2hfa.ir.mapping import Mapping
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.einsum import EinsumParser
from tests.utils.parse_tree import make_tensor


def test_missing_decl():
    tensors = [make_tensor("A", []), make_tensor("B", [])]
    with pytest.raises(ValueError) as excinfo:
        Mapping(tensors[:1], tensors)
    assert str(excinfo.value) == "Undeclared tensor: B"


def test_add_einsum_bad_tree():
    tensors = [make_tensor("A", []), make_tensor("B", [])]
    mapping = Mapping(tensors, [])
    tree = make_tensor("A", [])

    with pytest.raises(ValueError) as excinfo:
        mapping.add_einsum(tree, None)
    assert str(excinfo.value) == "Input parse tree must be an einsum"


def test_add_einsum_missing_decl():
    tensors = [make_tensor("A", []), make_tensor("B", [])]
    mapping = Mapping(tensors, [])
    tree = EinsumParser.parse("A[] = B[] + C[]")

    with pytest.raises(ValueError) as excinfo:
        mapping.add_einsum(tree, None)
    assert str(excinfo.value) == "Undeclared tensor: C"


def test_apply_loop_order_unconfigured():
    tensors = [make_tensor("A", ["I", "J"])]
    mapping = Mapping(tensors, [])

    with pytest.raises(ValueError) as excinfo:
        mapping.apply_loop_order("A")
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"


def test_apply_loop_order_default():
    tensors = [
        make_tensor(
            "A", [
                "I", "J"]), make_tensor(
            "B", [
                "I", "K"]), make_tensor(
            "C", [
                "K", "J"])]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[k, j])")
    mapping.add_einsum(tree, [])
    mapping.apply_loop_order(mapping.get_tensors()[2])

    C = Tensor(make_tensor("C", ["J", "K"]))
    assert mapping.get_tensors()[2] == C


def test_apply_loop_order_ordered():
    tensors = [
        make_tensor(
            "A", [
                "I", "J"]), make_tensor(
            "B", [
                "I", "K"]), make_tensor(
            "C", [
                "J", "K"])]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, ["K", "J", "I"])
    mapping.apply_loop_order(mapping.get_tensors()[2])

    C = Tensor(make_tensor("C", ["K", "J"]))
    assert mapping.get_tensors()[2] == C


def test_get_loop_order_unconfigured():
    tensors = [make_tensor("A", []), make_tensor("B", [])]
    mapping = Mapping(tensors, [])

    with pytest.raises(ValueError) as excinfo:
        mapping.get_loop_order()
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"


def test_get_loop_order_default():
    tensors = [
        make_tensor(
            "A", [
                "I", "J"]), make_tensor(
            "B", [
                "I", "K"]), make_tensor(
            "C", [
                "J", "K"])]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, [])

    assert mapping.get_loop_order() == ["I", "J", "K"]


def test_get_loop_order_ordered():
    tensors = [
        make_tensor(
            "A", [
                "I", "J"]), make_tensor(
            "B", [
                "I", "K"]), make_tensor(
            "C", [
                "J", "K"])]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, ["J", "K", "I"])

    assert mapping.get_loop_order() == ["J", "K", "I"]


def test_get_tensors_unconfigured():
    tensors = [make_tensor("A", []), make_tensor("B", [])]
    mapping = Mapping(tensors, [])

    with pytest.raises(ValueError) as excinfo:
        mapping.get_tensors()
    assert str(
        excinfo.value) == "Unconfigured mapping. Make sure to first call add_einsum()"


def test_get_tensors():
    tensor_trees = [
        make_tensor(
            "A", []), make_tensor(
            "B", []), make_tensor(
                "C", [])]
    mapping = Mapping(tensor_trees, [])

    tree = EinsumParser.parse("A[] = C[]")
    mapping.add_einsum(tree, [])

    A = Tensor(make_tensor("A", []))
    A.set_is_output(True)
    C = Tensor(make_tensor("C", []))

    assert mapping.get_tensors() == [A, C]


def test_get_tensors_ordered():
    tensors = [
        make_tensor(
            "A", [
                "I", "J"]), make_tensor(
            "B", [
                "I", "K"]), make_tensor(
            "C", [
                "J", "K"])]
    orders = [make_tensor("A", ["J", "I"]), make_tensor("C", ["K", "J"])]
    mapping = Mapping(tensors, orders)

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, [])

    A = Tensor(make_tensor("A", ["J", "I"]))
    A.set_is_output(True)
    B = Tensor(make_tensor("B", ["I", "K"]))
    C = Tensor(make_tensor("C", ["K", "J"]))

    assert mapping.get_tensors() == [A, B, C]


def test_reset():
    tensors = [
        make_tensor(
            "A", [
                "I", "J"]), make_tensor(
            "B", [
                "I", "K"]), make_tensor(
            "C", [
                "J", "K"])]
    orders = [make_tensor("A", ["J", "I"]), make_tensor("C", ["K", "J"])]
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

    A = Tensor(make_tensor("A", ["J", "I"]))
    A.set_is_output(True)
    B = Tensor(make_tensor("B", ["I", "K"]))
    C = Tensor(make_tensor("C", ["K", "J"]))

    assert mapping.get_tensors() == [A, B, C]
