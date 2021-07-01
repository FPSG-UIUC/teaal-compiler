import pytest

from es2hfa.ir.tensor import Tensor
from tests.utils.parse_tree import make_plus, make_uniform_shape


def test_repeat_inds():
    with pytest.raises(ValueError) as excinfo:
        Tensor("A", ["I", "J", "I"])

    assert str(
        excinfo.value) == "All indices must be unique; given A: [I, J, I]"


def test_fiber_name_ind():
    tensor = Tensor("A", ["I", "J"])
    assert tensor.fiber_name() == "a_i"


def test_fiber_name_ref():
    tensor = Tensor("A", [])
    tensor.set_is_output(True)
    assert tensor.fiber_name() == "a_ref"


def test_fiber_name_val():
    tensor = Tensor("A", [])
    assert tensor.fiber_name() == "a_val"


def test_get_access():
    tensor = Tensor("A", ["I", "J"])
    assert tensor.get_access() == ["i", "j"]


def test_get_inds():
    tensor = Tensor("A", ["I", "J"])
    assert tensor.get_inds() == ["I", "J"]


def test_partition():
    tensor = Tensor("A", ["I", "J", "K"])
    partitioning = {"I": make_uniform_shape(
        [3]), "K": make_uniform_shape([4, 2])}
    tensor.partition(partitioning)
    assert tensor.get_inds() == ["I1", "I0", "J", "K2", "K1", "K0"]


def test_inds_safe_after_partition():
    inds = ["I", "J", "K"]
    tensor = Tensor("A", inds)
    partitioning = {"I": make_uniform_shape(
        [3]), "K": make_uniform_shape([4, 2])}
    tensor.partition(partitioning)
    assert tensor.get_inds() == ["I1", "I0", "J", "K2", "K1", "K0"]
    assert inds == ["I", "J", "K"]


def test_peek_ind():
    tensor = Tensor("A", ["I", "J"])
    assert tensor.peek() == "I"


def test_peek_empty():
    tensor = Tensor("A", [])
    assert tensor.peek() is None


def test_pop():
    tensor = Tensor("A", ["I", "J"])
    assert tensor.pop() == "i"
    assert tensor.pop() == "j"
    assert tensor.peek() is None


def test_reset():
    tensor = Tensor("A", ["I", "J", "K"])

    tensor.set_is_output(True)
    tensor.swizzle(["J", "K", "I"])
    tensor.pop()

    tensor.reset()

    assert tensor == Tensor("A", ["I", "J", "K"])
    assert tensor.fiber_name() == "a_i"

    tensor.partition({"J": make_uniform_shape([4])})
    tensor.reset()

    assert tensor == Tensor("A", ["I", "J", "K"])


def test_root_name():
    tensor = Tensor("A", ["I", "J"])
    assert tensor.root_name() == "A"


def test_set_is_output():
    tensor = Tensor("A", [])
    assert tensor.fiber_name() == "a_val"

    tensor.set_is_output(True)
    assert tensor.fiber_name() == "a_ref"


def test_swizzle():
    tensor = Tensor("A", ["I", "J"])
    tensor.swizzle(["J", "K", "I"])

    assert tensor.pop() == "j"
    assert tensor.pop() == "i"
    assert tensor.peek() is None


def test_tensor_name():
    tensor = Tensor("A", ["I", "J", "K"])
    assert tensor.tensor_name() == "A_IJK"


def test_eq():
    args = ("A", ["I", "J"])
    assert Tensor(*args) == Tensor(*args)


def test_neq_name():
    tensor1 = Tensor("A", ["I", "J"])
    tensor2 = Tensor("B", ["I", "J"])
    assert tensor1 != tensor2


def test_neq_inds():
    tensor1 = Tensor("A", ["I", "J"])
    tensor2 = Tensor("A", ["I", "K"])
    assert tensor1 != tensor2


def test_neq_is_output():
    tensor1 = Tensor("A", ["I", "J"])
    tensor1.set_is_output(True)
    tensor2 = Tensor("A", ["I", "J"])
    assert tensor1 != tensor2


def test_neq_obj():
    tensor = Tensor("A", ["I", "J"])
    obj = "foo"
    assert tensor != obj
