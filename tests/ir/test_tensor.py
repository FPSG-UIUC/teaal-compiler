import pytest

from es2hfa.ir.tensor import Tensor
from tests.utils.parse_tree import make_output, make_plus, make_tensor


def test_bad_tree():
    tree = make_plus(["a", "b"])
    with pytest.raises(ValueError) as excinfo:
        Tensor(tree)

    assert str(excinfo.value) == "Input parse tree must be a tensor"


def test_fiber_name_ind():
    tree = make_tensor("A", ["I", "J"])
    assert Tensor(tree).fiber_name() == "a_i"


def test_fiber_name_ref():
    tree = make_output("A", [])
    assert Tensor(tree).fiber_name() == "a_ref"


def test_fiber_name_val():
    tree = make_tensor("A", [])
    assert Tensor(tree).fiber_name() == "a_val"


def test_get_inds():
    tree = make_tensor("A", ["I", "J"])
    assert Tensor(tree).get_inds() == ["I", "J"]


def test_peek_ind():
    tree = make_tensor("A", ["I", "J"])
    assert Tensor(tree).peek() == "i"


def test_peek_empty():
    tree = make_tensor("A", [])
    assert Tensor(tree).peek() is None


def test_pop():
    tensor = Tensor(make_tensor("A", ["I", "J"]))
    assert tensor.pop() == "i"
    assert tensor.pop() == "j"
    assert tensor.peek() is None


def test_reset():
    tensor = Tensor(make_tensor("A", ["I", "J", "K"]))

    tensor.set_is_output(True)
    tensor.swizzle(["J", "K", "I"])
    tensor.pop()
    tensor.reset()

    assert tensor == Tensor(make_tensor("A", ["I", "J", "K"]))
    assert tensor.fiber_name() == "a_i"


def test_root_name():
    tree = make_tensor("A", ["I", "J"])
    assert Tensor(tree).root_name() == "A"


def test_set_is_output():
    tensor = Tensor(make_tensor("A", []))
    assert tensor.fiber_name() == "a_val"

    tensor.set_is_output(True)
    assert tensor.fiber_name() == "a_ref"


def test_swizzle():
    tensor = Tensor(make_tensor("A", ["I", "J"]))
    tensor.swizzle(["J", "K", "I"])

    assert tensor.pop() == "j"
    assert tensor.pop() == "i"
    assert tensor.peek() is None


def test_eq():
    tensor = make_tensor("A", ["I", "J"])
    assert Tensor(tensor) == Tensor(tensor)


def test_neq_name():
    tensor1 = make_tensor("A", ["I", "J"])
    tensor2 = make_tensor("B", ["I", "J"])
    assert Tensor(tensor1) != Tensor(tensor2)


def test_neq_inds():
    tensor1 = make_tensor("A", ["I", "J"])
    tensor2 = make_tensor("A", ["K", "J"])
    assert Tensor(tensor1) != Tensor(tensor2)


def test_neq_is_output():
    tensor1 = make_output("A", ["I", "J"])
    tensor2 = make_tensor("A", ["I", "J"])
    assert Tensor(tensor1) != Tensor(tensor2)


def test_neq_obj():
    tensor = Tensor(make_tensor("A", ["I", "J"]))
    obj = "foo"
    assert tensor != obj
