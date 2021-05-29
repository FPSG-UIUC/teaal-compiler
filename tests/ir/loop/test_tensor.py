import pytest

from es2hfa.ir.loop.tensor import Tensor
from tests.utils.parse_tree import make_output, make_plus, make_tensor


def test_bad_tree():
    tree = make_plus(["a", "b"])
    with pytest.raises(ValueError) as excinfo:
        Tensor(tree)

    assert str(excinfo.value) == "Input parse tree must be a tensor"


def test_root_name():
    tree = make_tensor("A", ["i", "j"])
    assert Tensor(tree).root_name() == "A"


def test_fiber_name_ind():
    tree = make_tensor("A", ["i", "j"])
    assert Tensor(tree).fiber_name() == "a_i"


def test_fiber_name_ref():
    tree = make_output("A", [])
    assert Tensor(tree).fiber_name() == "a_ref"


def test_fiber_name_val():
    tree = make_tensor("A", [])
    assert Tensor(tree).fiber_name() == "a_val"


def test_peek_ind():
    tree = make_tensor("A", ["i", "j"])
    assert Tensor(tree).peek() == "i"


def test_peek_empty():
    tree = make_tensor("A", [])
    assert Tensor(tree).peek() is None


def test_pop():
    tensor = Tensor(make_tensor("A", ["i", "j"]))
    assert tensor.pop() == "i"
    assert tensor.pop() == "j"
    assert tensor.peek() is None


def test_swizzle():
    tensor = Tensor(make_tensor("A", ["i", "j"]))
    tensor.swizzle(["j", "k", "i"])
    assert tensor.pop() == "j"
    assert tensor.pop() == "i"
    assert tensor.peek() is None


def test_eq():
    tensor = make_tensor("A", ["i", "j"])
    assert Tensor(tensor) == Tensor(tensor)


def test_neq_name():
    tensor1 = make_tensor("A", ["i", "j"])
    tensor2 = make_tensor("B", ["i", "j"])
    assert Tensor(tensor1) != Tensor(tensor2)


def test_neq_inds():
    tensor1 = make_tensor("A", ["i", "j"])
    tensor2 = make_tensor("A", ["k", "j"])
    assert Tensor(tensor1) != Tensor(tensor2)


def test_neq_is_output():
    tensor1 = make_output("A", ["i", "j"])
    tensor2 = make_tensor("A", ["i", "j"])
    assert Tensor(tensor1) != Tensor(tensor2)


def test_neq_obj():
    tensor = Tensor(make_tensor("A", ["i", "j"]))
    obj = "foo"
    assert tensor != obj
