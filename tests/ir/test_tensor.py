import pytest

from es2hfa.ir.tensor import Tensor
from es2hfa.parse.tensor import TensorParser
from tests.utils.parse_tree import make_plus, make_uniform_shape


def test_bad_tree():
    tree = make_plus(["a", "b"])
    with pytest.raises(ValueError) as excinfo:
        Tensor.from_tree(tree)

    assert str(excinfo.value) == "Input parse tree must be a tensor"


def test_from_tree():
    tree = TensorParser.parse("A[I, J]")
    assert Tensor.from_tree(tree) == Tensor("A", ["I", "J"])


def test_fiber_name_ind():
    tree = TensorParser.parse("A[I, J]")
    assert Tensor.from_tree(tree).fiber_name() == "a_i"


def test_fiber_name_ref():
    tensor = Tensor.from_tree(TensorParser.parse("A[]"))
    tensor.set_is_output(True)
    assert tensor.fiber_name() == "a_ref"


def test_fiber_name_val():
    tree = TensorParser.parse("A[]")
    assert Tensor.from_tree(tree).fiber_name() == "a_val"


def test_get_inds():
    tree = TensorParser.parse("A[I, J]")
    assert Tensor.from_tree(tree).get_inds() == ["I", "J"]


def test_partition():
    tree = TensorParser.parse("A[I, J, K]")
    partitioning = {"I": make_uniform_shape(
        [3]), "K": make_uniform_shape([4, 2])}
    tensor = Tensor.from_tree(tree)
    tensor.partition(partitioning)
    assert tensor.get_inds() == ["I1", "I0", "J", "K2", "K1", "K0"]


def test_peek_ind():
    tree = TensorParser.parse("A[I, J]")
    assert Tensor.from_tree(tree).peek() == "I"


def test_peek_empty():
    tree = TensorParser.parse("A[]")
    assert Tensor.from_tree(tree).peek() is None


def test_pop():
    tensor = Tensor.from_tree(TensorParser.parse("A[I, J]"))
    assert tensor.pop() == "i"
    assert tensor.pop() == "j"
    assert tensor.peek() is None


def test_reset():
    tensor = Tensor.from_tree(TensorParser.parse("A[I, J, K]"))

    tensor.set_is_output(True)
    tensor.swizzle(["J", "K", "I"])
    tensor.pop()

    tensor.reset()

    assert tensor == Tensor.from_tree(TensorParser.parse("A[I, J, K]"))
    assert tensor.fiber_name() == "a_i"

    tensor.partition({"J": make_uniform_shape([4])})
    tensor.reset()

    assert tensor == Tensor.from_tree(TensorParser.parse("A[I, J, K]"))


def test_root_name():
    tree = TensorParser.parse("A[I, J]")
    assert Tensor.from_tree(tree).root_name() == "A"


def test_set_is_output():
    tensor = Tensor.from_tree(TensorParser.parse("A[]"))
    assert tensor.fiber_name() == "a_val"

    tensor.set_is_output(True)
    assert tensor.fiber_name() == "a_ref"


def test_swizzle():
    tensor = Tensor.from_tree(TensorParser.parse("A[I, J]"))
    tensor.swizzle(["J", "K", "I"])

    assert tensor.pop() == "j"
    assert tensor.pop() == "i"
    assert tensor.peek() is None


def test_tensor():
    tensor = Tensor.from_tree(TensorParser.parse("A[I, J, K]"))
    assert tensor.tensor_name() == "A_IJK"


def test_eq():
    tree = TensorParser.parse("A[I, J]")
    assert Tensor.from_tree(tree) == Tensor.from_tree(tree)


def test_neq_name():
    tensor1 = TensorParser.parse("A[I, J]")
    tensor2 = TensorParser.parse("B[I, J]")
    assert Tensor.from_tree(tensor1) != Tensor.from_tree(tensor2)


def test_neq_inds():
    tensor1 = TensorParser.parse("A[I, J]")
    tensor2 = TensorParser.parse("B[I, K]")
    assert Tensor.from_tree(tensor1) != Tensor.from_tree(tensor2)


def test_neq_is_output():
    tensor1 = Tensor.from_tree(TensorParser.parse("A[I, J]"))
    tensor1.set_is_output(True)
    tensor2 = Tensor.from_tree(TensorParser.parse("A[I, J]"))
    assert tensor1 != tensor2


def test_neq_obj():
    tensor = Tensor.from_tree(TensorParser.parse("A[I, J]"))
    obj = "foo"
    assert tensor != obj
