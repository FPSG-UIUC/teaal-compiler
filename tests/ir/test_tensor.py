import pytest

from teaal.ir.tensor import Tensor


def test_repeat_ranks():
    with pytest.raises(ValueError) as excinfo:
        Tensor("A", ["I", "J", "I"])

    assert str(
        excinfo.value) == "All ranks must be unique; given A: [I, J, I]"


def test_from_fiber():
    tensor = Tensor("A", ["I", "J", "K"])
    tensor.pop()
    tensor.from_fiber()

    assert tensor.get_access() == ["j", "k"]
    assert tensor.get_ranks() == ["J", "K"]
    assert tensor.tensor_name() == "A_JK"


def test_fiber_name_rank():
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


def test_get_init_ranks():
    tensor = Tensor("A", ["I", "J", "K"])

    tensor.update_ranks(["I1", "I0", "J", "K2", "K1", "K0"])
    tensor.swizzle(["K2", "I1", "J", "K1", "I0", "K0"])

    assert tensor.get_init_ranks() == ["I", "J", "K"]


def test_get_is_output():
    tensor = Tensor("A", ["I", "J"])
    assert not tensor.get_is_output()

    tensor.set_is_output(True)
    assert tensor.get_is_output()


def test_get_prefix():
    tensor = Tensor("A", ["I", "J"])

    assert tensor.get_prefix("root") == []
    assert tensor.get_prefix("I") == ["I"]
    assert tensor.get_prefix("J") == ["I", "J"]


def test_get_ranks():
    tensor = Tensor("A", ["I", "J"])
    assert tensor.get_ranks() == ["I", "J"]


def test_ranks_safe_after_partition():
    ranks = ["I", "J", "K"]
    tensor = Tensor("A", ranks)
    tensor.update_ranks(["I1", "I0", "J", "K2", "K1", "K0"])

    assert tensor.get_ranks() == ["I1", "I0", "J", "K2", "K1", "K0"]
    assert ranks == ["I", "J", "K"]


def test_peek_rank():
    tensor = Tensor("A", ["I", "J"])
    assert tensor.peek() == "i"


def test_peek_empty():
    tensor = Tensor("A", [])
    assert tensor.peek() is None


def test_peek_rest():
    tensor = Tensor("A", ["K1", "M", "K0"])
    assert tensor.peek_rest() == ["K1", "M", "K0"]

    tensor.pop()
    assert tensor.peek_rest() == ["M", "K0"]


def test_pop():
    tensor = Tensor("A", ["I", "J"])
    assert tensor.pop() == "i"
    assert tensor.pop() == "j"
    assert tensor.peek() is None


def test_reset():
    ranks = ["I", "J", "K"]
    tensor = Tensor("A", ranks)

    tensor.set_is_output(True)
    tensor.swizzle(["J", "K", "I"])
    tensor.update_ranks(["J", "K2", "K1", "K0", "I1", "I0"])
    tensor.pop()

    tensor.reset()

    assert tensor == Tensor("A", ranks)
    assert tensor.fiber_name() == "a_i"

    tensor.reset()

    assert tensor == Tensor("A", ranks)


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


def test_update_ranks():
    tensor = Tensor("A", ["I", "J"])
    tensor.pop()
    tensor.update_ranks(["J2", "J1", "J0"])

    assert tensor.get_init_ranks() == ["I", "J"]
    assert tensor.get_ranks() == ["J2", "J1", "J0"]


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


def test_neq_ranks():
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


def test_repr():
    tensor = Tensor("A", ["I", "J", "K"])
    tensor.set_is_output(True)
    assert repr(tensor) == "(Tensor, A, ['I', 'J', 'K'], True, 0, 0)"
