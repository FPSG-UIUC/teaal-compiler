import pytest

from es2hfa.ir.partitioning import Partitioning
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.mapping import Mapping
from tests.utils.parse_tree import make_plus, make_uniform_shape


def build_partitioning(parts, ranks):
    yaml = """
    mapping:
        partitioning:
            Z:
    """ + parts
    dict_ = Mapping.from_str(yaml).get_partitioning()
    return Partitioning(dict_["Z"], ranks)


def test_repeat_ranks():
    with pytest.raises(ValueError) as excinfo:
        Tensor("A", ["I", "J", "I"])

    assert str(
        excinfo.value) == "All ranks must be unique; given A: [I, J, I]"


def test_from_tensor_init():
    parent = Tensor("A", ["I", "J", "K"])
    child = Tensor.from_tensor(parent)
    assert parent == child


def test_from_tensor_is_output():
    parent = Tensor("A", ["I", "J", "K"])
    parent.set_is_output(True)
    child = Tensor.from_tensor(parent)
    assert parent == child


def test_from_tensor_intermediate():
    parent = Tensor("A", ["I", "J", "K"])
    parent.pop()

    child = Tensor.from_tensor(parent)
    corr = Tensor("A", ["J", "K"])

    assert child == corr


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


def test_get_ranks():
    tensor = Tensor("A", ["I", "J"])
    assert tensor.get_ranks() == ["I", "J"]


def test_partition():
    parts = """
                I: [uniform_shape(3)]
                K: [uniform_shape(4), uniform_shape(2)]
    """
    ranks = ["I", "J", "K"]
    partitioning = build_partitioning(parts, ranks)

    tensor = Tensor("A", ranks)
    tensor.partition(partitioning, ranks)
    assert tensor.get_ranks() == ["I1", "I0", "J", "K2", "K1", "K0"]


def test_ranks_safe_after_partition():
    parts = """
                I: [uniform_shape(3)]
                K: [uniform_shape(4), uniform_shape(2)]
    """
    ranks = ["I", "J", "K"]
    partitioning = build_partitioning(parts, ranks)

    tensor = Tensor("A", ranks)
    tensor.partition(partitioning, ranks)

    assert tensor.get_ranks() == ["I1", "I0", "J", "K2", "K1", "K0"]
    assert ranks == ["I", "J", "K"]


def test_peek_rank():
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
    ranks = ["I", "J", "K"]
    tensor = Tensor("A", ranks)

    tensor.set_is_output(True)
    tensor.swizzle(["J", "K", "I"])
    tensor.pop()

    tensor.reset()

    assert tensor == Tensor("A", ranks)
    assert tensor.fiber_name() == "a_i"

    parts = """
                I: [uniform_shape(3)]
                K: [uniform_shape(4), uniform_shape(2)]
    """
    partitioning = build_partitioning(parts, ranks)
    tensor.partition(partitioning, ranks)
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
