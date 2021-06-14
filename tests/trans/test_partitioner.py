from es2hfa.ir.mapping import Mapping
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.einsum import EinsumParser
from es2hfa.parse.tensor import TensorParser
from es2hfa.trans.partitioning import Partitioner
from tests.utils.parse_tree import make_uniform_shape


def test_no_partitioning():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, {}, {})

    partitioner = Partitioner(mapping)
    assert partitioner.partition(Tensor(tensors[1])).gen(depth=0) == ""


def test_uniform_shape():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(
        tree, {}, {"A": {"I": make_uniform_shape([5]), "K": make_uniform_shape([6, 3])}})

    hfa = "tmp = C_JK\n" + \
          "tmp = tmp.splitUniform(6, depth=1)\n" + \
          "tmp = tmp.splitUniform(3, depth=2)\n" + \
          "C_JK2K1K0 = tmp"

    partitioner = Partitioner(mapping)
    assert partitioner.partition(Tensor(tensors[2])).gen(depth=0) == hfa
