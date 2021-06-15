from es2hfa.ir.mapping import Mapping
from es2hfa.parse.einsum import EinsumParser
from es2hfa.parse.tensor import TensorParser
from es2hfa.trans.header import Header
from tests.utils.parse_tree import make_uniform_shape


def test_make_header():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, {}, {})

    hfa = "A_IJ = Tensor(rank_ids=[\"I\", \"J\"])\n" + \
          "a_i = A_IJ.getRoot()\n" + \
          "b_i = B_IK.getRoot()\n" + \
          "c_j = C_JK.getRoot()"

    assert Header.make_header(mapping).gen(depth=0) == hfa


def test_make_header_swizzle():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, {"A": ["K", "J", "I"]}, {})

    hfa = "A_IJ = Tensor(rank_ids=[\"I\", \"J\"])\n" + \
          "A_JI = A_IJ.swizzleRanks(rank_ids=[\"J\", \"I\"])\n" + \
          "a_j = A_JI.getRoot()\n" + \
          "B_KI = B_IK.swizzleRanks(rank_ids=[\"K\", \"I\"])\n" + \
          "b_k = B_KI.getRoot()\n" + \
          "C_KJ = C_JK.swizzleRanks(rank_ids=[\"K\", \"J\"])\n" + \
          "c_k = C_KJ.getRoot()"

    assert Header.make_header(mapping).gen(depth=0) == hfa


def test_make_header_partitioned():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(
        tree, {}, {"A": {"I": make_uniform_shape([5]), "K": make_uniform_shape([6, 3])}})

    hfa = "A_IJ = Tensor(rank_ids=[\"I\", \"J\"])\n" + \
          "tmp = A_IJ\n" + \
          "tmp = tmp.splitUniform(5, depth=0)\n" + \
          "A_I1I0J = tmp\n" + \
          "A_I1I0J.setRankIds(rank_ids=[\"I1\", \"I0\", \"J\"])\n" + \
          "a_i1 = A_I1I0J.getRoot()\n" + \
          "tmp = B_IK\n" + \
          "tmp = tmp.splitUniform(6, depth=1)\n" + \
          "tmp = tmp.splitUniform(3, depth=2)\n" + \
          "tmp = tmp.splitUniform(5, depth=0)\n" + \
          "B_I1I0K2K1K0 = tmp\n" + \
          "B_I1I0K2K1K0.setRankIds(rank_ids=[\"I1\", \"I0\", \"K2\", \"K1\", \"K0\"])\n" + \
          "b_i1 = B_I1I0K2K1K0.getRoot()\n" + \
          "tmp = C_JK\n" + \
          "tmp = tmp.splitUniform(6, depth=1)\n" + \
          "tmp = tmp.splitUniform(3, depth=2)\n" + \
          "C_JK2K1K0 = tmp\n" + \
          "C_JK2K1K0.setRankIds(rank_ids=[\"J\", \"K2\", \"K1\", \"K0\"])\n" + \
          "c_j = C_JK2K1K0.getRoot()"

    assert Header.make_header(mapping).gen(depth=0) == hfa
