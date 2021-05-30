from es2hfa.ir.mapping import Mapping
from es2hfa.parse.einsum import EinsumParser
from es2hfa.parse.tensor import TensorParser
from es2hfa.trans.header import Header


def test_make_header():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, [])

    hfa = "a_i = A_IJ.getRoot()\n" + \
          "b_i = B_IK.getRoot()\n" + \
          "c_j = C_JK.getRoot()"

    assert Header.make_header(mapping).gen(depth=0) == hfa


def test_make_header_swizzle():
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, [])

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, ["K", "J", "I"])

    hfa = "A_JI = A_IJ.swizzleRanks([\"J\", \"I\"])\n" + \
          "a_j = A_JI.getRoot()\n" + \
          "B_KI = B_IK.swizzleRanks([\"K\", \"I\"])\n" + \
          "b_k = B_KI.getRoot()\n" + \
          "C_KJ = C_JK.swizzleRanks([\"K\", \"J\"])\n" + \
          "c_k = C_KJ.getRoot()"

    assert Header.make_header(mapping).gen(depth=0) == hfa
