from es2hfa.ir.tensor import Tensor
from es2hfa.parse.tensor import TensorParser
from es2hfa.trans.utils import Utils


def test_build_rank_ids():
    tensor = Tensor.from_tree(TensorParser.parse("A[I, J]"))
    assert Utils.build_rank_ids(tensor).gen() == "rank_ids=[\"I\", \"J\"]"


def test_build_set_rank_ids():
    tensor = Tensor.from_tree(TensorParser.parse("A[I, J]"))
    hfa = "A_IJ.setRankIds(rank_ids=[\"I\", \"J\"])"
    assert Utils.build_set_rank_ids(tensor).gen(0) == hfa


def test_build_swizzle():
    new = Tensor.from_tree(TensorParser.parse("A[J, I]"))
    hfa = "A_JI = A_IJ.swizzleRanks(rank_ids=[\"J\", \"I\"])"
    assert Utils.build_swizzle(new, "A_IJ").gen(0) == hfa
