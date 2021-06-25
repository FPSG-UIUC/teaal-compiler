import pytest

from es2hfa.ir.tensor import Tensor
from es2hfa.trans.utils import Utils


def test_build_rank_ids():
    tensor = Tensor("A", ["I", "J"])
    assert Utils.build_rank_ids(tensor).gen() == "rank_ids=[\"I\", \"J\"]"


def test_build_set_rank_ids():
    tensor = Tensor("A", ["I", "J"])
    hfa = "A_IJ.setRankIds(rank_ids=[\"I\", \"J\"])"
    assert Utils.build_set_rank_ids(tensor).gen(0) == hfa


def test_build_swizzle():
    new = Tensor("A", ["J", "I"])
    hfa = "A_JI = A_IJ.swizzleRanks(rank_ids=[\"J\", \"I\"])"
    assert Utils.build_swizzle(new, "A_IJ").gen(0) == hfa


def test_next_tmp():
    utils = Utils()
    assert utils.next_tmp() == "tmp0"
    assert utils.next_tmp() == "tmp1"
    assert utils.next_tmp() == "tmp2"


def test_curr_tmp():
    utils = Utils()

    with pytest.raises(ValueError) as excinfo:
        utils.curr_tmp()
    assert str(excinfo.value) == "No previous temporary"

    tmp = utils.next_tmp()

    assert utils.curr_tmp() == tmp
