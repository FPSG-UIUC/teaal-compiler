import pytest

from teaal.hifiber import *
from teaal.ir.tensor import Tensor
from teaal.trans.utils import TransUtils


def test_build_expr_bad():
    with pytest.raises(ValueError) as excinfo:
        TransUtils.build_expr(range(5))
    assert str(
        excinfo.value) == "Unable to translate range(0, 5) with type <class 'range'>"


def test_build_expr():
    assert TransUtils.build_expr(5).gen() == "5"
    assert TransUtils.build_expr(1.23).gen() == "1.23"
    assert TransUtils.build_expr("foo").gen() == "\"foo\""
    assert TransUtils.build_expr([1, 2, 3, 4]).gen() == "[1, 2, 3, 4]"
    assert TransUtils.build_expr({1: 2, 3: 4}).gen() == "{1: 2, 3: 4}"


def test_build_rank_ids():
    tensor = Tensor("A", ["I", "J"])
    assert TransUtils.build_rank_ids(tensor).gen() == "rank_ids=[\"I\", \"J\"]"


def test_build_set_rank_ids():
    tensor = Tensor("A", ["I", "J"])
    hifiber = "A_IJ.setRankIds(rank_ids=[\"I\", \"J\"])"
    assert TransUtils.build_set_rank_ids(
        tensor, tensor.tensor_name()).gen(0) == hifiber


def test_build_shape():
    tensor = Tensor("A", ["I", "J"])
    assert TransUtils.build_shape(tensor).gen() == "shape=[I, J]"


def test_build_swizzle():
    new = Tensor("A", ["J", "I"])
    hifiber = "A_JI = A_IJ.swizzleRanks(rank_ids=[\"J\", \"I\"])"
    assert TransUtils.build_swizzle(new, "A_IJ", "A_JI").gen(0) == hifiber


def test_next_tmp():
    utils = TransUtils()
    assert utils.next_tmp() == "tmp0"
    assert utils.next_tmp() == "tmp1"
    assert utils.next_tmp() == "tmp2"


def test_curr_tmp():
    utils = TransUtils()

    with pytest.raises(ValueError) as excinfo:
        utils.curr_tmp()
    assert str(excinfo.value) == "No previous temporary"

    tmp = utils.next_tmp()

    assert utils.curr_tmp() == tmp

def test_sub_hifiber():
    expr = EBinOp(EVar("a"), OAdd(), EBinOp(EVar("b"), OMul(), EVar("c")))
    assert TransUtils.sub_hifiber(expr, EVar("b"), EVar("d")).gen() == "a + d * c"
    assert TransUtils.sub_hifiber(expr, OMul(), ODiv()).gen() == "a + b / c"
    assert TransUtils.sub_hifiber(SExpr(expr), OMul(), ODiv()).gen(0) == "a + b / c"
