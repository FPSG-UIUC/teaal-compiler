from es2hfa.ir.nodes import *


def test_node():
    assert Node() == Node()
    assert Node() != ""

    set_ = set()
    set_.add(Node())

    assert Node() in set_
    assert "" not in set_


def test_fiber_node():
    assert repr(FiberNode("a_k")) == "(FiberNode, a_k)"


def test_loop_node():
    assert repr(LoopNode("K")) == "(LoopNode, K)"


def test_part_node():
    assert repr(PartNode("A", ("K",))) == "(PartNode, A, ('K',))"


def test_rank_node():
    assert repr(RankNode("A", "K")) == "(RankNode, A, K)"


def test_sr_node():
    assert repr(SRNode("A", {"K"})) == "(SRNode, A, {'K'})"
