from teaal.ir.part_nodes import *


def test_rank_node():
    assert repr(RankNode("K", 2)) == "(RankNode, K, 2)"

    assert RankNode("K", 2).get_rank() == "K"
    assert RankNode("K", 2).get_priority() == 2
