from teaal.ir.part_nodes import *


def test_flatten_node():
    assert repr(FlattenNode(("K", "M"))) == "(FlattenNode, ('K', 'M'))"

    assert FlattenNode(("K", "M")).get_rank() == "KM"
    assert FlattenNode(("K", "M")).get_ranks() == ("K", "M")


def test_rank_node():
    assert repr(RankNode("K")) == "(RankNode, K)"

    assert RankNode("K").get_rank() == "K"
