from teaal.ir.node import Node


def test_node():
    assert Node() == Node()
    assert Node() != ""

    set_ = set()
    set_.add(Node())

    assert Node() in set_
    assert "" not in set_
