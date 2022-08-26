from lark.lexer import Token
from lark.tree import Tree

from teaal.parse.utils import ParseUtils


def test_find_int():
    leader = Tree("leader", [Token("NAME", "A")])
    size = Tree("size", [Token("NUMBER", 6)])
    tree = Tree("uniform_occupancy", [leader, size])
    assert ParseUtils.find_int(tree, "size") == 6


def test_find_str():
    leader = Tree("leader", [Token("NAME", "A")])
    size = Tree("size", [Token("NUMBER", 6)])
    tree = Tree("uniform_occupancy", [leader, size])
    assert ParseUtils.find_str(tree, "leader") == "A"


def test_next_int():
    tree = Tree("nway_shape", [Token("NUMBER", "5")])
    assert ParseUtils.next_int(tree) == 5


def test_next_str():
    tree = Tree("pos", [Token("NAME", "M0")])
    assert ParseUtils.next_str(tree) == "M0"
