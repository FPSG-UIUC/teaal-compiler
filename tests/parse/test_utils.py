from lark.lexer import Token
from lark.tree import Tree

from es2hfa.parse.utils import ParseUtils


def test_next_str():
    tree = Tree("pos", [Token("NAME", "M0")])
    assert ParseUtils.next_str(tree) == "M0"


def test_next_int():
    tree = Tree("nway_shape", [Token("NUMBER", "5")])
    assert ParseUtils.next_int(tree) == 5
