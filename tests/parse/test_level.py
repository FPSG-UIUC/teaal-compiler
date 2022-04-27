from lark.lexer import Token
from lark.tree import Tree

from es2hfa.parse.level import LevelParser


def test_multiple():
    tree = Tree("multiple", [Token("NAME", "PE"), Token("NUMBER", 7)])
    assert LevelParser.parse("PE[0..7]") == tree


def test_single():
    tree = Tree("single", [Token("NAME", "System")])
    assert LevelParser.parse("System") == tree
