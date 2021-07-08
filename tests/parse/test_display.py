from lark.lexer import Token
from lark.tree import Tree

from es2hfa.parse.display import DisplayParser


def test_default():
    tree = Tree("pos", [Token("NAME", "M0")])
    assert DisplayParser.parse("M0") == tree


def test_pos():
    tree = Tree("pos", [Token("NAME", "M0")])
    assert DisplayParser.parse("M0.pos") == tree


def test_coord():
    tree = Tree("coord", [Token("NAME", "M0")])
    assert DisplayParser.parse("M0.coord") == tree
