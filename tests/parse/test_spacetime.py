from lark.lexer import Token
from lark.tree import Tree

from teaal.parse.spacetime import SpaceTimeParser


def test_default():
    tree = Tree("pos", [Token("NAME", "M0")])
    assert SpaceTimeParser.parse("M0") == tree


def test_pos():
    tree = Tree("pos", [Token("NAME", "M0")])
    assert SpaceTimeParser.parse("M0.pos") == tree


def test_coord():
    tree = Tree("coord", [Token("NAME", "M0")])
    assert SpaceTimeParser.parse("M0.coord") == tree
