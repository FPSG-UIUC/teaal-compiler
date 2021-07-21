from lark.lexer import Token
from lark.tree import Tree

from es2hfa.parse.partitioning import PartitioningParser


def test_nway_shape():
    tree = Tree("nway_shape", [Token("NUMBER", 7)])
    assert PartitioningParser.parse("nway_shape(7)") == tree


def test_uniform_occupancy():
    tree = Tree("uniform_occupancy", [Token("NAME", "A"), Token("NUMBER", 6)])
    assert PartitioningParser.parse("uniform_occupancy(A.6)") == tree


def test_uniform_shape():
    tree = Tree("uniform_shape", [Token("NUMBER", 5)])
    assert PartitioningParser.parse("uniform_shape(5)") == tree
