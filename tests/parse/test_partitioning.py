from lark.lexer import Token
from lark.tree import Tree

from teaal.parse.partitioning import PartitioningParser


def test_nway_shape():
    tree = Tree("nway_shape", [Token("NUMBER", 7)])
    assert PartitioningParser.parse("nway_shape(7)") == tree


def test_uniform_occupancy():
    leader = Tree("leader", [Token("NAME", "A")])
    size = Tree("size", [Token("NUMBER", 6)])
    tree = Tree("uniform_occupancy", [leader, size])
    assert PartitioningParser.parse("uniform_occupancy(A.6)") == tree


def test_uniform_shape():
    tree = Tree("uniform_shape", [Token("NUMBER", 5)])
    assert PartitioningParser.parse("uniform_shape(5)") == tree
