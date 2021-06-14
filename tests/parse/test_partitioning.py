from lark.lexer import Token
from lark.tree import Tree

from es2hfa.parse.partitioning import PartitioningParser


def test_uniform_shape():
    tree = Tree("uniform_shape", [Token("NUMBER", 5)])
    assert PartitioningParser.parse("uniform_shape(5)") == tree


def test_divide_into():
    tree = Tree("divide_into", [Token("NUMBER", 7)])
    assert PartitioningParser.parse("divide_into(7)") == tree
