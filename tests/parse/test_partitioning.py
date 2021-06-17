from lark.lexer import Token
from lark.tree import Tree

from es2hfa.parse.partitioning import PartitioningParser


def test_uniform_shape():
    tree = Tree("uniform_shape", [Token("NUMBER", 5)])
    assert PartitioningParser.parse("uniform_shape(5)") == tree


def test_divide_uniform():
    tree = Tree("divide_uniform", [Token("NUMBER", 7)])
    assert PartitioningParser.parse("divide_uniform(7)") == tree
