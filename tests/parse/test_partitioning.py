from lark.lexer import Token
from lark.tree import Tree

from teaal.parse.partitioning import PartitioningParser

from tests.utils.parse_tree import *


def test_flatten():
    tree = Tree("flatten", [])
    assert PartitioningParser.parse_partitioning("flatten()") == tree


def test_nway_shape():
    tree = Tree("nway_shape", [Tree("int_sz", [Token("NUMBER", 7)])])
    assert PartitioningParser.parse_partitioning("nway_shape(7)") == tree


def test_rank():
    tree = make_prank("M")
    assert PartitioningParser.parse_ranks("M") == tree


def test_ranks():
    tree = make_pranks(["K", "M", "N"])
    assert PartitioningParser.parse_ranks("(K, M, N)") == tree


def test_uniform_occupancy():
    leader = Tree("leader", [Token("NAME", "A")])
    size = Tree("int_sz", [Token("NUMBER", 6)])
    tree = Tree("uniform_occupancy", [leader, size])
    assert PartitioningParser.parse_partitioning(
        "uniform_occupancy(A.6)") == tree


def test_uniform_shape():
    tree = Tree("uniform_shape", [Tree("int_sz", [Token("NUMBER", 5)])])
    assert PartitioningParser.parse_partitioning("uniform_shape(5)") == tree


def test_uniform_shape_name_shape():
    tree = Tree("uniform_shape", [Tree("str_sz", [Token("NAME", "M0")])])
    assert PartitioningParser.parse_partitioning("uniform_shape(M0)") == tree
