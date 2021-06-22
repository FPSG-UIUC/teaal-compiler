import pytest

from es2hfa.ir.mapping import Mapping
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.einsum import EinsumParser
from es2hfa.parse.tensor import TensorParser
from es2hfa.trans.partitioning import Partitioner
from tests.utils.parse_tree import make_nway_shape, make_var, make_uniform_shape


def assert_partition(tensor, parts, hfa):
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, {})

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, {}, parts)

    partitioner = Partitioner(mapping)
    assert partitioner.partition(tensor).gen(depth=0) == hfa


def test_no_partitioning():
    tensor = Tensor("B", ["I", "K"])
    assert_partition(tensor, {}, "")


def test_uniform_shape():
    tensor = Tensor("C", ["J", "K"])
    parts = {"A": {"I": make_uniform_shape(
        [5]), "K": make_uniform_shape([6, 3])}}
    hfa = "tmp = C_JK\n" + \
          "tmp = tmp.splitUniform(6, depth=1)\n" + \
          "tmp = tmp.splitUniform(3, depth=2)\n" + \
          "C_JK2K1K0 = tmp\n" + \
          "C_JK2K1K0.setRankIds(rank_ids=[\"J\", \"K2\", \"K1\", \"K0\"])"

    assert_partition(tensor, parts, hfa)


def test_nway_shape():
    tensor = Tensor("C", ["J", "K"])
    parts = {"A": {"I": make_nway_shape(
        [5]), "K": make_nway_shape([6, 3])}}
    hfa = "tmp = C_JK\n" + \
          "tmp = tmp.splitUniform((K - 1) // 6 + 1, depth=1)\n" + \
          "tmp = tmp.splitUniform((K - 1) // 3 + 1, depth=1)\n" + \
          "C_JK2K1K0 = tmp\n" + \
          "C_JK2K1K0.setRankIds(rank_ids=[\"J\", \"K2\", \"K1\", \"K0\"])"

    assert_partition(tensor, parts, hfa)


def test_bad_part():
    tensor = Tensor("B", ["I", "K"])
    with pytest.raises(ValueError) as excinfo:
        assert_partition(tensor, {"A": {"I": [make_var("a")]}}, "")

    assert str(excinfo.value) == "Unknown partitioning style: var"


def assert_unpartition(part, hfa):
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, {})

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, {}, part)

    mapping.apply_partitioning(mapping.get_output())

    partitioner = Partitioner(mapping)
    assert partitioner.unpartition(mapping.get_output()).gen(0) == hfa


def test_unpartition_none():
    part = {"A": {"K": make_uniform_shape([6, 3])}}
    hfa = ""
    assert_unpartition(part, hfa)


def test_unpartition_one():
    part = {"A": {"J": make_uniform_shape([6, 3])}}
    hfa = "tmp = A_IJ2J1J0\n" + \
          "tmp = tmp.flattenRanks(depth=1, levels=2, coord_style=\"absolute\")\n" + \
          "A_IJ = tmp\n" + \
          "A_IJ.setRankIds(rank_ids=[\"I\", \"J\"])"
    assert_unpartition(part, hfa)


def test_unpartition_all():
    part = {"A": {"I": make_uniform_shape(
        [5]), "J": make_uniform_shape([6, 3])}}
    hfa = "tmp = A_I1I0J2J1J0\n" + \
          "tmp = tmp.flattenRanks(depth=0, levels=1, coord_style=\"absolute\")\n" + \
          "tmp = tmp.flattenRanks(depth=1, levels=2, coord_style=\"absolute\")\n" + \
          "A_IJ = tmp\n" + \
          "A_IJ.setRankIds(rank_ids=[\"I\", \"J\"])"
    assert_unpartition(part, hfa)
