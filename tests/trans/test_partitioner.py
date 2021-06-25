import pytest

from es2hfa.ir.mapping import Mapping
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.input import Input
from es2hfa.trans.partitioning import Partitioner
from es2hfa.trans.utils import Utils


def assert_partition(tensor, parts, hfa):
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
        partitioning:
            Z:
        """ + parts
    mapping = Mapping(Input.from_str(yaml))
    mapping.add_einsum(0)

    partitioner = Partitioner(mapping, Utils())
    assert partitioner.partition(tensor).gen(depth=0) == hfa


def test_no_partitioning():
    tensor = Tensor("B", ["I", "K"])
    assert_partition(tensor, "", "")


def test_uniform_shape():
    tensor = Tensor("B", ["K", "N"])
    part = """
                M: [uniform_shape(5)]
                N: [uniform_shape(6), uniform_shape(3)]
    """
    hfa = "tmp0 = B_KN\n" + \
          "tmp1 = tmp0.splitUniform(6, depth=1)\n" + \
          "tmp2 = tmp1.splitUniform(3, depth=2)\n" + \
          "B_KN2N1N0 = tmp2\n" + \
          "B_KN2N1N0.setRankIds(rank_ids=[\"K\", \"N2\", \"N1\", \"N0\"])"

    assert_partition(tensor, part, hfa)


def test_nway_shape():
    tensor = Tensor("B", ["K", "N"])
    part = """
                M: [nway_shape(5)]
                N: [nway_shape(6), nway_shape(3)]
    """
    hfa = "tmp0 = B_KN\n" + \
          "tmp1 = tmp0.splitUniform((N - 1) // 6 + 1, depth=1)\n" + \
          "tmp2 = tmp1.splitUniform((N - 1) // 3 + 1, depth=1)\n" + \
          "B_KN2N1N0 = tmp2\n" + \
          "B_KN2N1N0.setRankIds(rank_ids=[\"K\", \"N2\", \"N1\", \"N0\"])"

    assert_partition(tensor, part, hfa)


def assert_unpartition(part, hfa):
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
        partitioning:
            Z:
        """ + part
    mapping = Mapping(Input.from_str(yaml))
    mapping.add_einsum(0)

    for tensor in mapping.get_tensors():
        mapping.apply_partitioning(tensor)

    partitioner = Partitioner(mapping, Utils())
    assert partitioner.unpartition(mapping.get_output()).gen(0) == hfa


def test_unpartition_none():
    part = """
                K: [uniform_shape(5)]
    """
    hfa = ""
    assert_unpartition(part, hfa)


def test_unpartition_one():
    part = """
                N: [uniform_shape(6), uniform_shape(3)]
    """
    hfa = "tmp0 = Z_MN2N1N0\n" + \
          "tmp1 = tmp0.flattenRanks(depth=1, levels=2, coord_style=\"absolute\")\n" + \
          "Z_MN = tmp1\n" + \
          "Z_MN.setRankIds(rank_ids=[\"M\", \"N\"])"
    assert_unpartition(part, hfa)


def test_unpartition_all():
    part = """
                M: [uniform_shape(5)]
                N: [uniform_shape(6), uniform_shape(3)]
    """
    hfa = "tmp0 = Z_M1M0N2N1N0\n" + \
          "tmp1 = tmp0.flattenRanks(depth=0, levels=1, coord_style=\"absolute\")\n" + \
          "tmp2 = tmp1.flattenRanks(depth=1, levels=2, coord_style=\"absolute\")\n" + \
          "Z_MN = tmp2\n" + \
          "Z_MN.setRankIds(rank_ids=[\"M\", \"N\"])"
    assert_unpartition(part, hfa)
