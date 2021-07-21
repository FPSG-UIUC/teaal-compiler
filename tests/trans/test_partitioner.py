import pytest

from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.einsum import Einsum
from es2hfa.parse.mapping import Mapping
from es2hfa.trans.partitioner import Partitioner
from es2hfa.trans.utils import TransUtils


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
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    partitioner = Partitioner(program, TransUtils())
    assert partitioner.partition(tensor).gen(depth=0) == hfa


def test_no_partitioning():
    tensor = Tensor("B", ["I", "K"])
    assert_partition(tensor, "", "")


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


def test_dynamic_part_during_static():
    tensor = Tensor("B", ["K", "N"])
    part = """
                K: [uniform_occupancy(A.5)]
    """
    with pytest.raises(ValueError) as excinfo:
        assert_partition(tensor, part, "")
    assert str(excinfo.value) \
        == "Dynamic or unknown partitioning style: uniform_occupancy"


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
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    for tensor in program.get_tensors():
        program.apply_all_partitioning(tensor)

    partitioner = Partitioner(program, TransUtils())
    assert partitioner.unpartition(program.get_output()).gen(0) == hfa


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
