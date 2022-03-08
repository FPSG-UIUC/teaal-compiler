import pytest

from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.einsum import Einsum
from es2hfa.parse.mapping import Mapping
from es2hfa.trans.partitioner import Partitioner
from es2hfa.trans.utils import TransUtils


def assert_partition(tensor, parts, hfa):
    program, partitioner = build_partitioner(parts)
    ranks = program.get_partitioning().get_all_parts().keys()
    assert partitioner.partition(tensor, ranks).gen(depth=0) == hfa


def build_partitioner(parts):
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
    return program, partitioner


def test_no_partitioning():
    tensor = Tensor("B", ["I", "K"])
    assert_partition(tensor, "", "")


def test_nway_shape():
    tensor = Tensor("B", ["K", "N"])
    spec = """
                M: [nway_shape(5)]
                N: [nway_shape(6), nway_shape(3)]
    """
    hfa = "tmp0 = B_KN\n" + \
          "tmp1 = tmp0.splitUniform((N - 1) // 6 + 1, depth=1)\n" + \
          "tmp2 = tmp1.splitUniform((N - 1) // 3 + 1, depth=1)\n" + \
          "B_KN2N1N0 = tmp2\n" + \
          "B_KN2N1N0.setRankIds(rank_ids=[\"K\", \"N2\", \"N1\", \"N0\"])"

    assert_partition(tensor, spec, hfa)


def test_uniform_occupancy_leader():
    tensor = Tensor("A", ["K", "M"])
    spec = """
                K: [uniform_occupancy(A.5)]
    """
    hfa = "tmp0 = A_KM\n" + \
          "tmp1 = tmp0.splitEqual(5)\n" + \
          "A_K1K0M = tmp1\n" + \
          "A_K1K0M.setRankIds(rank_ids=[\"K1\", \"K0\", \"M\"])"

    assert_partition(tensor, spec, hfa)


def test_uniform_occupancy_follower():
    tensor = Tensor("B", ["K", "N"])
    spec = """
                K: [uniform_occupancy(A.5)]
    """
    hfa = "tmp0 = B_KN\n" + \
          "tmp1 = tmp0.splitNonUniform(a_k1)\n" + \
          "B_K1K0N = tmp1\n" + \
          "B_K1K0N.setRankIds(rank_ids=[\"K1\", \"K0\", \"N\"])"

    program, partitioner = build_partitioner(spec)
    program.apply_partitioning(program.get_tensor("A"), "K")
    assert partitioner.partition(tensor, {"K"}).gen(depth=0) == hfa


def test_uniform_occupancy_multiple():
    spec = """
                K: [uniform_occupancy(A.6), uniform_occupancy(A.3)]
        loop-order:
            Z: [K2, K1, K0, M, N]
    """
    hfa = "tmp0 = A_KM\n" + \
          "tmp1 = tmp0.splitEqual(6)\n" + \
          "A_K2K1IM = tmp1\n" + \
          "A_K2K1IM.setRankIds(rank_ids=[\"K2\", \"K1I\", \"M\"])"

    program, partitioner = build_partitioner(spec)
    tensor = program.get_tensor("A")
    tensor.from_fiber()
    assert partitioner.partition(tensor, {"K"}).gen(depth=0) == hfa

    hfa = "tmp2 = A_K1IM\n" + \
          "tmp3 = tmp2.splitEqual(3)\n" + \
          "A_K1K0M = tmp3\n" + \
          "A_K1K0M.setRankIds(rank_ids=[\"K1\", \"K0\", \"M\"])"

    tensor.pop()
    tensor.from_fiber()
    assert partitioner.partition(tensor, {"K1I"}).gen(depth=0) == hfa


def test_uniform_shape():
    tensor = Tensor("B", ["K", "N"])
    spec = """
                M: [uniform_shape(5)]
                N: [uniform_shape(6), uniform_shape(3)]
    """
    hfa = "tmp0 = B_KN\n" + \
          "tmp1 = tmp0.splitUniform(6, depth=1)\n" + \
          "tmp2 = tmp1.splitUniform(3, depth=2)\n" + \
          "B_KN2N1N0 = tmp2\n" + \
          "B_KN2N1N0.setRankIds(rank_ids=[\"K\", \"N2\", \"N1\", \"N0\"])"

    assert_partition(tensor, spec, hfa)


def test_mixed():
    spec = """
                K:
                    - uniform_shape(500)
                    - uniform_shape(250)
                    - uniform_occupancy(A.100)
                    - uniform_occupancy(A.50)
                    - uniform_shape(10)
                    - uniform_occupancy(A.6)
                    - uniform_shape(4)
                    - uniform_shape(2)
    """
    hfa = "tmp0 = A_KM\n" + \
          "tmp1 = tmp0.splitUniform(500, depth=0)\n" + \
          "tmp2 = tmp1.splitUniform(250, depth=1)\n" + \
          "A_K8K7K6IM = tmp2\n" + \
          "A_K8K7K6IM.setRankIds(rank_ids=[\"K8\", \"K7\", \"K6I\", \"M\"])"

    program, partitioner = build_partitioner(spec)
    tensor = program.get_tensor("A")
    tensor.from_fiber()
    assert partitioner.partition(tensor, {"K"}).gen(depth=0) == hfa

    hfa = "tmp3 = A_K6IM\n" + \
          "tmp4 = tmp3.splitEqual(100)\n" + \
          "A_K6K5IM = tmp4\n" + \
          "A_K6K5IM.setRankIds(rank_ids=[\"K6\", \"K5I\", \"M\"])"

    tensor.pop()
    tensor.pop()
    tensor.from_fiber()
    assert partitioner.partition(tensor, {"K6I"}).gen(depth=0) == hfa

    hfa = "tmp5 = A_K5IM\n" + \
          "tmp6 = tmp5.splitEqual(50)\n" + \
          "tmp7 = tmp6.splitUniform(10, depth=1)\n" + \
          "A_K5K4K3IM = tmp7\n" + \
          "A_K5K4K3IM.setRankIds(rank_ids=[\"K5\", \"K4\", \"K3I\", \"M\"])"

    tensor.pop()
    tensor.from_fiber()
    assert partitioner.partition(tensor, {"K5I"}).gen(depth=0) == hfa

    hfa = "tmp8 = A_K3IM\n" + \
          "tmp9 = tmp8.splitEqual(6)\n" + \
          "tmp10 = tmp9.splitUniform(4, depth=1)\n" + \
          "tmp11 = tmp10.splitUniform(2, depth=2)\n" + \
          "A_K3K2K1K0M = tmp11\n" + \
          "A_K3K2K1K0M.setRankIds(rank_ids=[\"K3\", \"K2\", \"K1\", \"K0\", \"M\"])"

    tensor.pop()
    tensor.pop()
    tensor.from_fiber()
    assert partitioner.partition(tensor, {"K3I"}).gen(depth=0) == hfa


def assert_unpartition(spec, hfa):
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
        """ + spec
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    for tensor in program.get_tensors():
        program.apply_all_partitioning(tensor)

    partitioner = Partitioner(program, TransUtils())
    assert partitioner.unpartition(program.get_output()).gen(0) == hfa


def test_unpartition_none():
    spec = """
                K: [uniform_shape(5)]
    """
    hfa = ""
    assert_unpartition(spec, hfa)


def test_unpartition_one():
    spec = """
                N: [uniform_shape(6), uniform_shape(3)]
    """
    hfa = "tmp0 = Z_MN2N1N0\n" + \
          "tmp1 = tmp0.flattenRanks(depth=1, levels=2, coord_style=\"absolute\")\n" + \
          "Z_MN = tmp1\n" + \
          "Z_MN.setRankIds(rank_ids=[\"M\", \"N\"])"
    assert_unpartition(spec, hfa)


def test_unpartition_all():
    spec = """
                M: [uniform_shape(5)]
                N: [uniform_shape(6), uniform_shape(3)]
    """
    hfa = "tmp0 = Z_M1M0N2N1N0\n" + \
          "tmp1 = tmp0.flattenRanks(depth=0, levels=1, coord_style=\"absolute\")\n" + \
          "tmp2 = tmp1.flattenRanks(depth=1, levels=2, coord_style=\"absolute\")\n" + \
          "Z_MN = tmp2\n" + \
          "Z_MN.setRankIds(rank_ids=[\"M\", \"N\"])"
    assert_unpartition(spec, hfa)
