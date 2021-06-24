from es2hfa.ir.mapping import Mapping
from es2hfa.parse.input import Input
from es2hfa.trans.canvas import Canvas
from es2hfa.trans.header import Header
from tests.utils.parse_tree import make_uniform_shape


def test_make_header():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
            C: [M, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
        loop-order:
            Z: [K, M, N]
    """
    mapping = Mapping(Input.from_str(yaml))
    mapping.add_einsum(0)
    canvas = Canvas(mapping)

    hfa = "Z_MN = Tensor(rank_ids=[\"M\", \"N\"])\n" + \
          "z_m = Z_MN.getRoot()\n" + \
          "a_k = A_KM.getRoot()\n" + \
          "b_k = B_KN.getRoot()"
    assert Header.make_header(mapping, canvas).gen(depth=0) == hfa


def test_make_header_swizzle():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
            C: [M, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    """
    mapping = Mapping(Input.from_str(yaml))
    mapping.add_einsum(0)
    canvas = Canvas(mapping)

    hfa = "Z_MN = Tensor(rank_ids=[\"M\", \"N\"])\n" + \
          "z_m = Z_MN.getRoot()\n" + \
          "A_MK = A_KM.swizzleRanks(rank_ids=[\"M\", \"K\"])\n" + \
          "a_m = A_MK.getRoot()\n" + \
          "B_NK = B_KN.swizzleRanks(rank_ids=[\"N\", \"K\"])\n" + \
          "b_n = B_NK.getRoot()"
    assert Header.make_header(mapping, canvas).gen(depth=0) == hfa


def test_make_header_partitioned():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
            C: [M, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
        partitioning:
            Z:
                K: [uniform_shape(6), uniform_shape(3)]
                M: [uniform_shape(5)]
    """
    mapping = Mapping(Input.from_str(yaml))
    mapping.add_einsum(0)
    canvas = Canvas(mapping)

    hfa = "Z_MN = Tensor(rank_ids=[\"M\", \"N\"])\n" + \
          "tmp = Z_MN\n" + \
          "tmp = tmp.splitUniform(5, depth=0)\n" + \
          "Z_M1M0N = tmp\n" + \
          "Z_M1M0N.setRankIds(rank_ids=[\"M1\", \"M0\", \"N\"])\n" + \
          "z_m1 = Z_M1M0N.getRoot()\n" + \
          "tmp = A_KM\n" + \
          "tmp = tmp.splitUniform(5, depth=1)\n" + \
          "tmp = tmp.splitUniform(6, depth=0)\n" + \
          "tmp = tmp.splitUniform(3, depth=1)\n" + \
          "A_K2K1K0M1M0 = tmp\n" + \
          "A_K2K1K0M1M0.setRankIds(rank_ids=[\"K2\", \"K1\", \"K0\", \"M1\", \"M0\"])\n" + \
          "A_M1M0K2K1K0 = A_K2K1K0M1M0.swizzleRanks(rank_ids=[\"M1\", \"M0\", \"K2\", \"K1\", \"K0\"])\n" + \
          "a_m1 = A_M1M0K2K1K0.getRoot()\n" + \
          "tmp = B_KN\n" + \
          "tmp = tmp.splitUniform(6, depth=0)\n" + \
          "tmp = tmp.splitUniform(3, depth=1)\n" + \
          "B_K2K1K0N = tmp\n" + \
          "B_K2K1K0N.setRankIds(rank_ids=[\"K2\", \"K1\", \"K0\", \"N\"])\n" + \
          "B_NK2K1K0 = B_K2K1K0N.swizzleRanks(rank_ids=[\"N\", \"K2\", \"K1\", \"K0\"])\n" + \
          "b_n = B_NK2K1K0.getRoot()"
    assert Header.make_header(mapping, canvas).gen(depth=0) == hfa


def test_make_header_displayed():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
            C: [M, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
        display:
            Z:
                space: [N]
                time: [K, M]
    """
    mapping = Mapping(Input.from_str(yaml))
    mapping.add_einsum(0)
    canvas = Canvas(mapping)

    hfa = "Z_MN = Tensor(rank_ids=[\"M\", \"N\"])\n" + \
          "z_m = Z_MN.getRoot()\n" + \
          "A_MK = A_KM.swizzleRanks(rank_ids=[\"M\", \"K\"])\n" + \
          "a_m = A_MK.getRoot()\n" + \
          "B_NK = B_KN.swizzleRanks(rank_ids=[\"N\", \"K\"])\n" + \
          "b_n = B_NK.getRoot()\n" + \
          "canvas = createCanvas(A_MK, B_NK, Z_MN)"
    assert Header.make_header(mapping, canvas).gen(depth=0) == hfa
