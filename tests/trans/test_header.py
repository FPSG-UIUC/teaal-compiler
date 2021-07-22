from es2hfa.ir.program import Program
from es2hfa.parse.einsum import Einsum
from es2hfa.parse.mapping import Mapping
from es2hfa.trans.graphics import Graphics
from es2hfa.trans.header import Header
from es2hfa.trans.partitioner import Partitioner
from es2hfa.trans.utils import TransUtils
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
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)
    graphics = Graphics(program)

    hfa = "Z_MN = Tensor(rank_ids=[\"M\", \"N\"])\n" + \
          "z_m = Z_MN.getRoot()\n" + \
          "a_k = A_KM.getRoot()\n" + \
          "b_k = B_KN.getRoot()"
    assert Header.make_header(
        program,
        graphics,
        Partitioner(program, TransUtils())).gen(
        depth=0) == hfa


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
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)
    graphics = Graphics(program)

    hfa = "Z_MN = Tensor(rank_ids=[\"M\", \"N\"])\n" + \
          "z_m = Z_MN.getRoot()\n" + \
          "A_MK = A_KM.swizzleRanks(rank_ids=[\"M\", \"K\"])\n" + \
          "a_m = A_MK.getRoot()\n" + \
          "B_NK = B_KN.swizzleRanks(rank_ids=[\"N\", \"K\"])\n" + \
          "b_n = B_NK.getRoot()"
    assert Header.make_header(
        program,
        graphics,
        Partitioner(program, TransUtils())).gen(
        depth=0) == hfa


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
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)
    graphics = Graphics(program)

    hfa = "Z_M1M0N = Tensor(rank_ids=[\"M1\", \"M0\", \"N\"])\n" + \
          "z_m1 = Z_M1M0N.getRoot()\n" + \
          "tmp0 = A_KM\n" + \
          "tmp1 = tmp0.splitUniform(5, depth=1)\n" + \
          "tmp2 = tmp1.splitUniform(6, depth=0)\n" + \
          "tmp3 = tmp2.splitUniform(3, depth=1)\n" + \
          "A_K2K1K0M1M0 = tmp3\n" + \
          "A_K2K1K0M1M0.setRankIds(rank_ids=[\"K2\", \"K1\", \"K0\", \"M1\", \"M0\"])\n" + \
          "A_M1M0K2K1K0 = A_K2K1K0M1M0.swizzleRanks(rank_ids=[\"M1\", \"M0\", \"K2\", \"K1\", \"K0\"])\n" + \
          "a_m1 = A_M1M0K2K1K0.getRoot()\n" + \
          "tmp4 = B_KN\n" + \
          "tmp5 = tmp4.splitUniform(6, depth=0)\n" + \
          "tmp6 = tmp5.splitUniform(3, depth=1)\n" + \
          "B_K2K1K0N = tmp6\n" + \
          "B_K2K1K0N.setRankIds(rank_ids=[\"K2\", \"K1\", \"K0\", \"N\"])\n" + \
          "B_NK2K1K0 = B_K2K1K0N.swizzleRanks(rank_ids=[\"N\", \"K2\", \"K1\", \"K0\"])\n" + \
          "b_n = B_NK2K1K0.getRoot()"
    assert Header.make_header(
        program,
        graphics,
        Partitioner(program, TransUtils())).gen(
        depth=0) == hfa


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
        spacetime:
            Z:
                space: [N]
                time: [K.pos, M.coord]
    """
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)
    graphics = Graphics(program)

    hfa = "Z_MN = Tensor(rank_ids=[\"M\", \"N\"])\n" + \
          "z_m = Z_MN.getRoot()\n" + \
          "A_MK = A_KM.swizzleRanks(rank_ids=[\"M\", \"K\"])\n" + \
          "a_m = A_MK.getRoot()\n" + \
          "B_NK = B_KN.swizzleRanks(rank_ids=[\"N\", \"K\"])\n" + \
          "b_n = B_NK.getRoot()\n" + \
          "canvas = createCanvas(A_MK, B_NK, Z_MN)"
    assert Header.make_header(
        program,
        graphics,
        Partitioner(program, TransUtils())).gen(
        depth=0) == hfa
