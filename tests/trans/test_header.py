from es2hfa.ir.iter_graph import IterationGraph
from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.einsum import Einsum
from es2hfa.parse.mapping import Mapping
from es2hfa.trans.graphics import Graphics
from es2hfa.trans.header import Header
from es2hfa.trans.partitioner import Partitioner
from es2hfa.trans.utils import TransUtils


def build_header(mapping):
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
    """ + mapping

    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)
    graphics = Graphics(program)

    header = Header(program, Partitioner(program, TransUtils()))

    return header, graphics, program


def test_make_global_header():
    mapping = """
        loop-order:
            Z: [K, M, N]
    """

    hfa = "Z_MN = Tensor(rank_ids=[\"M\", \"N\"])\n" + \
          "z_m = Z_MN.getRoot()\n" + \
          "a_k = A_KM.getRoot()\n" + \
          "b_k = B_KN.getRoot()"

    header, graphics, _ = build_header(mapping)
    assert header.make_global_header(graphics).gen(depth=0) == hfa


def test_make_global_header_swizzle():
    hfa = "Z_MN = Tensor(rank_ids=[\"M\", \"N\"])\n" + \
          "A_MK = A_KM.swizzleRanks(rank_ids=[\"M\", \"K\"])\n" + \
          "B_NK = B_KN.swizzleRanks(rank_ids=[\"N\", \"K\"])\n" + \
          "z_m = Z_MN.getRoot()\n" + \
          "a_m = A_MK.getRoot()\n" + \
          "b_n = B_NK.getRoot()"

    header, graphics, _ = build_header("")
    assert header.make_global_header(graphics).gen(depth=0) == hfa


def test_make_global_header_partitioned():
    mapping = """
        partitioning:
            Z:
                K: [uniform_shape(6), uniform_shape(3)]
                M: [uniform_shape(5)]
    """

    hfa = "Z_M1M0N = Tensor(rank_ids=[\"M1\", \"M0\", \"N\"])\n" + \
          "tmp0 = A_KM\n" + \
          "tmp1 = tmp0.splitUniform(5, depth=1)\n" + \
          "tmp2 = tmp1.splitUniform(6, depth=0)\n" + \
          "tmp3 = tmp2.splitUniform(3, depth=1)\n" + \
          "A_K2K1K0M1M0 = tmp3\n" + \
          "A_K2K1K0M1M0.setRankIds(rank_ids=[\"K2\", \"K1\", \"K0\", \"M1\", \"M0\"])\n" + \
          "A_M1M0K2K1K0 = A_K2K1K0M1M0.swizzleRanks(rank_ids=[\"M1\", \"M0\", \"K2\", \"K1\", \"K0\"])\n" + \
          "tmp4 = B_KN\n" + \
          "tmp5 = tmp4.splitUniform(6, depth=0)\n" + \
          "tmp6 = tmp5.splitUniform(3, depth=1)\n" + \
          "B_K2K1K0N = tmp6\n" + \
          "B_K2K1K0N.setRankIds(rank_ids=[\"K2\", \"K1\", \"K0\", \"N\"])\n" + \
          "B_NK2K1K0 = B_K2K1K0N.swizzleRanks(rank_ids=[\"N\", \"K2\", \"K1\", \"K0\"])\n" + \
          "z_m1 = Z_M1M0N.getRoot()\n" + \
          "a_m1 = A_M1M0K2K1K0.getRoot()\n" + \
          "b_n = B_NK2K1K0.getRoot()"

    header, graphics, _ = build_header(mapping)
    assert header.make_global_header(graphics).gen(depth=0) == hfa


def test_make_global_header_displayed():
    mapping = """
        spacetime:
            Z:
                space: [N]
                time: [K.pos, M.coord]
    """

    hfa = "Z_MN = Tensor(rank_ids=[\"M\", \"N\"])\n" + \
          "A_MK = A_KM.swizzleRanks(rank_ids=[\"M\", \"K\"])\n" + \
          "B_NK = B_KN.swizzleRanks(rank_ids=[\"N\", \"K\"])\n" + \
          "z_m = Z_MN.getRoot()\n" + \
          "a_m = A_MK.getRoot()\n" + \
          "b_n = B_NK.getRoot()\n" + \
          "canvas = createCanvas(A_MK, B_NK, Z_MN)"

    header, graphics, _ = build_header(mapping)
    assert header.make_global_header(graphics).gen(depth=0) == hfa


def test_make_loop_header_empty():
    header, _, _ = build_header("")
    assert header.make_loop_header("M").gen(depth=0) == ""


def test_make_loop_header_leader():
    mapping = """
        partitioning:
            Z:
                M: [uniform_occupancy(A.6)]
    """

    hfa = "A_MK = Tensor.fromFiber(rank_ids=[\"M\", \"K\"], fiber=a_m)\n" + \
          "tmp0 = A_MK\n" + \
          "tmp1 = tmp0.splitEqual(6)\n" + \
          "A_M1M0K = tmp1\n" + \
          "A_M1M0K.setRankIds(rank_ids=[\"M1\", \"M0\", \"K\"])\n" + \
          "a_m1 = A_M1M0K.getRoot()"

    header, graphics, _ = build_header(mapping)
    header.make_global_header(graphics)
    assert header.make_loop_header("M1").gen(depth=0) == hfa


def test_make_loop_header_follower():
    mapping = """
        loop-order:
            Z: [K1, K0, M, N]
        partitioning:
            Z:
                K: [uniform_occupancy(A.6)]
    """

    hfa = "A_KM = Tensor.fromFiber(rank_ids=[\"K\", \"M\"], fiber=a_k)\n" + \
          "tmp0 = A_KM\n" + \
          "tmp1 = tmp0.splitEqual(6)\n" + \
          "A_K1K0M = tmp1\n" + \
          "A_K1K0M.setRankIds(rank_ids=[\"K1\", \"K0\", \"M\"])\n" + \
          "a_k1 = A_K1K0M.getRoot()\n" + \
          "B_KN = Tensor.fromFiber(rank_ids=[\"K\", \"N\"], fiber=b_k)\n" + \
          "tmp2 = B_KN\n" + \
          "tmp3 = tmp2.splitNonUniform(a_k1)\n" + \
          "B_K1K0N = tmp3\n" + \
          "B_K1K0N.setRankIds(rank_ids=[\"K1\", \"K0\", \"N\"])\n" + \
          "b_k1 = B_K1K0N.getRoot()"

    header, graphics, _ = build_header(mapping)
    header.make_global_header(graphics)
    assert header.make_loop_header("K1").gen(depth=0) == hfa


def test_make_loop_header_swizzle():
    mapping = """
        loop-order:
            Z: [N, K1, M, K0]
        partitioning:
            Z:
                K: [uniform_occupancy(A.6)]
    """

    hfa = "A_KM = Tensor.fromFiber(rank_ids=[\"K\", \"M\"], fiber=a_k)\n" + \
          "tmp0 = A_KM\n" + \
          "tmp1 = tmp0.splitEqual(6)\n" + \
          "A_K1K0M = tmp1\n" + \
          "A_K1K0M.setRankIds(rank_ids=[\"K1\", \"K0\", \"M\"])\n" + \
          "A_K1MK0 = A_K1K0M.swizzleRanks(rank_ids=[\"K1\", \"M\", \"K0\"])\n" + \
          "a_k1 = A_K1MK0.getRoot()\n" + \
          "B_K = Tensor.fromFiber(rank_ids=[\"K\"], fiber=b_k)\n" + \
          "tmp2 = B_K\n" + \
          "tmp3 = tmp2.splitNonUniform(a_k1)\n" + \
          "B_K1K0 = tmp3\n" + \
          "B_K1K0.setRankIds(rank_ids=[\"K1\", \"K0\"])\n" + \
          "b_k1 = B_K1K0.getRoot()"

    header, graphics, program = build_header(mapping)
    header.make_global_header(graphics)

    graph = IterationGraph(program)
    graph.pop()

    assert header.make_loop_header("K1").gen(depth=0) == hfa


def test_make_output():
    mapping = """
        loop-order:
            Z: [K, M, N]
    """

    hfa = "Z_MN = Tensor(rank_ids=[\"M\", \"N\"])"

    header, _, program = build_header(mapping)
    assert header.make_output().gen(depth=0) == hfa


def test_make_swizzle_root():
    hfa = "A_MK = A_KM.swizzleRanks(rank_ids=[\"M\", \"K\"])\n" + \
          "a_m = A_MK.getRoot()"

    header, _, _ = build_header("")
    tensor = Tensor("A", ["K", "M"])
    assert header.make_swizzle_root(tensor).gen(depth=0) == hfa


def test_make_tensor_from_fiber():
    hfa = "A_KM = Tensor.fromFiber(rank_ids=[\"K\", \"M\"], fiber=a_k)"
    tensor = Tensor("A", ["K", "M"])
    assert Header.make_tensor_from_fiber(tensor).gen(depth=0) == hfa
