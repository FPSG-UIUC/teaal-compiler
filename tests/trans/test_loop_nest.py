from es2hfa.ir.program import Program
from es2hfa.ir.iter_graph import IterationGraph
from es2hfa.parse.einsum import Einsum
from es2hfa.parse.mapping import Mapping
from es2hfa.trans.equation import Equation
from es2hfa.trans.graphics import Graphics
from es2hfa.trans.header import Header
from es2hfa.trans.loop_nest import LoopNest
from es2hfa.trans.partitioner import Partitioner
from es2hfa.trans.utils import TransUtils


def build_loop_nest(yaml):
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)
    trans_utils = TransUtils()

    graphics = Graphics(program)
    partitioner = Partitioner(program, trans_utils)

    header = Header(program, partitioner)
    header.make_global_header(graphics)

    eqn = Equation(program)
    graph = IterationGraph(program)
    return LoopNest.make_loop_nest(eqn, graph, graphics, header).gen(depth=0)


def test_no_loops():
    yaml = """
    einsum:
        declaration:
            Z: []
        expressions:
            - Z[] = a + b
    """

    hfa = "z_ref += a + b"

    assert build_loop_nest(yaml) == hfa


def test_no_partitioning():
    yaml = """
    einsum:
        declaration:
            A: [K, M]
            B: [K, N]
            Z: [M, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    """

    hfa = "for m, (z_n, a_k) in z_m << a_m:\n" + \
          "    for n, (z_ref, b_k) in z_n << b_n:\n" + \
          "        for k, (a_val, b_val) in a_k & b_k:\n" + \
          "            z_ref += a_val * b_val"

    assert build_loop_nest(yaml) == hfa


def test_static_partitioning():
    yaml = """
    einsum:
        declaration:
            A: [K, M]
            B: [K, N]
            Z: [M, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
        partitioning:
            Z:
                M: [uniform_shape(12), uniform_shape(4)]
                K: [uniform_shape(5)]
    """

    hfa = "for m2, (z_m1, a_m1) in z_m2 << a_m2:\n" + \
          "    for m1, (z_m0, a_m0) in z_m1 << a_m1:\n" + \
          "        for m0, (z_n, a_k1) in z_m0 << a_m0:\n" + \
          "            for n, (z_ref, b_k1) in z_n << b_n:\n" + \
          "                for k1, (a_k0, b_k0) in a_k1 & b_k1:\n" + \
          "                    for k0, (a_val, b_val) in a_k0 & b_k0:\n" + \
          "                        z_ref += a_val * b_val"

    assert build_loop_nest(yaml) == hfa


def test_dynamic_partitioning():
    yaml = """
    einsum:
        declaration:
            A: [K, M]
            B: [K, N]
            Z: [M, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
        loop-order:
            Z: [N, K1, M, K0]
        partitioning:
            Z:
                K: [uniform_occupancy(A.6)]
    """

    hfa = "for n, (z_m, b_k) in z_n << b_n:\n" + \
          "    A_KM = Tensor.fromFiber(rank_ids=[\"K\", \"M\"], fiber=a_k)\n" + \
          "    tmp0 = A_KM\n" + \
          "    tmp1 = tmp0.splitEqual(6)\n" + \
          "    A_K1K0M = tmp1\n" + \
          "    A_K1K0M.setRankIds(rank_ids=[\"K1\", \"K0\", \"M\"])\n" + \
          "    A_K1MK0 = A_K1K0M.swizzleRanks(rank_ids=[\"K1\", \"M\", \"K0\"])\n" + \
          "    a_k1 = A_K1MK0.getRoot()\n" + \
          "    B_K = Tensor.fromFiber(rank_ids=[\"K\"], fiber=b_k)\n" + \
          "    tmp2 = B_K\n" + \
          "    tmp3 = tmp2.splitNonUniform(a_k1)\n" + \
          "    B_K1K0 = tmp3\n" + \
          "    B_K1K0.setRankIds(rank_ids=[\"K1\", \"K0\"])\n" + \
          "    b_k1 = B_K1K0.getRoot()\n" + \
          "    for k1, (a_m, b_k0) in a_k1 & b_k1:\n" + \
          "        for m, (z_ref, a_k0) in z_m << a_m:\n" + \
          "            for k0, (a_val, b_val) in a_k0 & b_k0:\n" + \
          "                z_ref += a_val * b_val"

    assert build_loop_nest(yaml) == hfa
