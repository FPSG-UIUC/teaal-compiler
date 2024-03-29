import pytest

from teaal.ir.program import Program
from teaal.ir.tensor import Tensor
from teaal.parse.einsum import Einsum
from teaal.parse.mapping import Mapping
from teaal.trans.partitioner import Partitioner
from teaal.trans.utils import TransUtils


def assert_partition(tensor, parts, rank, hifiber):
    program, partitioner = build_partitioner(parts)
    assert partitioner.partition(tensor, (rank,)).gen(depth=0) == hifiber


def build_partitioner(parts):
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
        expressions:
            - Z[m, n] = A[k, m] * B[k, n]
    mapping:
        partitioning:
            Z:
        """ + parts
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    partitioner = Partitioner(program, TransUtils(program))
    return program, partitioner


def build_partitioner_conv(expr, parts):
    yaml = """
    einsum:
        declaration:
            F: [S]
            I: [W]
            J: [W]
            K: [V]
            O: [Q]
        expressions:
            - """ + expr + """
    mapping:
        partitioning:
            O:
        """ + parts
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    partitioner = Partitioner(program, TransUtils(program))
    return program, partitioner


def build_partitioner_copy(parts):
    yaml = """
    einsum:
        declaration:
            A: [M, N, O, P, Q]
            Z: [M, N, O, P, Q]
        expressions:
            - Z[m, n, o, p, q] = A[m, n, o, p, q]
    mapping:
        partitioning:
            Z:
        """ + parts
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    partitioner = Partitioner(program, TransUtils(program))
    return program, partitioner


def build_partitioner_math_no_halo(parts):
    yaml = """
    einsum:
      declaration:
        A: [K]
        Z: [M]
      expressions:
        - Z[m] = A[2 * m]
    mapping:
        partitioning:
            Z:
    """ + parts

    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    partitioner = Partitioner(program, TransUtils(program))
    return program, partitioner


def test_no_partitioning():
    tensor = Tensor("B", ["K", "N"])
    _, partitioner = build_partitioner("")
    assert partitioner.partition(tensor, ("N",)).gen(0) == ""


def test_flatten_unswizzled():
    spec = """
                (O, N, P): [flatten()]
    """
    program, partitioner = build_partitioner_copy(spec)

    with pytest.raises(ValueError) as excinfo:
        partitioner.partition(
            program.get_equation().get_tensor("A"), ("O", "N", "P"))

    assert str(
        excinfo.value) == "Cannot flatten together ('O', 'N', 'P') on tensor with ranks ['M', 'N', 'O', 'P', 'Q']"


def test_flatten():
    spec = """
                (N, O, P): [flatten()]
    """
    program, partitioner = build_partitioner_copy(spec)
    hifiber = "tmp0 = A_MNOPQ\n" + \
        "tmp1 = tmp0.flattenRanks(depth=1, levels=2, coord_style=\"tuple\")\n" + \
        "A_MNOPQ_flat = tmp1\n" + \
        "A_MNOPQ_flat.setRankIds(rank_ids=[\"M\", \"NOP\", \"Q\"])"

    assert partitioner.partition(
        program.get_equation().get_tensor("A"), ("N", "O", "P")).gen(
        depth=0) == hifiber


def test_nway_shape():
    tensor = Tensor("B", ["K", "N"])
    spec = """
                N: [nway_shape(6), nway_shape(3)]
    """
    hifiber = "tmp0 = B_KN\n" + \
        "tmp1 = tmp0.splitUniform((N - 1) // 6 + 1, depth=1)\n" + \
        "tmp2 = tmp1.splitUniform((N - 1) // 3 + 1, depth=1)\n" + \
        "B_KN2N1N0 = tmp2\n" + \
        "B_KN2N1N0.setRankIds(rank_ids=[\"K\", \"N2\", \"N1\", \"N0\"])"

    assert_partition(tensor, spec, "N", hifiber)


def test_nway_shape_var():
    tensor = Tensor("B", ["K", "N"])
    spec = """
                N: [nway_shape(N1), nway_shape(N0)]
    """
    hifiber = "tmp0 = B_KN\n" + \
        "tmp1 = tmp0.splitUniform((N - 1) // N1 + 1, depth=1)\n" + \
        "tmp2 = tmp1.splitUniform((N - 1) // N0 + 1, depth=1)\n" + \
        "B_KN2N1N0 = tmp2\n" + \
        "B_KN2N1N0.setRankIds(rank_ids=[\"K\", \"N2\", \"N1\", \"N0\"])"

    assert_partition(tensor, spec, "N", hifiber)


def test_nway_shape_conv():
    tensor = Tensor("I", ["W"])
    expr = "O[q] = I[q + s] * F[s]"
    spec = """
                Q: [nway_shape(6), nway_shape(3)]
                W: [follow(Q)]
    """
    hifiber = "tmp0 = I_W\n" + \
        "tmp1 = tmp0.splitUniform((Q - 1) // 6 + 1, depth=0, post_halo=-1 + S)\n" + \
        "tmp2 = tmp1.splitUniform((Q - 1) // 3 + 1, depth=0)\n" + \
        "I_W2W1W0 = tmp2\n" + \
        "I_W2W1W0.setRankIds(rank_ids=[\"W2\", \"W1\", \"W0\"])"

    _, partitioner = build_partitioner_conv(expr, spec)
    assert partitioner.partition(tensor, ("W",)).gen(depth=0) == hifiber


def test_uniform_occupancy_leader():
    tensor = Tensor("A", ["K", "M"])
    spec = """
                K: [uniform_occupancy(A.5)]
    """
    hifiber = "tmp0 = A_KM\n" + \
        "tmp1 = tmp0.splitEqual(5)\n" + \
        "A_K1K0M = tmp1\n" + \
        "A_K1K0M.setRankIds(rank_ids=[\"K1\", \"K0\", \"M\"])"

    assert_partition(tensor, spec, "K", hifiber)


def test_uniform_occupancy_leader_var():
    tensor = Tensor("A", ["K", "M"])
    spec = """
                K: [uniform_occupancy(A.K0)]
    """
    hifiber = "tmp0 = A_KM\n" + \
        "tmp1 = tmp0.splitEqual(K0)\n" + \
        "A_K1K0M = tmp1\n" + \
        "A_K1K0M.setRankIds(rank_ids=[\"K1\", \"K0\", \"M\"])"

    assert_partition(tensor, spec, "K", hifiber)


def test_uniform_occupancy_follower():
    tensor = Tensor("B", ["K", "N"])
    spec = """
                K: [uniform_occupancy(A.5)]
    """
    hifiber = "tmp0 = B_KN\n" + \
        "tmp1 = tmp0.splitNonUniform(a_k1)\n" + \
        "B_K1K0N = tmp1\n" + \
        "B_K1K0N.setRankIds(rank_ids=[\"K1\", \"K0\", \"N\"])"

    program, partitioner = build_partitioner(spec)
    program.apply_partitioning(program.get_equation().get_tensor("A"), ("K",))
    assert partitioner.partition(tensor, ("K",)).gen(depth=0) == hifiber


def test_uniform_occupancy_multiple():
    spec = """
                K: [uniform_occupancy(A.6), uniform_occupancy(A.3)]
        loop-order:
            Z: [K2, K1, K0, M, N]
    """
    hifiber = "tmp0 = A_KM\n" + \
        "tmp1 = tmp0.splitEqual(6)\n" + \
        "A_K2K1IM = tmp1\n" + \
        "A_K2K1IM.setRankIds(rank_ids=[\"K2\", \"K1I\", \"M\"])"

    program, partitioner = build_partitioner(spec)
    tensor = program.get_equation().get_tensor("A")
    tensor.from_fiber()
    assert partitioner.partition(tensor, ("K",)).gen(depth=0) == hifiber

    hifiber = "tmp2 = A_K1IM\n" + \
        "tmp3 = tmp2.splitEqual(3)\n" + \
        "A_K1K0M = tmp3\n" + \
        "A_K1K0M.setRankIds(rank_ids=[\"K1\", \"K0\", \"M\"])"

    tensor.pop()
    tensor.from_fiber()
    assert partitioner.partition(tensor, ("K1I",)).gen(depth=0) == hifiber


def test_uniform_occupancy_pre_halo():
    tensor = Tensor("I", ["W"])
    expr = "O[q] = I[q + -1 * s] * F[s]"
    spec = """
                Q: [uniform_occupancy(I.6)]
                W: [follow(Q)]
    """
    hifiber = "tmp0 = I_W\n" + \
        "tmp1 = tmp0.splitEqual(6, pre_halo=-1 + S)\n" + \
        "I_W1W0 = tmp1\n" + \
        "I_W1W0.setRankIds(rank_ids=[\"W1\", \"W0\"])"

    _, partitioner = build_partitioner_conv(expr, spec)
    assert partitioner.partition(tensor, ("W",)).gen(depth=0) == hifiber


def test_uniform_occupancy_post_halo():
    tensor = Tensor("I", ["W"])
    expr = "O[q] = I[q + s] * F[s]"
    spec = """
                Q: [uniform_occupancy(I.6)]
                W: [follow(Q)]
    """
    hifiber = "tmp0 = I_W\n" + \
        "tmp1 = tmp0.splitEqual(6, post_halo=-1 + S)\n" + \
        "I_W1W0 = tmp1\n" + \
        "I_W1W0.setRankIds(rank_ids=[\"W1\", \"W0\"])"

    _, partitioner = build_partitioner_conv(expr, spec)
    assert partitioner.partition(tensor, ("W",)).gen(depth=0) == hifiber


def test_uniform_occupancy_pre_post_halo():
    tensor = Tensor("I", ["W"])
    expr = "O[q] = I[q + s + -2 * v] * F[s] * K[v]"
    spec = """
                Q: [uniform_occupancy(I.6)]
                W: [follow(Q)]
    """
    hifiber = "tmp0 = I_W\n" + \
        "tmp1 = tmp0.splitEqual(6, pre_halo=-2 + 2 * V, post_halo=-1 + S)\n" + \
        "I_W1W0 = tmp1\n" + \
        "I_W1W0.setRankIds(rank_ids=[\"W1\", \"W0\"])"

    _, partitioner = build_partitioner_conv(expr, spec)
    assert partitioner.partition(tensor, ("W",)).gen(depth=0) == hifiber


def test_uniform_occupancy_follower_pre_halo():
    tensor = Tensor("J", ["W"])
    expr = "O[q] = I[q + -1 * s] * J[q + -1 * s] * F[s]"
    spec = """
                Q: [uniform_occupancy(I.6)]
                W: [follow(Q)]
    """
    # TODO: What if the coordinates need to be translated?
    hifiber = "tmp0 = J_W\n" + \
        "tmp1 = tmp0.splitNonUniform(i_w1, pre_halo=-1 + S)\n" + \
        "J_W1W0 = tmp1\n" + \
        "J_W1W0.setRankIds(rank_ids=[\"W1\", \"W0\"])"

    program, partitioner = build_partitioner_conv(expr, spec)
    program.apply_partitioning(program.get_equation().get_tensor("I"), ("W",))
    assert partitioner.partition(tensor, ("W",)).gen(depth=0) == hifiber


def test_uniform_occupancy_follower_post_halo():
    tensor = Tensor("J", ["W"])
    expr = "O[q] = I[q + s] * J[q + s] * F[s]"
    spec = """
                Q: [uniform_occupancy(I.6)]
                W: [follow(Q)]
    """
    # TODO: What if the coordinates need to be translated?
    hifiber = "tmp0 = J_W\n" + \
        "tmp1 = tmp0.splitNonUniform(i_w1, post_halo=-1 + S)\n" + \
        "J_W1W0 = tmp1\n" + \
        "J_W1W0.setRankIds(rank_ids=[\"W1\", \"W0\"])"

    program, partitioner = build_partitioner_conv(expr, spec)
    program.apply_partitioning(program.get_equation().get_tensor("I"), ("W",))
    assert partitioner.partition(tensor, ("W",)).gen(depth=0) == hifiber


def test_uniform_occupancy_follower_pre_post_halo():
    tensor = Tensor("J", ["W"])
    expr = "O[q] = I[q + s + -2 * v] * J[q + s + -2 * v] * F[s] * K[v]"
    spec = """
                Q: [uniform_occupancy(I.6)]
                W: [follow(Q)]
    """
    # TODO: What if the coordinates need to be translated?
    hifiber = "tmp0 = J_W\n" + \
        "tmp1 = tmp0.splitNonUniform(i_w1, pre_halo=-2 + 2 * V, post_halo=-1 + S)\n" + \
        "J_W1W0 = tmp1\n" + \
        "J_W1W0.setRankIds(rank_ids=[\"W1\", \"W0\"])"

    program, partitioner = build_partitioner_conv(expr, spec)
    program.apply_partitioning(program.get_equation().get_tensor("I"), ("W",))
    assert partitioner.partition(tensor, ("W",)).gen(depth=0) == hifiber


def test_uniform_occupancy_multiple_levels_with_halo():
    expr = "O[q] = I[2 * q + s] * F[s]"
    spec = """
                Q: [uniform_occupancy(I.10), uniform_occupancy(I.5)]
                W: [follow(Q)]
    """

    program, partitioner = build_partitioner_conv(expr, spec)
    tensor = program.get_equation().get_tensor("I")

    # [W] -> [W2, W1I] -> [W1I]
    program.apply_partitioning(tensor, ("W",))
    tensor.pop()
    tensor.from_fiber()

    hifiber = "tmp0 = I_W1I\n" + \
        "tmp1 = tmp0.splitEqual(5, post_halo=-1 + S)\n" + \
        "I_W1W0 = tmp1\n" + \
        "I_W1W0.setRankIds(rank_ids=[\"W1\", \"W0\"])"

    assert partitioner.partition(tensor, ("W1I",)).gen(depth=0) == hifiber


def test_uniform_occupancy_follower_translate_follow():
    tensor = Tensor("K", ["V"])
    expr = "O[q] = I[q + s] * K[2 * q + s] * F[s]"
    spec = """
                Q: [uniform_occupancy(I.6)]
                W: [follow(Q)]
                V: [follow(Q)]
    """
    program, partitioner = build_partitioner_conv(expr, spec)
    program.apply_partitioning(program.get_equation().get_tensor("I"), ("W",))

    with pytest.raises(ValueError) as excinfo:
        partitioner.partition(tensor, ("V",)).gen(depth=0)

    assert str(
        excinfo.value) == "Cannot partition rank V with a leader of a different rank (W1)"


def test_uniform_shape():
    tensor = Tensor("B", ["K", "N"])
    spec = """
                N: [uniform_shape(6), uniform_shape(3)]
    """
    hifiber = "tmp0 = B_KN\n" + \
        "tmp1 = tmp0.splitUniform(6, depth=1)\n" + \
        "tmp2 = tmp1.splitUniform(3, depth=2)\n" + \
        "B_KN2N1N0 = tmp2\n" + \
        "B_KN2N1N0.setRankIds(rank_ids=[\"K\", \"N2\", \"N1\", \"N0\"])"

    assert_partition(tensor, spec, "N", hifiber)


def test_uniform_shape_var():
    tensor = Tensor("B", ["K", "N"])
    spec = """
                N: [uniform_shape(N1), uniform_shape(N0)]
    """
    hifiber = "tmp0 = B_KN\n" + \
        "tmp1 = tmp0.splitUniform(N1, depth=1)\n" + \
        "tmp2 = tmp1.splitUniform(N0, depth=2)\n" + \
        "B_KN2N1N0 = tmp2\n" + \
        "B_KN2N1N0.setRankIds(rank_ids=[\"K\", \"N2\", \"N1\", \"N0\"])"

    assert_partition(tensor, spec, "N", hifiber)


def test_uniform_shape_conv_pre_halo():
    tensor = Tensor("I", ["W"])
    expr = "O[q] = I[q + -1 * s] * F[s]"
    spec = """
                Q: [uniform_shape(6)]
                W: [follow(Q)]
    """
    hifiber = "tmp0 = I_W\n" + \
        "tmp1 = tmp0.splitUniform(6, depth=0, pre_halo=-1 + S)\n" + \
        "I_W1W0 = tmp1\n" + \
        "I_W1W0.setRankIds(rank_ids=[\"W1\", \"W0\"])"

    _, partitioner = build_partitioner_conv(expr, spec)
    assert partitioner.partition(tensor, ("W",)).gen(depth=0) == hifiber


def test_uniform_shape_conv_post_halo():
    tensor = Tensor("I", ["W"])
    expr = "O[q] = I[q + s] * F[s]"
    spec = """
                Q: [uniform_shape(6)]
                W: [follow(Q)]
    """
    hifiber = "tmp0 = I_W\n" + \
        "tmp1 = tmp0.splitUniform(6, depth=0, post_halo=-1 + S)\n" + \
        "I_W1W0 = tmp1\n" + \
        "I_W1W0.setRankIds(rank_ids=[\"W1\", \"W0\"])"

    _, partitioner = build_partitioner_conv(expr, spec)
    assert partitioner.partition(tensor, ("W",)).gen(depth=0) == hifiber


def test_uniform_shape_conv_pre_post_halo():
    tensor = Tensor("I", ["W"])
    expr = "O[q] = I[q + s + -2 * v] * F[s] * K[v]"
    spec = """
                Q: [uniform_shape(6)]
                W: [follow(Q)]
    """
    hifiber = "tmp0 = I_W\n" + \
        "tmp1 = tmp0.splitUniform(6, depth=0, pre_halo=-2 + 2 * V, post_halo=-1 + S)\n" + \
        "I_W1W0 = tmp1\n" + \
        "I_W1W0.setRankIds(rank_ids=[\"W1\", \"W0\"])"

    _, partitioner = build_partitioner_conv(expr, spec)
    assert partitioner.partition(tensor, ("W",)).gen(depth=0) == hifiber


def test_uniform_shape_conv_double():
    tensor = Tensor("I", ["W"])
    expr = "O[q] = I[q + s + 2 * v] * F[s] * K[v]"
    spec = """
                Q: [uniform_shape(6)]
                W: [follow(Q)]
    """
    hifiber = "tmp0 = I_W\n" + \
        "tmp1 = tmp0.splitUniform(6, depth=0, post_halo=-3 + S + 2 * V)\n" + \
        "I_W1W0 = tmp1\n" + \
        "I_W1W0.setRankIds(rank_ids=[\"W1\", \"W0\"])"

    _, partitioner = build_partitioner_conv(expr, spec)
    assert partitioner.partition(tensor, ("W",)).gen(depth=0) == hifiber


def test_uniform_shape_math_no_halo():
    tensor = Tensor("A", ["K"])
    spec = """
                M: [uniform_shape(10)]
                K: [follow(M)]
    """
    hifiber = "tmp0 = A_K\n" + \
        "tmp1 = tmp0.splitUniform(2 * 10, depth=0)\n" + \
        "A_K1K0 = tmp1\n" + \
        "A_K1K0.setRankIds(rank_ids=[\"K1\", \"K0\"])"

    _, partitioner = build_partitioner_math_no_halo(spec)
    assert partitioner.partition(tensor, ("K",)).gen(depth=0) == hifiber


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
    hifiber = "tmp0 = A_KM\n" + \
        "tmp1 = tmp0.splitUniform(500, depth=0)\n" + \
        "tmp2 = tmp1.splitUniform(250, depth=1)\n" + \
        "A_K8K7K6IM = tmp2\n" + \
        "A_K8K7K6IM.setRankIds(rank_ids=[\"K8\", \"K7\", \"K6I\", \"M\"])"

    program, partitioner = build_partitioner(spec)
    tensor = program.get_equation().get_tensor("A")
    tensor.from_fiber()
    assert partitioner.partition(tensor, ("K",)).gen(depth=0) == hifiber

    hifiber = "tmp3 = A_K6IM\n" + \
        "tmp4 = tmp3.splitEqual(100)\n" + \
        "A_K6K5IM = tmp4\n" + \
        "A_K6K5IM.setRankIds(rank_ids=[\"K6\", \"K5I\", \"M\"])"

    tensor.pop()
    tensor.pop()
    tensor.from_fiber()
    assert partitioner.partition(tensor, ("K6I",)).gen(depth=0) == hifiber

    hifiber = "tmp5 = A_K5IM\n" + \
        "tmp6 = tmp5.splitEqual(50)\n" + \
        "tmp7 = tmp6.splitUniform(10, depth=1)\n" + \
        "A_K5K4K3IM = tmp7\n" + \
        "A_K5K4K3IM.setRankIds(rank_ids=[\"K5\", \"K4\", \"K3I\", \"M\"])"

    tensor.pop()
    tensor.from_fiber()
    assert partitioner.partition(tensor, ("K5I",)).gen(depth=0) == hifiber

    hifiber = "tmp8 = A_K3IM\n" + \
        "tmp9 = tmp8.splitEqual(6)\n" + \
        "tmp10 = tmp9.splitUniform(4, depth=1)\n" + \
        "tmp11 = tmp10.splitUniform(2, depth=2)\n" + \
        "A_K3K2K1K0M = tmp11\n" + \
        "A_K3K2K1K0M.setRankIds(rank_ids=[\"K3\", \"K2\", \"K1\", \"K0\", \"M\"])"

    tensor.pop()
    tensor.pop()
    tensor.from_fiber()
    assert partitioner.partition(tensor, ("K3I",)).gen(depth=0) == hifiber


def assert_unpartition(spec, hifiber_options):
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
        expressions:
            - Z[m, n] = A[k, m] * B[k, n]
    mapping:
        partitioning:
            Z:
        """ + spec
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    for tensor in program.get_equation().get_tensors():
        program.apply_all_partitioning(tensor)

    partitioner = Partitioner(program, TransUtils(program))
    hifiber = partitioner.unpartition(
        program.get_equation().get_output()).gen(0)

    assert hifiber in hifiber_options


def test_unpartition_none():
    spec = """
                K: [uniform_shape(5)]
    """
    hifiber = ""
    assert_unpartition(spec, [hifiber])


def test_unpartition_one():
    spec = """
                N: [uniform_shape(6), uniform_shape(3)]
    """
    hifiber = "tmp0 = Z_MN2N1N0\n" + \
        "tmp1 = tmp0.mergeRanks(depth=1, levels=2, coord_style=\"absolute\")\n" + \
        "tmp1.setRankIds(rank_ids=[\"M\", \"N\"])\n" + \
        "Z_MN = tmp1"
    assert_unpartition(spec, [hifiber])


def test_unpartition_all():
    spec = """
                M: [uniform_shape(5)]
                N: [uniform_shape(6), uniform_shape(3)]
    """
    hifiber_option1 = "tmp0 = Z_M1M0N2N1N0\n" + \
        "tmp1 = tmp0.mergeRanks(depth=2, levels=2, coord_style=\"absolute\")\n" + \
        "tmp2 = tmp1.mergeRanks(depth=0, levels=1, coord_style=\"absolute\")\n" + \
        "tmp2.setRankIds(rank_ids=[\"M\", \"N\"])\n" + \
        "Z_MN = tmp2"
    hifiber_option2 = "tmp0 = Z_M1M0N2N1N0\n" + \
        "tmp1 = tmp0.mergeRanks(depth=0, levels=1, coord_style=\"absolute\")\n" + \
        "tmp2 = tmp1.mergeRanks(depth=1, levels=2, coord_style=\"absolute\")\n" + \
        "tmp2.setRankIds(rank_ids=[\"M\", \"N\"])\n" + \
        "Z_MN = tmp2"
    assert_unpartition(spec, [hifiber_option1, hifiber_option2])


def test_unpartition_multiple_dyn():
    spec = """
                M: [uniform_occupancy(A.10), uniform_occupancy(A.5)]
    """

    hifiber = "tmp0 = Z_M2M1M0N\n" + \
        "tmp1 = tmp0.mergeRanks(depth=0, levels=2, coord_style=\"absolute\")\n" + \
        "tmp1.setRankIds(rank_ids=[\"M\", \"N\"])\n" + \
        "Z_MN = tmp1"

    assert_unpartition(spec, [hifiber])


def test_unpartition_flatten():
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [M, N]
        expressions:
            - Z[m, n] = A[m, n]
    mapping:
        partitioning:
            Z:
                M: [uniform_shape(10)]
                (N, M0): [flatten()]
                NM0: [uniform_occupancy(A.5)]
    """
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    program.apply_all_partitioning(program.get_equation().get_output())

    partitioner = Partitioner(program, TransUtils(program))
    hifiber = partitioner.unpartition(
        program.get_equation().get_output()).gen(0)
    corr = "tmp0 = Z_M1NM01NM00\n" + \
        "tmp1 = tmp0.mergeRanks(depth=1, levels=1, coord_style=\"absolute\")\n" + \
        "tmp2 = tmp1.unflattenRanks(depth=1, levels=1)\n" + \
        "tmp2.setRankIds(rank_ids=[\"M1\", \"N\", \"M0\"])\n" + \
        "tmp3 = tmp2.swizzleRanks(rank_ids=[\"M1\", \"M0\", \"N\"])\n" + \
        "tmp4 = tmp3.mergeRanks(depth=0, levels=1, coord_style=\"absolute\")\n" + \
        "tmp4.setRankIds(rank_ids=[\"M\", \"N\"])\n" + \
        "Z_MN = tmp4"
    assert hifiber == corr


def test_unswizzle_unpartition():
    spec = """
                M: [uniform_shape(5)]
                N: [uniform_shape(6), uniform_shape(3)]
        loop-order:
            Z: [M1, N2, K, N1, M0, N0]
    """
    program, partitioner = build_partitioner(spec)

    output = program.get_equation().get_tensor("Z")
    program.apply_all_partitioning(output)
    program.get_loop_order().apply(output)

    hifiber_option1 = "tmp0 = Z_M1N2N1M0N0\n" + \
        "tmp1 = tmp0.swizzleRanks(rank_ids=[\"M1\", \"M0\", \"N2\", \"N1\", \"N0\"])\n" + \
        "tmp2 = tmp1.mergeRanks(depth=0, levels=1, coord_style=\"absolute\")\n" + \
        "tmp3 = tmp2.mergeRanks(depth=1, levels=2, coord_style=\"absolute\")\n" + \
        "tmp3.setRankIds(rank_ids=[\"M\", \"N\"])\n" + \
        "Z_MN = tmp3"

    hifiber_option2 = "tmp0 = Z_M1N2N1M0N0\n" + \
        "tmp1 = tmp0.swizzleRanks(rank_ids=[\"M1\", \"M0\", \"N2\", \"N1\", \"N0\"])\n" + \
        "tmp2 = tmp1.mergeRanks(depth=2, levels=2, coord_style=\"absolute\")\n" + \
        "tmp3 = tmp2.mergeRanks(depth=0, levels=1, coord_style=\"absolute\")\n" + \
        "tmp3.setRankIds(rank_ids=[\"M\", \"N\"])\n" + \
        "Z_MN = tmp3"

    assert partitioner.unpartition(
        program.get_equation().get_output()).gen(0) in [
        hifiber_option1, hifiber_option2]
