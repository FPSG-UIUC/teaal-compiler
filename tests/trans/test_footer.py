from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.einsum import Einsum
from es2hfa.parse.mapping import Mapping
from es2hfa.trans.graphics import Graphics
from es2hfa.trans.footer import Footer
from es2hfa.trans.partitioner import Partitioner
from es2hfa.trans.utils import TransUtils


def assert_make_footer(loop_order, partitioning, display, hfa):
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
        """ + partitioning + """
        loop-order:
            Z: """ + loop_order + """
        spacetime:
        """ + display
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    graphics = Graphics(program)
    graphics.make_header()

    # TODO: Are we sure this should be static partitioning?
    for ind in program.get_all_static_partitioning():
        program.start_partitioning(ind)

    for tensor in program.get_tensors():
        program.apply_all_partitioning(tensor)
        program.apply_loop_order(tensor)

    assert Footer.make_footer(
        program,
        graphics,
        Partitioner(program, TransUtils())).gen(
        depth=0) == hfa

    return program


def test_make_footer_default():
    assert_make_footer("", "", "", "")


def test_output_still_output():
    program = assert_make_footer("", "", "", "")
    output = program.get_output()

    desired = Tensor("Z", ["M", "N"])
    desired.set_is_output(True)

    assert output == desired


def test_make_footer_swizzle():
    loop_order = "[K, N, M]"
    hfa = "Z_MN = Z_NM.swizzleRanks(rank_ids=[\"M\", \"N\"])"
    assert_make_footer(loop_order, "", "", hfa)


def test_make_footer_partitioned():
    part = """
                M: [uniform_shape(5)]
                N: [uniform_shape(6), uniform_shape(3)]
    """
    hfa = "tmp0 = Z_M1M0N2N1N0\n" + \
          "tmp1 = tmp0.flattenRanks(depth=0, levels=1, coord_style=\"absolute\")\n" + \
          "tmp2 = tmp1.flattenRanks(depth=1, levels=2, coord_style=\"absolute\")\n" + \
          "Z_MN = tmp2\n" + \
          "Z_MN.setRankIds(rank_ids=[\"M\", \"N\"])"
    assert_make_footer("", part, "", hfa)


def test_make_footer_display():
    display = """
            Z:
                space: [N]
                time: [K.pos, M.coord]
    """
    hfa = "displayCanvas(canvas)"
    assert_make_footer("", "", display, hfa)


def test_make_footer_all():
    part = """
                N: [uniform_shape(6), uniform_shape(3)]
    """
    loop_order = "[N2, K, N1, M, N0]"
    display = """
            Z:
                space: [N2, N1]
                time: [K, M, N0]
    """
    hfa = "Z_MN2N1N0 = Z_N2N1MN0.swizzleRanks(rank_ids=[\"M\", \"N2\", \"N1\", \"N0\"])\n" + \
          "tmp0 = Z_MN2N1N0\n" + \
          "tmp1 = tmp0.flattenRanks(depth=1, levels=2, coord_style=\"absolute\")\n" + \
          "Z_MN = tmp1\n" + \
          "Z_MN.setRankIds(rank_ids=[\"M\", \"N\"])\n" + \
          "displayCanvas(canvas)"
    assert_make_footer(loop_order, part, display, hfa)
