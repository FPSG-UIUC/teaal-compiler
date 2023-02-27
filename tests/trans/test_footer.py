from teaal.ir.program import Program
from teaal.ir.tensor import Tensor
from teaal.parse.einsum import Einsum
from teaal.parse.mapping import Mapping
from teaal.trans.graphics import Graphics
from teaal.trans.footer import Footer
from teaal.trans.partitioner import Partitioner
from teaal.trans.utils import TransUtils


def assert_make_footer(loop_order, partitioning, display, hifiber_options):
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

    for tensor in program.get_tensors():
        program.apply_all_partitioning(tensor)
        program.get_loop_order().apply(tensor)

    hifiber = Footer.make_footer(
        program, graphics, Partitioner(
            program, TransUtils())).gen(
        depth=0)
    assert hifiber in hifiber_options

    return program


def test_make_footer_default():
    assert_make_footer("", "", "", [""])


def test_output_still_output():
    program = assert_make_footer("", "", "", [""])
    output = program.get_output()

    desired = Tensor("Z", ["M", "N"])
    desired.set_is_output(True)

    assert output == desired


def test_make_footer_swizzle():
    loop_order = "[K, N, M]"
    hifiber = "tmp0 = Z_NM\n" + \
        "tmp1 = tmp0.swizzleRanks(rank_ids=[\"M\", \"N\"])\n" + \
        "Z_MN = tmp1"
    assert_make_footer(loop_order, "", "", [hifiber])


def test_make_footer_partitioned():
    part = """
                M: [uniform_shape(5)]
                N: [uniform_shape(6), uniform_shape(3)]
    """
    hifiber_option1 = "tmp0 = Z_M1M0N2N1N0\n" + \
        "tmp1 = tmp0.flattenRanks(depth=2, levels=2, coord_style=\"absolute\")\n" + \
        "tmp2 = tmp1.flattenRanks(depth=0, levels=1, coord_style=\"absolute\")\n" + \
        "tmp2.setRankIds(rank_ids=[\"M\", \"N\"])\n" + \
        "Z_MN = tmp2"
    hifiber_option2 = "tmp0 = Z_M1M0N2N1N0\n" + \
        "tmp1 = tmp0.flattenRanks(depth=0, levels=1, coord_style=\"absolute\")\n" + \
        "tmp2 = tmp1.flattenRanks(depth=1, levels=2, coord_style=\"absolute\")\n" + \
        "tmp2.setRankIds(rank_ids=[\"M\", \"N\"])\n" + \
        "Z_MN = tmp2"
    assert_make_footer("", part, "", [hifiber_option1, hifiber_option2])


def test_make_footer_display():
    display = """
            Z:
                space: [N]
                time: [K.pos, M.coord]
    """
    hifiber = "displayCanvas(canvas)"
    assert_make_footer("", "", display, [hifiber])


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
    hifiber = "tmp0 = Z_N2N1MN0\n" + \
        "tmp1 = tmp0.swizzleRanks(rank_ids=[\"M\", \"N2\", \"N1\", \"N0\"])\n" + \
        "tmp2 = tmp1.flattenRanks(depth=1, levels=2, coord_style=\"absolute\")\n" + \
        "tmp2.setRankIds(rank_ids=[\"M\", \"N\"])\n" + \
        "Z_MN = tmp2\n" + \
        "displayCanvas(canvas)"
    assert_make_footer(loop_order, part, display, [hifiber])
