from es2hfa.ir.mapping import Mapping
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.input import Input
from es2hfa.trans.canvas import Canvas
from es2hfa.trans.footer import Footer


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
        display:
        """ + display
    mapping = Mapping(Input.from_str(yaml))
    mapping.add_einsum(0)

    canvas = Canvas(mapping)
    canvas.create_canvas()

    for tensor in mapping.get_tensors():
        mapping.apply_partitioning(tensor)
        mapping.apply_loop_order(tensor)

    assert Footer.make_footer(mapping, canvas).gen(depth=0) == hfa

    return mapping


def test_make_footer_default():
    assert_make_footer("", "", "", "")


def test_output_still_output():
    mapping = assert_make_footer("", "", "", "")
    output = mapping.get_output()

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
    hfa = "tmp = Z_M1M0N2N1N0\n" + \
          "tmp = tmp.flattenRanks(depth=0, levels=1, coord_style=\"absolute\")\n" + \
          "tmp = tmp.flattenRanks(depth=1, levels=2, coord_style=\"absolute\")\n" + \
          "Z_MN = tmp\n" + \
          "Z_MN.setRankIds(rank_ids=[\"M\", \"N\"])"
    assert_make_footer("", part, "", hfa)


def test_make_footer_display():
    display = """
            Z:
                space: [N]
                time: [K, M]
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
          "tmp = Z_MN2N1N0\n" + \
          "tmp = tmp.flattenRanks(depth=1, levels=2, coord_style=\"absolute\")\n" + \
          "Z_MN = tmp\n" + \
          "Z_MN.setRankIds(rank_ids=[\"M\", \"N\"])\n" + \
          "displayCanvas(canvas)"
    assert_make_footer(loop_order, part, display, hfa)
