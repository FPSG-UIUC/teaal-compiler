from es2hfa.ir.mapping import Mapping
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.einsum import EinsumParser
from es2hfa.parse.tensor import TensorParser
from es2hfa.trans.footer import Footer
from tests.utils.parse_tree import make_uniform_shape


def assert_make_footer(loop_order, partitioning, hfa):
    tensors = ["A[I, J]", "B[I, K]", "C[J, K]"]
    tensors = [TensorParser.parse(tensor) for tensor in tensors]
    mapping = Mapping(tensors, {})

    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[j, k])")
    mapping.add_einsum(tree, loop_order, partitioning)

    for tensor in mapping.get_tensors():
        mapping.apply_partitioning(tensor)
        mapping.apply_loop_order(tensor)

    assert Footer.make_footer(mapping).gen(depth=0) == hfa

    return mapping


def test_make_footer_default():
    assert_make_footer({}, {}, "")


def test_output_still_output():
    mapping = assert_make_footer({}, {}, "")
    output = mapping.get_output()

    desired = Tensor.from_tree(TensorParser.parse("A[I, J]"))
    desired.set_is_output(True)

    assert output == desired


def test_make_footer_swizzle():
    loop_order = {"A": ["J", "I", "K"]}
    hfa = "A_IJ = A_JI.swizzleRanks(rank_ids=[\"I\", \"J\"])"
    assert_make_footer(loop_order, {}, hfa)


def test_make_footer_partitioned():
    part = {"A": {"I": make_uniform_shape(
        [5]), "J": make_uniform_shape([3, 4])}}
    hfa = "tmp = A_I1I0J2J1J0\n" + \
          "tmp = tmp.flattenRanks(depth=0, levels=1, coord_style=\"absolute\")\n" + \
          "tmp = tmp.flattenRanks(depth=1, levels=1, coord_style=\"absolute\")\n" + \
          "tmp = tmp.flattenRanks(depth=1, levels=1, coord_style=\"absolute\")\n" + \
          "A_IJ = tmp\n" + \
          "A_IJ.setRankIds(rank_ids=[\"I\", \"J\"])"
    assert_make_footer({}, part, hfa)


def test_make_footer_all():
    part = {"A": {"J": make_uniform_shape([6, 3])}}
    loop_order = {"A": ["J2", "J1", "I", "J0", "K"]}
    hfa = "A_IJ2J1J0 = A_J2J1IJ0.swizzleRanks(rank_ids=[\"I\", \"J2\", \"J1\", \"J0\"])\n" + \
          "tmp = A_IJ2J1J0\n" + \
          "tmp = tmp.flattenRanks(depth=1, levels=1, coord_style=\"absolute\")\n" + \
          "tmp = tmp.flattenRanks(depth=1, levels=1, coord_style=\"absolute\")\n" + \
          "A_IJ = tmp\n" + \
          "A_IJ.setRankIds(rank_ids=[\"I\", \"J\"])"
    assert_make_footer(loop_order, part, hfa)
