from lark.lexer import Token
from lark.tree import Tree
import pytest
from sympy import symbols

from es2hfa.ir.coord_math import CoordMath
from es2hfa.ir.partitioning import Partitioning
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.equation import EquationParser
from es2hfa.parse.mapping import Mapping
from tests.utils.parse_tree import *


def parse_partitioning():
    yaml = """
    mapping:
        partitioning:
            O:
                Q: [uniform_shape(10)]
    """
    return Mapping.from_str(yaml).get_partitioning()["O"]


def test_add_bad_expr():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [0, 1, 2])

    with pytest.raises(ValueError) as excinfo:
        coord_math.add(tensor, tranks)
    assert str(excinfo.value) == "Unknown coord tree: 0"


def test_add_bad_term():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [Tree("iplus", [0, 1, 2])])

    with pytest.raises(ValueError) as excinfo:
        coord_math.add(tensor, tranks)
    assert str(excinfo.value) == "Unknown coord term: 0"


def test_add_bad_factor():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks",
                  [Tree("iplus", [Tree("itimes", [Tree("foo", [])])])])

    with pytest.raises(ValueError) as excinfo:
        coord_math.add(tensor, tranks)
    assert str(excinfo.value) == "Unknown coord factor: Tree('foo', [])"


def test_add_bad_itimes():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks",
                  [Tree("iplus", [Tree("itimes", [Token("NUMBER", 5)])])])

    with pytest.raises(ValueError) as excinfo:
        coord_math.add(tensor, tranks)
    assert str(excinfo.value) == \
        "Unknown coord term: Tree('itimes', [Token('NUMBER', 5)])"


def test_add_unknown_term():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [Tree("iplus", [Tree("bar", [Tree("foo", [])])])])

    with pytest.raises(ValueError) as excinfo:
        coord_math.add(tensor, tranks)
    assert str(excinfo.value) == "Unknown coord term: bar"


def test_add():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    tranks = make_tranks(["w"])
    coord_math.add(tensor, tranks)

    assert coord_math.get_all_exprs("w") == [symbols("w")]


def test_add_plus():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [make_iplus(["q", "s"])])
    coord_math.add(tensor, tranks)

    q, s, w = symbols("q s w")

    assert coord_math.get_all_exprs("w") == [w, q + s]
    assert coord_math.get_all_exprs("q") == [q, w - s]
    assert coord_math.get_all_exprs("s") == [s, w - q]


def test_add_times():
    coord_math = CoordMath()
    tranks = Tree("tranks", [make_itimes(["2", "q"])])
    tensor = Tensor("I", ["W"])
    coord_math.add(tensor, tranks)

    q, w = symbols("q w")
    assert coord_math.get_all_exprs("w") == [w, 2 * q]
    assert coord_math.get_all_exprs("q") == [q, w / 2]


def test_get_eqn_exprs():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [make_iplus(["q", "s"])])
    coord_math.add(tensor, tranks)

    q, s, w = symbols("q s w")

    assert coord_math.get_eqn_exprs() == {w: q + s}


def test_get_trans_no_prune():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    tranks = make_tranks(["w"])
    coord_math.add(tensor, tranks)

    with pytest.raises(ValueError) as excinfo:
        coord_math.get_trans("w")
    assert str(excinfo.value) == "Unconfigured coord math. First call prune()"


def test_prune():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [make_iplus(["q", "s"])])

    loop_order = ["W", "Q"]
    part = Partitioning({}, ["Q", "S", "W"], coord_math.get_eqn_exprs())

    coord_math.add(tensor, tranks)
    coord_math.prune(loop_order, part)

    q, s, w = symbols("q s w")

    assert coord_math.get_trans("w") == w
    assert coord_math.get_trans("q") == q
    assert coord_math.get_trans("s") == w - q


def test_prune_partitioned():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [make_iplus(["q", "s"])])
    coord_math.add(tensor, tranks)

    tensor = Tensor("O", ["Q"])
    tranks = make_tranks(["q"])
    coord_math.add(tensor, tranks)

    tensor = Tensor("F", ["S"])
    tranks = make_tranks(["s"])
    coord_math.add(tensor, tranks)

    loop_order = ["Q1", "S", "Q0"]
    part = Partitioning(
        parse_partitioning(), [
            "Q", "S", "W"], coord_math.get_eqn_exprs())

    coord_math.prune(loop_order, part)

    q, s, w = symbols("q s w")

    assert coord_math.get_trans("w") == q + s
    assert coord_math.get_trans("q") == q
    assert coord_math.get_trans("s") == s


def test_eq_true():
    coord_math0 = CoordMath()
    coord_math1 = CoordMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [make_iplus(["q", "s"])])

    loop_order = ["W", "Q"]
    part0 = Partitioning({}, ["Q", "S", "W"], coord_math0.get_eqn_exprs())
    part1 = Partitioning({}, ["Q", "S", "W"], coord_math1.get_eqn_exprs())

    coord_math0.add(tensor, tranks)
    coord_math0.prune(loop_order, part0)

    coord_math1.add(tensor, tranks)
    coord_math1.prune(loop_order, part1)

    assert coord_math0 == coord_math1


def test_eq_false():
    coord_math0 = CoordMath()
    coord_math1 = CoordMath()

    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [make_iplus(["q", "s"])])
    loop_order = ["W", "Q"]
    part = Partitioning({}, ["Q", "S", "W"], coord_math0.get_eqn_exprs())

    coord_math0.add(tensor, tranks)
    coord_math0.prune(loop_order, part)

    coord_math1.add(tensor, tranks)

    assert coord_math0 != coord_math1


def test_eq_not_im():
    coord_math = CoordMath()

    assert coord_math != "foo"