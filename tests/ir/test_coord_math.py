from lark.lexer import Token
from lark.tree import Tree
import pytest
from sympy import symbols

from teaal.ir.coord_math import CoordMath
from teaal.ir.partitioning import Partitioning
from teaal.ir.tensor import Tensor
from teaal.parse.equation import EquationParser
from teaal.parse.mapping import Mapping
from tests.utils.parse_tree import *


def parse_partitioning():
    yaml = """
    mapping:
        partitioning:
            O:
                Q: [uniform_shape(10)]
                W: [follow(Q)]
    """
    return Mapping.from_str(yaml).get_partitioning()["O"]


def test_add_bad_expr():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    ranks = Tree("ranks", [0, 1, 2])

    with pytest.raises(ValueError) as excinfo:
        coord_math.add(tensor, ranks)
    assert str(excinfo.value) == "Unknown coord tree: 0"


def test_add_bad_term():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    ranks = Tree("ranks", [Tree("iplus", [0, 1, 2])])

    with pytest.raises(ValueError) as excinfo:
        coord_math.add(tensor, ranks)
    assert str(excinfo.value) == "Unknown coord term: 0"


def test_add_bad_factor():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    ranks = Tree("ranks",
                 [Tree("iplus", [Tree("itimes", [Tree("foo", [])])])])

    with pytest.raises(ValueError) as excinfo:
        coord_math.add(tensor, ranks)
    assert str(excinfo.value) == "Unknown coord factor: Tree('foo', [])"


def test_add_bad_itimes():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    ranks = Tree("ranks",
                 [Tree("iplus", [Tree("itimes", [Token("NUMBER", 5)])])])

    with pytest.raises(ValueError) as excinfo:
        coord_math.add(tensor, ranks)
    assert str(excinfo.value) == \
        "Unknown coord term: Tree('itimes', [Token('NUMBER', 5)])"


def test_add_unknown_term():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    ranks = Tree("ranks", [Tree("iplus", [Tree("bar", [Tree("foo", [])])])])

    with pytest.raises(ValueError) as excinfo:
        coord_math.add(tensor, ranks)
    assert str(excinfo.value) == "Unknown coord term: bar"


def test_add():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    ranks = make_ranks(["w"])
    coord_math.add(tensor, ranks)

    assert coord_math.get_all_exprs("w") == [symbols("w")]


def test_add_plus():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    ranks = Tree("ranks", [make_iplus(["q", "s"])])
    coord_math.add(tensor, ranks)

    q, s, w = symbols("q s w")

    assert coord_math.get_all_exprs("w") == [w, q + s]
    assert coord_math.get_all_exprs("q") == [q, w - s]
    assert coord_math.get_all_exprs("s") == [s, w - q]


def test_add_times():
    coord_math = CoordMath()
    ranks = Tree("ranks", [make_itimes(["2", "q"])])
    tensor = Tensor("I", ["W"])
    coord_math.add(tensor, ranks)

    q, w = symbols("q w")
    assert coord_math.get_all_exprs("w") == [w, 2 * q]
    assert coord_math.get_all_exprs("q") == [q, w / 2]


def test_get_all_exprs():
    coord_math = CoordMath()
    ranks = Tree("ranks", [make_iplus(["m"]), make_iplus(["k"])])
    tensor = Tensor("A", ["M", "K"])
    coord_math.add(tensor, ranks)

    m, k, mk = symbols("m k mk")

    assert coord_math.get_all_exprs("m") == [m]
    assert coord_math.get_all_exprs("k") == [k]
    assert coord_math.get_all_exprs("mk") == [mk]


def test_get_eqn_exprs():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    ranks = Tree("ranks", [make_iplus(["q", "s"])])
    coord_math.add(tensor, ranks)

    q, s, w = symbols("q s w")

    assert coord_math.get_eqn_exprs() == {w: q + s}


def test_get_trans_no_prune():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    ranks = make_ranks(["w"])
    coord_math.add(tensor, ranks)

    with pytest.raises(ValueError) as excinfo:
        coord_math.get_trans("w")
    assert str(excinfo.value) == "Unconfigured coord math. First call prune()"


def test_prune():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    ranks = Tree("ranks", [make_iplus(["q", "s"])])

    coord_math.add(tensor, ranks)
    coord_math.prune({"Q", "W"})

    q, s, w = symbols("q s w")

    assert coord_math.get_trans("w") == w
    assert coord_math.get_trans("q") == q
    assert coord_math.get_trans("s") == w - q


def test_prune_partitioned():
    coord_math = CoordMath()
    tensor = Tensor("I", ["W"])
    ranks = Tree("ranks", [make_iplus(["q", "s"])])
    coord_math.add(tensor, ranks)

    tensor = Tensor("O", ["Q"])
    ranks = make_ranks(["q"])
    coord_math.add(tensor, ranks)

    tensor = Tensor("F", ["S"])
    ranks = make_ranks(["s"])
    coord_math.add(tensor, ranks)

    coord_math.prune({"Q", "S"})

    q, s, w = symbols("q s w")

    assert coord_math.get_trans("w") == q + s
    assert coord_math.get_trans("q") == q
    assert coord_math.get_trans("s") == s


def test_eq_true():
    coord_math0 = CoordMath()
    coord_math1 = CoordMath()
    tensor = Tensor("I", ["W"])
    ranks = Tree("ranks", [make_iplus(["q", "s"])])

    coord_math0.add(tensor, ranks)
    coord_math0.prune({"W", "Q"})

    coord_math1.add(tensor, ranks)
    coord_math1.prune({"W", "Q"})

    assert coord_math0 == coord_math1


def test_eq_false():
    coord_math0 = CoordMath()
    coord_math1 = CoordMath()

    tensor = Tensor("I", ["W"])
    ranks = Tree("ranks", [make_iplus(["q", "s"])])

    coord_math0.add(tensor, ranks)
    coord_math0.prune({"W", "Q"})

    coord_math1.add(tensor, ranks)

    assert coord_math0 != coord_math1


def test_eq_not_im():
    coord_math = CoordMath()

    assert coord_math != "foo"
