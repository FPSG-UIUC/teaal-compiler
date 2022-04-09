from lark.lexer import Token
from lark.tree import Tree
import pytest
from sympy import symbols

from es2hfa.ir.index_math import IndexMath
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
    index_math = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [0, 1, 2])

    with pytest.raises(ValueError) as excinfo:
        index_math.add(tensor, tranks)
    assert str(excinfo.value) == "Unknown index tree: 0"


def test_add_bad_term():
    index_math = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [Tree("iplus", [0, 1, 2])])

    with pytest.raises(ValueError) as excinfo:
        index_math.add(tensor, tranks)
    assert str(excinfo.value) == "Unknown index term: 0"


def test_add_bad_factor():
    index_math = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks",
                  [Tree("iplus", [Tree("itimes", [Tree("foo", [])])])])

    with pytest.raises(ValueError) as excinfo:
        index_math.add(tensor, tranks)
    assert str(excinfo.value) == "Unknown index factor: Tree('foo', [])"


def test_add_bad_itimes():
    index_math = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks",
                  [Tree("iplus", [Tree("itimes", [Token("NUMBER", 5)])])])

    with pytest.raises(ValueError) as excinfo:
        index_math.add(tensor, tranks)
    assert str(excinfo.value) == \
        "Unknown index term: Tree('itimes', [Token('NUMBER', 5)])"


def test_add_unknown_term():
    index_math = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [Tree("iplus", [Tree("bar", [Tree("foo", [])])])])

    with pytest.raises(ValueError) as excinfo:
        index_math.add(tensor, tranks)
    assert str(excinfo.value) == "Unknown index term: bar"


def test_add():
    index_math = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = make_tranks(["w"])
    index_math.add(tensor, tranks)

    assert index_math.get_all_exprs("w") == [symbols("w")]


def test_add_plus():
    index_math = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [make_iplus(["q", "s"])])
    index_math.add(tensor, tranks)

    q, s, w = symbols("q s w")

    assert index_math.get_all_exprs("w") == [w, q + s]
    assert index_math.get_all_exprs("q") == [q, w - s]
    assert index_math.get_all_exprs("s") == [s, w - q]


def test_add_times():
    index_math = IndexMath()
    tranks = Tree("tranks", [make_itimes(["2", "q"])])
    tensor = Tensor("I", ["W"])
    index_math.add(tensor, tranks)

    q, w = symbols("q w")
    assert index_math.get_all_exprs("w") == [w, 2 * q]
    assert index_math.get_all_exprs("q") == [q, w / 2]


def test_get_trans_no_prune():
    index_math = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = make_tranks(["w"])
    index_math.add(tensor, tranks)

    with pytest.raises(ValueError) as excinfo:
        index_math.get_trans("w")
    assert str(excinfo.value) == "Unconfigured index math. First call prune()"


def test_prune():
    index_math = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [make_iplus(["q", "s"])])

    loop_order = ["W", "Q"]
    part = Partitioning({}, ["Q", "S", "W"])

    index_math.add(tensor, tranks)
    index_math.prune(loop_order, part)

    q, s, w = symbols("q s w")

    assert index_math.get_trans("w") == w
    assert index_math.get_trans("q") == q
    assert index_math.get_trans("s") == w - q


def test_prune_partitioned():
    index_math = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [make_iplus(["q", "s"])])

    loop_order = ["Q1", "S", "Q0"]
    part = Partitioning(parse_partitioning(), ["Q", "S", "W"])

    index_math.add(tensor, tranks)
    index_math.prune(loop_order, part)

    q, s, w = symbols("q s w")

    assert index_math.get_trans("w") == q + s
    assert index_math.get_trans("q") == q
    assert index_math.get_trans("s") == s


def test_eq_true():
    index_math0 = IndexMath()
    index_math1 = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [make_iplus(["q", "s"])])

    loop_order = ["W", "Q"]
    part = Partitioning({}, ["Q", "S", "W"])

    index_math0.add(tensor, tranks)
    index_math0.prune(loop_order, part)

    index_math1.add(tensor, tranks)
    index_math1.prune(loop_order, part)

    assert index_math0 == index_math1


def test_eq_false():
    index_math0 = IndexMath()
    index_math1 = IndexMath()

    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [make_iplus(["q", "s"])])
    loop_order = ["W", "Q"]
    part = Partitioning({}, ["Q", "S", "W"])

    index_math0.add(tensor, tranks)
    index_math0.prune(loop_order, part)

    index_math1.add(tensor, tranks)

    assert index_math0 != index_math1


def test_eq_not_im():
    index_math = IndexMath()

    assert index_math != "foo"
