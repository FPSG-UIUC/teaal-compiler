from lark.lexer import Token
from lark.tree import Tree
import pytest
from sympy import symbols

from es2hfa.ir.index_math import IndexMath
from es2hfa.ir.loop_order import LoopOrder
from es2hfa.ir.partitioning import Partitioning
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.equation import EquationParser
from tests.utils.parse_tree import *


def create_loop_order():
    equation = EquationParser.parse("O[q] = sum(S).(I[q + s] + F[s])")
    output = Tensor("O", ["Q"])
    output.set_is_output(True)
    order = ["W", "Q"]

    loop_order = LoopOrder(equation, output)
    loop_order.add(order, Partitioning({}, order))

    return loop_order


def test_get_all_exprs_no_tranks():
    im = IndexMath()

    with pytest.raises(ValueError) as excinfo:
        im.get_all_exprs("w")
    assert str(
        excinfo.value) == "Unconfigured index math. First call add_tranks()"


def test_add_tranks_bad_expr():
    im = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [0, 1, 2])

    with pytest.raises(ValueError) as excinfo:
        im.add_tranks(tensor, tranks)
    assert str(excinfo.value) == "Unknown index tree: 0"


def test_add_tranks_bad_term():
    im = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [Tree("iplus", [0, 1, 2])])

    with pytest.raises(ValueError) as excinfo:
        im.add_tranks(tensor, tranks)
    assert str(excinfo.value) == "Unknown index term: 0"


def test_add_tranks_bad_factor():
    im = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree(
        "tranks", [
            Tree(
                "iplus", [
                    Tree(
                        "itimes", [
                            Tree(
                                "foo", [])])])])

    with pytest.raises(ValueError) as excinfo:
        im.add_tranks(tensor, tranks)
    assert str(excinfo.value) == "Unknown index factor: Tree('foo', [])"


def test_add_tranks_bad_itimes():
    im = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree(
        "tranks", [
            Tree(
                "iplus", [
                    Tree(
                        "itimes", [
                            Token(
                                "NUMBER", 5)])])])

    with pytest.raises(ValueError) as excinfo:
        im.add_tranks(tensor, tranks)
    assert str(
        excinfo.value) == "Unknown index term: Tree('itimes', [Token('NUMBER', 5)])"


def test_add_tranks_unknown_term():
    im = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [Tree("iplus", [Tree("bar", [Tree("foo", [])])])])

    with pytest.raises(ValueError) as excinfo:
        im.add_tranks(tensor, tranks)
    assert str(excinfo.value) == "Unknown index term: bar"


def test_get_all_exprs_unconfigured():
    im = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = make_tranks(["w"])

    with pytest.raises(ValueError) as excinfo:
        im.get_all_exprs("w")
    assert str(
        excinfo.value) == "Unconfigured index math. First call add_tranks()"


def test_add_tranks():
    im = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = make_tranks(["w"])
    im.add_tranks(tensor, tranks)

    assert im.get_all_exprs("w") == [symbols("w")]


def test_add_tranks_plus():
    im = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [make_iplus(["q", "s"])])
    im.add_tranks(tensor, tranks)

    q, s, w = symbols("q s w")

    assert im.get_all_exprs("w") == [w, q + s]
    assert im.get_all_exprs("q") == [q, w - s]
    assert im.get_all_exprs("s") == [s, w - q]


def test_add_tranks_times():
    im = IndexMath()
    tranks = Tree("tranks", [make_itimes(["2", "q"])])
    tensor = Tensor("I", ["W"])
    im.add_tranks(tensor, tranks)

    q, w = symbols("q w")
    assert im.get_all_exprs("w") == [w, 2 * q]
    assert im.get_all_exprs("q") == [q, w / 2]


def test_get_trans_no_prune():
    im = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = make_tranks(["w"])
    im.add_tranks(tensor, tranks)

    with pytest.raises(ValueError) as excinfo:
        im.get_trans("w")
    assert str(excinfo.value) == "Unconfigured index math. First call prune()"


def test_get_all_exprs_no_tranks():
    im = IndexMath()
    lo = create_loop_order()

    with pytest.raises(ValueError) as excinfo:
        im.prune(lo)
    assert str(
        excinfo.value) == "Unconfigured index math. First call add_tranks()"


def test_prune():
    im = IndexMath()
    tensor = Tensor("I", ["W"])
    tranks = Tree("tranks", [make_iplus(["q", "s"])])
    lo = create_loop_order()

    im.add_tranks(tensor, tranks)
    im.prune(lo)

    q, s, w = symbols("q s w")

    assert im.get_trans("w") == w
    assert im.get_trans("q") == q
    assert im.get_trans("s") == w - q
