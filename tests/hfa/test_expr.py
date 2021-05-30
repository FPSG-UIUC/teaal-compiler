from es2hfa.hfa.expr import *
from es2hfa.hfa.op import OAdd
from es2hfa.hfa.arg import AJust


def test_ebinop():
    binop = EBinOp(EVar("a"), OAdd(), EVar("b"))
    assert binop.gen() == "a + b"


def test_elist():
    list_ = EList([EVar("a"), EVar("b")])
    assert list_.gen() == "[a, b]"


def test_efunc():
    func = EFunc("foo", [AJust(EVar("x")), AJust(EVar("y"))])
    assert func.gen() == "foo(x, y)"


def test_emethod():
    method = EMethod("foo", "bar", [AJust(EVar("x")), AJust(EVar("y"))])
    assert method.gen() == "foo.bar(x, y)"


def test_eparens():
    parens = EParens(EVar("z"))
    assert parens.gen() == "(z)"


def test_estring():
    string = EString("y")
    assert string.gen() == "\"y\""


def test_evar():
    var = EVar("x")
    assert var.gen() == "x"
