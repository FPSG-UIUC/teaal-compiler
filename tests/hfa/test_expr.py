from es2hfa.hfa.expr import *
from es2hfa.hfa.op import OAdd
from es2hfa.hfa.arg import AJust


def test_ebinop():
    binop = EBinOp(EVar("a"), OAdd(), EVar("b"))
    assert binop.gen() == "a + b"


def test_efield():
    field = EField("foo", "bar")
    assert field.gen() == "foo.bar"


def test_efunc():
    func = EFunc("foo", [AJust(EVar("x")), AJust(EVar("y"))])
    assert func.gen() == "foo(x, y)"


def test_eint():
    int_ = EInt(5)
    assert int_.gen() == "5"


def test_elist():
    list_ = EList([EVar("a"), EVar("b")])
    assert list_.gen() == "[a, b]"


def test_emethod():
    method = EMethod("foo", "bar", [AJust(EVar("x")), AJust(EVar("y"))])
    assert method.gen() == "foo.bar(x, y)"


def test_eparens():
    parens = EParens(EVar("z"))
    assert parens.gen() == "(z)"


def test_estring():
    string = EString("y")
    assert string.gen() == "\"y\""


def test_etuple_one_elem():
    tuple_ = ETuple([EVar("x")])
    assert tuple_.gen() == "(x,)"


def test_etuple():
    tuple_ = ETuple([EVar("x"), EVar("y"), EVar("z")])
    assert tuple_.gen() == "(x, y, z)"


def test_evar():
    var = EVar("x")
    assert var.gen() == "x"
