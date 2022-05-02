from es2hfa.hfa import *


def test_eaccess():
    access = EAccess(EVar("A"), EVar("i"))
    assert access.gen() == "A[i]"


def test_ebinop():
    binop = EBinOp(EVar("a"), OAdd(), EVar("b"))
    assert binop.gen() == "a + b"


def test_ebool():
    bool_ = EBool(True)
    assert bool_.gen() == "True"


def test_ecomp():
    comp = EComp(EVar("a"), "a", EFunc("range", [AJust(EInt(5))]))
    assert comp.gen() == "[a for a in range(5)]"


def test_edict():
    dict_ = EDict({EString("A"): EInt(5), EString("B"): EInt(10)})
    assert dict_.gen() == "{\"A\": 5, \"B\": 10}"


def test_efield():
    field = EField("foo", "bar")
    assert field.gen() == "foo.bar"


def test_efloat():
    float_ = EFloat(1.23)
    assert float_.gen() == "1.23"

    float_ = EFloat(float("inf"))
    assert float_.gen() == "float(\"inf\")"

    float_ = EFloat(-float("inf"))
    assert float_.gen() == "-float(\"inf\")"


def test_efunc():
    func = EFunc("foo", [AJust(EVar("x")), AJust(EVar("y"))])
    assert func.gen() == "foo(x, y)"


def test_eint():
    int_ = EInt(5)
    assert int_.gen() == "5"


def test_elambda():
    lambda_ = ELambda(["a", "b"], EBinOp(EVar("a"), OAdd(), EVar("b")))
    assert lambda_.gen() == "lambda a, b: a + b"


def test_elist():
    list_ = EList([EVar("a"), EVar("b")])
    assert list_.gen() == "[a, b]"


def test_emethod():
    method = EMethod(EVar("foo"), "bar", [AJust(EVar("x")), AJust(EVar("y"))])
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
