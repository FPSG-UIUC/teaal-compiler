from teaal.hifiber import *


def test_aaccess():
    access = AAccess(EVar("A"), EVar("i"))
    assert access.gen() == "A[i]"


def test_afield():
    field = AField("foo", "bar")
    assert field.gen() == "foo.bar"


def test_avar():
    var = AVar("x")
    assert var.gen() == "x"
