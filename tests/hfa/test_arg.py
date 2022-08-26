from teaal.hfa import *


def test_ajust():
    just = AJust(EVar("i"))
    assert just.gen() == "i"


def test_aparam():
    param = AParam("j", EVar("k"))
    assert param.gen() == "j=k"
