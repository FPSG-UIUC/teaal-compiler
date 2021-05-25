from es2hfa.hfa.arg import *
from es2hfa.hfa.expr import EVar


def test_ajust():
    just = AJust(EVar("i"))
    assert just.gen() == "i"


def test_aparam():
    param = AParam("j", EVar("k"))
    assert param.gen() == "j=k"
