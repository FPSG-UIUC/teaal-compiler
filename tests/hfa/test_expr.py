from es2hfa.hfa.expr import *

def test_evar():
    var = EVar("x")
    assert var.gen() == "x"
