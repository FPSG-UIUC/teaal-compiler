from es2hfa.hfa.payload import *


def test_ptuple():
    tuple_ = PTuple([PVar("a"), PVar("b")])
    assert tuple_.gen() == "(a, b)"


def test_pvar():
    var = PVar("a")
    assert var.gen() == "a"
