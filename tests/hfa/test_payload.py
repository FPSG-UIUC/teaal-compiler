from es2hfa.hfa import *


def test_ptuple_parens():
    tuple_ = PTuple([PVar("a"), PVar("b")])
    assert tuple_.gen(True) == "(a, b)"


def test_ptuple_no_parens_one_level():
    tuple_ = PTuple([PVar("a"), PVar("b")])
    assert tuple_.gen(False) == "a, b"


def test_ptuple_no_parens_mult_levels():
    tuple_ = PTuple([PTuple([PVar("a"), PVar("b")]),
                    PTuple([PVar("c"), PVar("d")])])
    assert tuple_.gen(False) == "(a, b), (c, d)"


def test_pvar_parens():
    var = PVar("a")
    assert var.gen(True) == "a"


def test_pvar_no_parens():
    var = PVar("a")
    assert var.gen(False) == "a"
