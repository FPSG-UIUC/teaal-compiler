from es2hfa.hfa.op import *


def test_add():
    add = OAdd()
    assert add.gen() == "+"


def test_and():
    and_ = OAnd()
    assert and_.gen() == "&"


def test_ltlt():
    ltlt = OLtLt()
    assert ltlt.gen() == "<<"


def test_mul():
    mul = OMul()
    assert mul.gen() == "*"


def test_or():
    or_ = OOr()
    assert or_.gen() == "|"


def test_sub():
    sub = OSub()
    assert sub.gen() == "-"
