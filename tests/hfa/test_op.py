from es2hfa.hfa.op import *


def test_oadd():
    add = OAdd()
    assert add.gen() == "+"


def test_oand():
    and_ = OAnd()
    assert and_.gen() == "&"


def test_oltlt():
    ltlt = OLtLt()
    assert ltlt.gen() == "<<"


def test_omul():
    mul = OMul()
    assert mul.gen() == "*"


def test_oor():
    or_ = OOr()
    assert or_.gen() == "|"
