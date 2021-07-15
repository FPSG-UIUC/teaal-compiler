from es2hfa.hfa import *


def test_oadd():
    add = OAdd()
    assert add.gen() == "+"


def test_oand():
    and_ = OAnd()
    assert and_.gen() == "&"


def test_ofdiv():
    fdiv = OFDiv()
    assert fdiv.gen() == "//"


def test_oin():
    in_ = OIn()
    assert in_.gen() == "in"


def test_oltlt():
    ltlt = OLtLt()
    assert ltlt.gen() == "<<"


def test_omul():
    mul = OMul()
    assert mul.gen() == "*"


def test_oor():
    or_ = OOr()
    assert or_.gen() == "|"


def test_osub():
    sub = OSub()
    assert sub.gen() == "-"
