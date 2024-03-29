from teaal.hifiber import *


def test_oadd():
    add = OAdd()
    assert add.gen() == "+"


def test_oand():
    and_ = OAnd()
    assert and_.gen() == "&"


def test_odiv():
    div = ODiv()
    assert div.gen() == "/"


def test_oeqeq():
    eqeq = OEqEq()
    assert eqeq.gen() == "=="


def test_ofdiv():
    fdiv = OFDiv()
    assert fdiv.gen() == "//"


def test_oin():
    in_ = OIn()
    assert in_.gen() == "in"


def test_olt():
    lt = OLt()
    assert lt.gen() == "<"


def test_oltlt():
    ltlt = OLtLt()
    assert ltlt.gen() == "<<"


def test_omod():
    mod = OMod()
    assert mod.gen() == "%"


def test_omul():
    mul = OMul()
    assert mul.gen() == "*"


def test_onotin():
    notin = ONotIn()
    assert notin.gen() == "not in"


def test_oor():
    or_ = OOr()
    assert or_.gen() == "|"


def test_osub():
    sub = OSub()
    assert sub.gen() == "-"
