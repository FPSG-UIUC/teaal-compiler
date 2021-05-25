from es2hfa.hfa.expr import EVar
from es2hfa.hfa.op import OAdd
from es2hfa.hfa.payload import PVar
from es2hfa.hfa.stmt import *


def test_sassign():
    assign = SAssign("x", EVar("y"))
    assert assign.gen(2) == "        x = y"


def test_sblock():
    block = SBlock([SAssign("x", EVar("y")), SAssign("a", EVar("b"))])
    assert block.gen(2) == "        x = y\n        a = b"


def test_sexpr():
    expr = SExpr(EVar("c"))
    assert expr.gen(2) == "        c"


def test_sfor():
    for_ = SFor("i", PVar("a_j"), EVar("a_i"), SExpr(EVar("a_j")))
    assert for_.gen(1) == "    for i, a_j in a_i:\n        a_j"


def test_siassign():
    iassign = SIAssign("i", OAdd(), EVar("j"))
    assert iassign.gen(2) == "        i += j"
