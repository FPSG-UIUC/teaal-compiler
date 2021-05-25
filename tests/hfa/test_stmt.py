from es2hfa.hfa.stmt import *
from es2hfa.hfa.expr import EVar


def test_sassign():
    assign = SAssign("x", EVar("y"))
    assert assign.gen() == "x = y"


def test_sblock():
    block = SBlock([SAssign("x", EVar("y")), SAssign("a", EVar("b"))])
    assert block.gen() == "x = y\na = b"
