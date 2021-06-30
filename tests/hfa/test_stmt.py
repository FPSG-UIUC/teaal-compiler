from es2hfa.hfa.expr import EBinOp, EVar
from es2hfa.hfa.op import OAdd
from es2hfa.hfa.payload import PVar
from es2hfa.hfa.stmt import *


def test_sassign():
    assign = SAssign("x", EVar("y"))
    assert assign.gen(2) == "        x = y"


def test_sblock():
    block = SBlock([SAssign("x", EVar("y")), SAssign("a", EVar("b"))])
    assert block.gen(2) == "        x = y\n        a = b"


def test_sblock_add_sblock():
    block1 = SBlock([SAssign("x", EVar("y")), SAssign("a", EVar("b"))])
    block2 = SBlock([SAssign("z", EVar("w")), SAssign("c", EVar("d"))])
    block1.add(block2)
    assert block1.gen(0) == "x = y\na = b\nz = w\nc = d"


def test_sblock_add_other():
    block = SBlock([SAssign("x", EVar("y")), SAssign("a", EVar("b"))])
    assign = SAssign("z", EVar("w"))
    block.add(assign)
    assert block.gen(0) == "x = y\na = b\nz = w"


def test_sexpr():
    expr = SExpr(EVar("c"))
    assert expr.gen(2) == "        c"


def test_sfor():
    for_ = SFor(PVar("a_j"), EVar("a_i"), SExpr(EVar("a_j")))
    assert for_.gen(1) == "    for a_j in a_i:\n        a_j"


def test_sfunc():
    stmt = SAssign("z", EBinOp(EVar("x"), OAdd(), EVar("y")))
    func = SFunc("foo", [EVar("x"), EVar("y")],
                 SBlock([stmt, SReturn(EVar("z"))]))
    assert func.gen(
        1) == "    def foo(x, y):\n        z = x + y\n        return z"


def test_siassign():
    iassign = SIAssign("i", OAdd(), EVar("j"))
    assert iassign.gen(2) == "        i += j"


def tst_sreturn():
    return_ = SReturn(EVar("foo"))
    assert return_.gen(2) == "        return foo"
