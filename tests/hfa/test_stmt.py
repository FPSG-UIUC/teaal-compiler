from es2hfa.hfa.stmt import *

def test_sblock():
    block = SBlock(["a", "b", "c"])
    assert block.gen() == "a\nb\nc"
