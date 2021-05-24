from es2hfa.hfa import *

def test_sblock():
    block = SBlock(["a", "b", "c"])
    assert block.generate() == "a\nb\nc"
