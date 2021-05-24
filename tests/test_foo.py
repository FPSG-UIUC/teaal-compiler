from es2hfa.tester import foo, baz

def test_foo():
    assert foo() == "bar"

def test_baz():
    assert baz() == "nan"
