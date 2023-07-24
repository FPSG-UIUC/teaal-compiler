from teaal.hifiber import *


class HiFiberTestClass(Base):
    def __init__(self, a, b):
        self.a = a
        self.b = b


def test_eq():
    tf1 = HiFiberTestClass(1, 2)
    tf2 = HiFiberTestClass(1, 2)

    assert tf1 == tf2

    tf3 = HiFiberTestClass(3, 2)

    assert tf1 != tf3

    tf4 = HiFiberTestClass(1, 3)

    assert tf1 != tf4

    assert tf1 != "foo"


def test_hash():
    tf1 = HiFiberTestClass(1, 2)
    tf2 = HiFiberTestClass(1, 2)

    assert hash(tf1) == hash(tf2)

    tf3 = HiFiberTestClass(3, 2)

    assert hash(tf1) != hash(tf3)

    tf4 = HiFiberTestClass(1, 3)

    assert hash(tf1) != hash(tf4)


def test_repr():
    tf = HiFiberTestClass(1, "c")
    assert repr(tf) == "(HiFiberTestClass, a=1, b=c)"
