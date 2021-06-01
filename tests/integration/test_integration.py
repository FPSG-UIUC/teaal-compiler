from es2hfa.parse.input import Input
from es2hfa.trans.translate import Translator


def read_hfa(filename):
    data = ""
    with open(filename, 'r') as stream:
        data = stream.read()
    return data


def test_integration():
    input_ = Input("tests/integration/example.yml")
    hfa = read_hfa("tests/integration/example.hfa")

    print(Translator.translate(input_).gen(depth=0))

    assert Translator.translate(input_).gen(depth=0) == hfa
