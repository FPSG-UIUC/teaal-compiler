from es2hfa.parse.input import Input
from es2hfa.trans.translate import Translator


def read_hfa(filename):
    data = ""
    with open(filename, 'r') as stream:
        data = stream.read()
    return data


test_names = ['example', 'gemm']


def test_integration():
    for i in range(len(test_names)):
        filename = 'tests/integration/' + test_names[i]
        input_ = Input(filename + ".yml")
        hfa = read_hfa(filename + ".hfa")
        output = Translator.translate(input_).gen(
            depth=0)
        assert output == hfa, test_names[i] + " :integration test failed!\n" + \
            "output:\n" + output + "\n" + "expected:\n" + hfa
