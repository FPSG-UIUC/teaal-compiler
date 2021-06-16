from es2hfa.parse.input import Input
from es2hfa.trans.translate import Translator


def read_hfa(filename):
    data = ""
    with open(filename, 'r') as stream:
        data = stream.read()
    return data


test_names = [
    'dotprod',
    'example',
    'example2',
    'example3',
    'example4',
    'example5',
    'gemm',
    'gemv',
    'gram',
    'mttkrp',
    'outerprod',
    'sddmm',
    'spmv',
    'ttm',
    'ttv']

# TODO: example6, example7, nrm_sq, spmm


def test_integration():
    for test_name in test_names:
        filename = 'tests/integration/' + test_name

        input_ = Input(filename + ".yml")
        output = Translator.translate(input_).gen(depth=0)

        hfa = read_hfa(filename + ".hfa")

        assert output == hfa, test_name + " integration test failed!"
