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
    'example6',
    'example7',
    'gemm',
    'gemv',
    'gram',
    'mttkrp',
    'nrm_sq',
    'outerprod',
    'sddmm',
    'spmv',
    'spmm',
    'ttm',
    'ttv']


def test_integration():
    for test_name in test_names:
        filename = 'tests/integration/' + test_name

        input_ = Input.from_file(filename + ".yaml")
        output = Translator.translate(input_).gen(depth=0)

        hfa = read_hfa(filename + ".hfa")
        assert output == hfa, test_name + " integration test failed!"
