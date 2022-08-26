from teaal.parse.einsum import Einsum
from teaal.parse.mapping import Mapping
from teaal.trans.hifiber import HiFiber


def read_hifiber(filename):
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

        einsum = Einsum.from_file(filename + ".yaml")
        mapping = Mapping.from_file(filename + ".yaml")
        output = str(HiFiber(einsum, mapping))

        hifiber = read_hifiber(filename + ".py")
        assert output == hifiber, test_name + " integration test failed!"