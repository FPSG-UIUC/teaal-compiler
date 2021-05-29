from es2hfa.parse.einsum import EinsumParser
from es2hfa.trans.translate import Translator


def test_translate_no_loops():
    tree = EinsumParser.parse("A[] = b")
    hfa = "a_ref += b"
    assert Translator.translate(tree, None).gen(depth=0) == hfa


def test_translate_default_order():
    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[k, j])")
    hfa = "for i, (a_j, b_k) in a_i << b_i:\n" + \
          "    for j, (a_ref, c_k) in a_j << c_j:\n" + \
          "        for k, (b_val, c_val) in b_k & c_k:\n" + \
          "            a_ref += b_val * c_val"
    assert Translator.translate(tree, None).gen(depth=0) == hfa


def test_translate_loop_order():
    tree = EinsumParser.parse("A[i, j] = sum(K).(B[i, k] * C[k, j])")
    hfa = "for j, (a_i, c_k) in a_j << c_j:\n" + \
          "    for k, (b_i, c_val) in b_k & c_k:\n" + \
          "        for i, (a_ref, b_val) in a_i << b_i:\n" + \
          "            a_ref += b_val * c_val"
    assert Translator.translate(tree, ["j", "k", "i"]).gen(depth=0) == hfa
