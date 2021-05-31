from es2hfa.parse.input import Input
from es2hfa.trans.translate import Translator


def test_translate_no_loops():
    input_ = Input("tests/integration/test_translate_no_loops.yml")
    hfa = "a_ref = A_.getRoot()\n" + \
          "a_ref += b"
    assert Translator.translate(input_).gen(depth=0) == hfa


def test_translate_defaults():
    input_ = Input("tests/integration/test_input_no_mapping.yml")
    hfa = "t1_m = T1_MN.getRoot()\n" + \
          "A_MK = A_KM.swizzleRanks([\"M\", \"K\"])\n" + \
          "a_m = A_MK.getRoot()\n" + \
          "B_NK = B_KN.swizzleRanks([\"N\", \"K\"])\n" + \
          "b_n = B_NK.getRoot()\n" + \
          "for m, (t1_n, a_k) in t1_m << a_m:\n" + \
          "    for n, (t1_ref, b_k) in t1_n << b_n:\n" + \
          "        for k, (a_val, b_val) in a_k & b_k:\n" + \
          "            t1_ref += a_val * b_val\n" + \
          "z_m = Z_MN.getRoot()\n" + \
          "t1_m = T1_MN.getRoot()\n" + \
          "c_m = C_MN.getRoot()\n" + \
          "for m, (z_n, (_, t1_n, c_n)) in z_m << (t1_m | c_m):\n" + \
          "    for n, (z_ref, (_, t1_val, c_val)) in z_n << (t1_n | c_n):\n" + \
          "        z_ref += t1_val + c_val"
    assert Translator.translate(input_).gen(depth=0) == hfa


def test_translate_specified():
    input_ = Input("tests/integration/test_input.yml")
    hfa = "T1_NM = T1_MN.swizzleRanks([\"N\", \"M\"])\n" + \
          "t1_n = T1_NM.getRoot()\n" + \
          "A_KM = A_MK.swizzleRanks([\"K\", \"M\"])\n" + \
          "a_k = A_KM.getRoot()\n" + \
          "b_k = B_KN.getRoot()\n" + \
          "for k, (a_m, b_n) in a_k & b_k:\n" + \
          "    for n, (t1_m, b_val) in t1_n << b_n:\n" + \
          "        for m, (t1_ref, a_val) in t1_m << a_m:\n" + \
          "            t1_ref += a_val * b_val\n" + \
          "z_n = Z_NM.getRoot()\n" + \
          "T1_NM = T1_MN.swizzleRanks([\"N\", \"M\"])\n" + \
          "t1_n = T1_NM.getRoot()\n" + \
          "c_n = C_NM.getRoot()\n" + \
          "for n, (z_m, (_, t1_m, c_m)) in z_n << (t1_n | c_n):\n" + \
          "    for m, (z_ref, (_, t1_val, c_val)) in z_m << (t1_m | c_m):\n" + \
          "        z_ref += t1_val + c_val"
    assert Translator.translate(input_).gen(depth=0) == hfa
