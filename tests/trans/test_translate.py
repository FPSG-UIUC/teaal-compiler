from es2hfa.parse.input import Input
from es2hfa.trans.translate import Translator


def test_translate_no_loops():
    input_ = Input("tests/integration/test_translate_no_loops.yml")
    hfa = "A_ = Tensor(rank_ids=[])\n" +\
          "a_ref = A_.getRoot()\n" + \
          "a_ref += b"
    assert Translator.translate(input_).gen(depth=0) == hfa


def test_translate_defaults():
    input_ = Input("tests/integration/test_input_no_mapping.yml")
    hfa = "T1_MN = Tensor(rank_ids=[\"M\", \"N\"])\n" + \
          "t1_m = T1_MN.getRoot()\n" + \
          "A_MK = A_KM.swizzleRanks(rank_ids=[\"M\", \"K\"])\n" + \
          "a_m = A_MK.getRoot()\n" + \
          "B_NK = B_KN.swizzleRanks(rank_ids=[\"N\", \"K\"])\n" + \
          "b_n = B_NK.getRoot()\n" + \
          "for m, (t1_n, a_k) in t1_m << a_m:\n" + \
          "    for n, (t1_ref, b_k) in t1_n << b_n:\n" + \
          "        for k, (a_val, b_val) in a_k & b_k:\n" + \
          "            t1_ref += a_val * b_val\n" + \
          "Z_MN = Tensor(rank_ids=[\"M\", \"N\"])\n" + \
          "z_m = Z_MN.getRoot()\n" + \
          "t1_m = T1_MN.getRoot()\n" + \
          "c_m = C_MN.getRoot()\n" + \
          "for m, (z_n, (_, t1_n, c_n)) in z_m << (t1_m | c_m):\n" + \
          "    for n, (z_ref, (_, t1_val, c_val)) in z_n << (t1_n | c_n):\n" + \
          "        z_ref += t1_val + c_val"
    assert Translator.translate(input_).gen(depth=0) == hfa


def test_translate_specified():
    input_ = Input("tests/integration/test_input.yml")

    hfa = "T1_MN = Tensor(rank_ids=[\"M\", \"N\"])\n" + \
          "T1_NM = T1_MN.swizzleRanks(rank_ids=[\"N\", \"M\"])\n" + \
          "t1_n = T1_NM.getRoot()\n" + \
          "A_KM = A_MK.swizzleRanks(rank_ids=[\"K\", \"M\"])\n" + \
          "a_k = A_KM.getRoot()\n" + \
          "b_k = B_KN.getRoot()\n" + \
          "for k, (a_m, b_n) in a_k & b_k:\n" + \
          "    for n, (t1_m, b_val) in t1_n << b_n:\n" + \
          "        for m, (t1_ref, a_val) in t1_m << a_m:\n" + \
          "            t1_ref += a_val * b_val\n" + \
          "Z_NM = Tensor(rank_ids=[\"N\", \"M\"])\n" + \
          "tmp = Z_NM\n" + \
          "tmp = tmp.splitUniform(4, depth=1)\n" + \
          "tmp = tmp.splitUniform(2, depth=2)\n" + \
          "tmp = tmp.splitUniform(6, depth=0)\n" + \
          "tmp = tmp.splitUniform(3, depth=1)\n" + \
          "Z_N2N1N0M2M1M0 = tmp\n" + \
          "z_n2 = Z_N2N1N0M2M1M0.getRoot()\n" + \
          "tmp = T1_MN\n" + \
          "tmp = tmp.splitUniform(6, depth=1)\n" + \
          "tmp = tmp.splitUniform(3, depth=2)\n" + \
          "tmp = tmp.splitUniform(4, depth=0)\n" + \
          "tmp = tmp.splitUniform(2, depth=1)\n" + \
          "T1_M2M1M0N2N1N0 = tmp\n" + \
          "T1_N2N1N0M2M1M0 = T1_M2M1M0N2N1N0.swizzleRanks(rank_ids=[\"N2\", \"N1\", \"N0\", \"M2\", \"M1\", \"M0\"])\n" + \
          "t1_n2 = T1_N2N1N0M2M1M0.getRoot()\n" + \
          "tmp = C_NM\n" + \
          "tmp = tmp.splitUniform(4, depth=1)\n" + \
          "tmp = tmp.splitUniform(2, depth=2)\n" + \
          "tmp = tmp.splitUniform(6, depth=0)\n" + \
          "tmp = tmp.splitUniform(3, depth=1)\n" + \
          "C_N2N1N0M2M1M0 = tmp\n" + \
          "c_n2 = C_N2N1N0M2M1M0.getRoot()\n" + \
          "for n2, (z_n1, (_, t1_n1, c_n1)) in z_n2 << (t1_n2 | c_n2):\n" + \
          "    for n1, (z_n0, (_, t1_n0, c_n0)) in z_n1 << (t1_n1 | c_n1):\n" + \
          "        for n0, (z_m2, (_, t1_m2, c_m2)) in z_n0 << (t1_n0 | c_n0):\n" + \
          "            for m2, (z_m1, (_, t1_m1, c_m1)) in z_m2 << (t1_m2 | c_m2):\n" + \
          "                for m1, (z_m0, (_, t1_m0, c_m0)) in z_m1 << (t1_m1 | c_m1):\n" + \
          "                    for m0, (z_ref, (_, t1_val, c_val)) in z_m0 << (t1_m0 | c_m0):\n" + \
          "                        z_ref += t1_val + c_val"

    assert Translator.translate(input_).gen(depth=0) == hfa
