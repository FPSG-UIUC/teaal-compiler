from teaal.parse import *
from teaal.trans.hifiber import HiFiber


def test_translate_no_loops():
    einsum = Einsum.from_file("tests/integration/test_translate_no_loops.yaml")
    mapping = Mapping.from_file(
        "tests/integration/test_translate_no_loops.yaml")
    hifiber = "A_ = Tensor(rank_ids=[])\n" + \
        "a_ref = A_.getRoot()\n" + \
        "a_ref += b"
    assert str(HiFiber(einsum, mapping)) == hifiber


def test_translate_defaults():
    einsum = Einsum.from_file("tests/integration/test_input_no_mapping.yaml")
    mapping = Mapping.from_file(
        "tests/integration/test_input_no_mapping.yaml")
    hifiber = "T1_MN = Tensor(rank_ids=[\"M\", \"N\"])\n" + \
        "A_MK = A_KM.swizzleRanks(rank_ids=[\"M\", \"K\"])\n" + \
        "B_NK = B_KN.swizzleRanks(rank_ids=[\"N\", \"K\"])\n" + \
        "t1_m = T1_MN.getRoot()\n" + \
        "a_m = A_MK.getRoot()\n" + \
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

    assert str(HiFiber(einsum, mapping)) == hifiber


def test_translate_specified():
    einsum = Einsum.from_file("tests/integration/test_input.yaml")
    mapping = Mapping.from_file("tests/integration/test_input.yaml")

    hifiber_option1 = "T1_NM = Tensor(rank_ids=[\"N\", \"M\"])\n" + \
        "A_KM = A_MK.swizzleRanks(rank_ids=[\"K\", \"M\"])\n" + \
        "canvas = createCanvas(A_KM, B_KN, T1_NM)\n" + \
        "t1_n = T1_NM.getRoot()\n" + \
        "a_k = A_KM.getRoot()\n" + \
        "b_k = B_KN.getRoot()\n" + \
        "for k_pos, (k, (a_m, b_n)) in enumerate(a_k & b_k):\n" + \
        "    for n_pos, (n, (t1_m, b_val)) in enumerate(t1_n << b_n):\n" + \
        "        for m, (t1_ref, a_val) in t1_m << a_m:\n" + \
        "            t1_ref += a_val * b_val\n" + \
        "            canvas.addActivity((k, m), (k, n), (n, m), spacetime=((n_pos,), (k_pos, m)))\n" + \
        "tmp0 = T1_NM\n" + \
        "tmp1 = tmp0.swizzleRanks(rank_ids=[\"M\", \"N\"])\n" + \
        "T1_MN = tmp1\n" + \
        "displayCanvas(canvas)\n" + \
        "Z_M2N2M1N1M0N0 = Tensor(rank_ids=[\"M2\", \"N2\", \"M1\", \"N1\", \"M0\", \"N0\"])\n" + \
        "tmp2 = T1_MN\n" + \
        "tmp3 = tmp2.splitUniform(6, depth=1)\n" + \
        "tmp4 = tmp3.splitUniform(3, depth=2)\n" + \
        "T1_MN2N1N0 = tmp4\n" + \
        "T1_MN2N1N0.setRankIds(rank_ids=[\"M\", \"N2\", \"N1\", \"N0\"])\n" + \
        "tmp5 = T1_MN2N1N0\n" + \
        "tmp6 = tmp5.splitUniform(4, depth=0)\n" + \
        "tmp7 = tmp6.splitUniform(2, depth=1)\n" + \
        "T1_M2M1M0N2N1N0 = tmp7\n" + \
        "T1_M2M1M0N2N1N0.setRankIds(rank_ids=[\"M2\", \"M1\", \"M0\", \"N2\", \"N1\", \"N0\"])\n" + \
        "tmp8 = C_NM\n" + \
        "tmp9 = tmp8.splitUniform(6, depth=0)\n" + \
        "tmp10 = tmp9.splitUniform(3, depth=1)\n" + \
        "C_N2N1N0M = tmp10\n" + \
        "C_N2N1N0M.setRankIds(rank_ids=[\"N2\", \"N1\", \"N0\", \"M\"])\n" + \
        "tmp11 = C_N2N1N0M\n" + \
        "tmp12 = tmp11.splitUniform(4, depth=3)\n" + \
        "tmp13 = tmp12.splitUniform(2, depth=4)\n" + \
        "C_N2N1N0M2M1M0 = tmp13\n" + \
        "C_N2N1N0M2M1M0.setRankIds(rank_ids=[\"N2\", \"N1\", \"N0\", \"M2\", \"M1\", \"M0\"])\n" + \
        "z_m2 = Z_M2N2M1N1M0N0.getRoot()\n" + \
        "T1_M2N2M1N1M0N0 = T1_M2M1M0N2N1N0.swizzleRanks(rank_ids=[\"M2\", \"N2\", \"M1\", \"N1\", \"M0\", \"N0\"])\n" + \
        "C_M2N2M1N1M0N0 = C_N2N1N0M2M1M0.swizzleRanks(rank_ids=[\"M2\", \"N2\", \"M1\", \"N1\", \"M0\", \"N0\"])\n" + \
        "t1_m2 = T1_M2N2M1N1M0N0.getRoot()\n" + \
        "c_m2 = C_M2N2M1N1M0N0.getRoot()\n" + \
        "for m2, (z_n2, (_, t1_n2, c_n2)) in z_m2 << (t1_m2 | c_m2):\n" + \
        "    for n2, (z_m1, (_, t1_m1, c_m1)) in z_n2 << (t1_n2 | c_n2):\n" + \
        "        for m1, (z_n1, (_, t1_n1, c_n1)) in z_m1 << (t1_m1 | c_m1):\n" + \
        "            for n1, (z_m0, (_, t1_m0, c_m0)) in z_n1 << (t1_n1 | c_n1):\n" + \
        "                for m0, (z_n0, (_, t1_n0, c_n0)) in z_m0 << (t1_m0 | c_m0):\n" + \
        "                    for n0, (z_ref, (_, t1_val, c_val)) in z_n0 << (t1_n0 | c_n0):\n" + \
        "                        z_ref += t1_val + c_val\n" + \
        "tmp14 = Z_M2N2M1N1M0N0\n" + \
        "tmp15 = tmp14.swizzleRanks(rank_ids=[\"N2\", \"N1\", \"N0\", \"M2\", \"M1\", \"M0\"])\n" + \
        "tmp16 = tmp15.mergeRanks(depth=3, levels=2, coord_style=\"absolute\")\n" + \
        "tmp17 = tmp16.mergeRanks(depth=0, levels=2, coord_style=\"absolute\")\n" + \
        "tmp17.setRankIds(rank_ids=[\"N\", \"M\"])\n" + \
        "Z_NM = tmp17"

    hifiber_option2 = "T1_NM = Tensor(rank_ids=[\"N\", \"M\"])\n" + \
        "A_KM = A_MK.swizzleRanks(rank_ids=[\"K\", \"M\"])\n" + \
        "canvas = createCanvas(A_KM, B_KN, T1_NM)\n" + \
        "t1_n = T1_NM.getRoot()\n" + \
        "a_k = A_KM.getRoot()\n" + \
        "b_k = B_KN.getRoot()\n" + \
        "for k_pos, (k, (a_m, b_n)) in enumerate(a_k & b_k):\n" + \
        "    for n_pos, (n, (t1_m, b_val)) in enumerate(t1_n << b_n):\n" + \
        "        for m, (t1_ref, a_val) in t1_m << a_m:\n" + \
        "            t1_ref += a_val * b_val\n" + \
        "            canvas.addActivity((k, m), (k, n), (n, m), spacetime=((n_pos,), (k_pos, m)))\n" + \
        "tmp0 = T1_NM\n" + \
        "tmp1 = tmp0.swizzleRanks(rank_ids=[\"M\", \"N\"])\n" + \
        "T1_MN = tmp1\n" + \
        "displayCanvas(canvas)\n" + \
        "Z_M2N2M1N1M0N0 = Tensor(rank_ids=[\"M2\", \"N2\", \"M1\", \"N1\", \"M0\", \"N0\"])\n" + \
        "tmp2 = T1_MN\n" + \
        "tmp3 = tmp2.splitUniform(4, depth=0)\n" + \
        "tmp4 = tmp3.splitUniform(2, depth=1)\n" + \
        "T1_M2M1M0N = tmp4\n" + \
        "T1_M2M1M0N.setRankIds(rank_ids=[\"M2\", \"M1\", \"M0\", \"N\"])\n" + \
        "tmp5 = T1_M2M1M0N\n" + \
        "tmp6 = tmp5.splitUniform(6, depth=3)\n" + \
        "tmp7 = tmp6.splitUniform(3, depth=4)\n" + \
        "T1_M2M1M0N2N1N0 = tmp7\n" + \
        "T1_M2M1M0N2N1N0.setRankIds(rank_ids=[\"M2\", \"M1\", \"M0\", \"N2\", \"N1\", \"N0\"])\n" + \
        "tmp8 = C_NM\n" + \
        "tmp9 = tmp8.splitUniform(4, depth=1)\n" + \
        "tmp10 = tmp9.splitUniform(2, depth=2)\n" + \
        "C_NM2M1M0 = tmp10\n" + \
        "C_NM2M1M0.setRankIds(rank_ids=[\"N\", \"M2\", \"M1\", \"M0\"])\n" + \
        "tmp11 = C_NM2M1M0\n" + \
        "tmp12 = tmp11.splitUniform(6, depth=0)\n" + \
        "tmp13 = tmp12.splitUniform(3, depth=1)\n" + \
        "C_N2N1N0M2M1M0 = tmp13\n" + \
        "C_N2N1N0M2M1M0.setRankIds(rank_ids=[\"N2\", \"N1\", \"N0\", \"M2\", \"M1\", \"M0\"])\n" + \
        "z_m2 = Z_M2N2M1N1M0N0.getRoot()\n" + \
        "T1_M2N2M1N1M0N0 = T1_M2M1M0N2N1N0.swizzleRanks(rank_ids=[\"M2\", \"N2\", \"M1\", \"N1\", \"M0\", \"N0\"])\n" + \
        "C_M2N2M1N1M0N0 = C_N2N1N0M2M1M0.swizzleRanks(rank_ids=[\"M2\", \"N2\", \"M1\", \"N1\", \"M0\", \"N0\"])\n" + \
        "t1_m2 = T1_M2N2M1N1M0N0.getRoot()\n" + \
        "c_m2 = C_M2N2M1N1M0N0.getRoot()\n" + \
        "for m2, (z_n2, (_, t1_n2, c_n2)) in z_m2 << (t1_m2 | c_m2):\n" + \
        "    for n2, (z_m1, (_, t1_m1, c_m1)) in z_n2 << (t1_n2 | c_n2):\n" + \
        "        for m1, (z_n1, (_, t1_n1, c_n1)) in z_m1 << (t1_m1 | c_m1):\n" + \
        "            for n1, (z_m0, (_, t1_m0, c_m0)) in z_n1 << (t1_n1 | c_n1):\n" + \
        "                for m0, (z_n0, (_, t1_n0, c_n0)) in z_m0 << (t1_m0 | c_m0):\n" + \
        "                    for n0, (z_ref, (_, t1_val, c_val)) in z_n0 << (t1_n0 | c_n0):\n" + \
        "                        z_ref += t1_val + c_val\n" + \
        "tmp14 = Z_M2N2M1N1M0N0\n" + \
        "tmp15 = tmp14.swizzleRanks(rank_ids=[\"N2\", \"N1\", \"N0\", \"M2\", \"M1\", \"M0\"])\n" + \
        "tmp16 = tmp15.mergeRanks(depth=0, levels=2, coord_style=\"absolute\")\n" + \
        "tmp17 = tmp16.mergeRanks(depth=1, levels=2, coord_style=\"absolute\")\n" + \
        "tmp17.setRankIds(rank_ids=[\"N\", \"M\"])\n" + \
        "Z_NM = tmp17"

    hifiber_option3 = "T1_NM = Tensor(rank_ids=[\"N\", \"M\"])\n" + \
        "A_KM = A_MK.swizzleRanks(rank_ids=[\"K\", \"M\"])\n" + \
        "t1_n = T1_NM.getRoot()\n" + \
        "a_k = A_KM.getRoot()\n" + \
        "b_k = B_KN.getRoot()\n" + \
        "canvas = createCanvas(A_KM, B_KN, T1_NM)\n" + \
        "for k_pos, (k, (a_m, b_n)) in enumerate(a_k & b_k):\n" + \
        "    for n_pos, (n, (t1_m, b_val)) in enumerate(t1_n << b_n):\n" + \
        "        for m, (t1_ref, a_val) in t1_m << a_m:\n" + \
        "            t1_ref += a_val * b_val\n" + \
        "            canvas.addActivity((k, m), (k, n), (n, m), spacetime=((n_pos,), (k_pos, m)))\n" + \
        "tmp0 = T1_NM\n" + \
        "tmp1 = tmp0.swizzleRanks(rank_ids=[\"M\", \"N\"])\n" + \
        "T1_MN = tmp1\n" + \
        "displayCanvas(canvas)\n" + \
        "Z_M2N2M1N1M0N0 = Tensor(rank_ids=[\"M2\", \"N2\", \"M1\", \"N1\", \"M0\", \"N0\"])\n" + \
        "tmp2 = T1_MN\n" + \
        "tmp3 = tmp2.splitUniform(4, depth=0)\n" + \
        "tmp4 = tmp3.splitUniform(2, depth=1)\n" + \
        "T1_M2M1M0N = tmp4\n" + \
        "T1_M2M1M0N.setRankIds(rank_ids=[\"M2\", \"M1\", \"M0\", \"N\"])\n" + \
        "tmp5 = T1_M2M1M0N\n" + \
        "tmp6 = tmp5.splitUniform(6, depth=3)\n" + \
        "tmp7 = tmp6.splitUniform(3, depth=4)\n" + \
        "T1_M2M1M0N2N1N0 = tmp7\n" + \
        "T1_M2M1M0N2N1N0.setRankIds(rank_ids=[\"M2\", \"M1\", \"M0\", \"N2\", \"N1\", \"N0\"])\n" + \
        "tmp8 = C_NM\n" + \
        "tmp9 = tmp8.splitUniform(4, depth=1)\n" + \
        "tmp10 = tmp9.splitUniform(2, depth=2)\n" + \
        "C_NM2M1M0 = tmp10\n" + \
        "C_NM2M1M0.setRankIds(rank_ids=[\"N\", \"M2\", \"M1\", \"M0\"])\n" + \
        "tmp11 = C_NM2M1M0\n" + \
        "tmp12 = tmp11.splitUniform(6, depth=0)\n" + \
        "tmp13 = tmp12.splitUniform(3, depth=1)\n" + \
        "C_N2N1N0M2M1M0 = tmp13\n" + \
        "C_N2N1N0M2M1M0.setRankIds(rank_ids=[\"N2\", \"N1\", \"N0\", \"M2\", \"M1\", \"M0\"])\n" + \
        "z_m2 = Z_M2N2M1N1M0N0.getRoot()\n" + \
        "T1_M2N2M1N1M0N0 = T1_M2M1M0N2N1N0.swizzleRanks(rank_ids=[\"M2\", \"N2\", \"M1\", \"N1\", \"M0\", \"N0\"])\n" + \
        "C_M2N2M1N1M0N0 = C_N2N1N0M2M1M0.swizzleRanks(rank_ids=[\"M2\", \"N2\", \"M1\", \"N1\", \"M0\", \"N0\"])\n" + \
        "t1_m2 = T1_M2N2M1N1M0N0.getRoot()\n" + \
        "c_m2 = C_M2N2M1N1M0N0.getRoot()\n" + \
        "for m2, (z_n2, (_, t1_n2, c_n2)) in z_m2 << (t1_m2 | c_m2):\n" + \
        "    for n2, (z_m1, (_, t1_m1, c_m1)) in z_n2 << (t1_n2 | c_n2):\n" + \
        "        for m1, (z_n1, (_, t1_n1, c_n1)) in z_m1 << (t1_m1 | c_m1):\n" + \
        "            for n1, (z_m0, (_, t1_m0, c_m0)) in z_n1 << (t1_n1 | c_n1):\n" + \
        "                for m0, (z_n0, (_, t1_n0, c_n0)) in z_m0 << (t1_m0 | c_m0):\n" + \
        "                    for n0, (z_ref, (_, t1_val, c_val)) in z_n0 << (t1_n0 | c_n0):\n" + \
        "                        z_ref += t1_val + c_val\n" + \
        "tmp14 = Z_M2N2M1N1M0N0\n" + \
        "tmp15 = tmp14.swizzleRanks(rank_ids=[\"N2\", \"N1\", \"N0\", \"M2\", \"M1\", \"M0\"])\n" + \
        "tmp16 = tmp15.mergeRanks(depth=0, levels=2, coord_style=\"absolute\")\n" + \
        "tmp17 = tmp16.mergeRanks(depth=1, levels=2, coord_style=\"absolute\")\n" + \
        "tmp17.setRankIds(rank_ids=[\"N\", \"M\"])\n" + \
        "Z_NM = tmp17"

    hifiber_option4 = "T1_NM = Tensor(rank_ids=[\"N\", \"M\"])\n" + \
        "A_KM = A_MK.swizzleRanks(rank_ids=[\"K\", \"M\"])\n" + \
        "t1_n = T1_NM.getRoot()\n" + \
        "a_k = A_KM.getRoot()\n" + \
        "b_k = B_KN.getRoot()\n" + \
        "canvas = createCanvas(A_KM, B_KN, T1_NM)\n" + \
        "for k_pos, (k, (a_m, b_n)) in enumerate(a_k & b_k):\n" + \
        "    for n_pos, (n, (t1_m, b_val)) in enumerate(t1_n << b_n):\n" + \
        "        for m, (t1_ref, a_val) in t1_m << a_m:\n" + \
        "            t1_ref += a_val * b_val\n" + \
        "            canvas.addActivity((k, m), (k, n), (n, m), spacetime=((n_pos,), (k_pos, m)))\n" + \
        "tmp0 = T1_NM\n" + \
        "tmp1 = tmp0.swizzleRanks(rank_ids=[\"M\", \"N\"])\n" + \
        "T1_MN = tmp1\n" + \
        "displayCanvas(canvas)\n" + \
        "Z_M2N2M1N1M0N0 = Tensor(rank_ids=[\"M2\", \"N2\", \"M1\", \"N1\", \"M0\", \"N0\"])\n" + \
        "tmp2 = T1_MN\n" + \
        "tmp3 = tmp2.splitUniform(6, depth=1)\n" + \
        "tmp4 = tmp3.splitUniform(3, depth=2)\n" + \
        "T1_MN2N1N0 = tmp4\n" + \
        "T1_MN2N1N0.setRankIds(rank_ids=[\"M\", \"N2\", \"N1\", \"N0\"])\n" + \
        "tmp5 = T1_MN2N1N0\n" + \
        "tmp6 = tmp5.splitUniform(4, depth=0)\n" + \
        "tmp7 = tmp6.splitUniform(2, depth=1)\n" + \
        "T1_M2M1M0N2N1N0 = tmp7\n" + \
        "T1_M2M1M0N2N1N0.setRankIds(rank_ids=[\"M2\", \"M1\", \"M0\", \"N2\", \"N1\", \"N0\"])\n" + \
        "tmp8 = C_NM\n" + \
        "tmp9 = tmp8.splitUniform(6, depth=0)\n" + \
        "tmp10 = tmp9.splitUniform(3, depth=1)\n" + \
        "C_N2N1N0M = tmp10\n" + \
        "C_N2N1N0M.setRankIds(rank_ids=[\"N2\", \"N1\", \"N0\", \"M\"])\n" + \
        "tmp11 = C_N2N1N0M\n" + \
        "tmp12 = tmp11.splitUniform(4, depth=3)\n" + \
        "tmp13 = tmp12.splitUniform(2, depth=4)\n" + \
        "C_N2N1N0M2M1M0 = tmp13\n" + \
        "C_N2N1N0M2M1M0.setRankIds(rank_ids=[\"N2\", \"N1\", \"N0\", \"M2\", \"M1\", \"M0\"])\n" + \
        "z_m2 = Z_M2N2M1N1M0N0.getRoot()\n" + \
        "T1_M2N2M1N1M0N0 = T1_M2M1M0N2N1N0.swizzleRanks(rank_ids=[\"M2\", \"N2\", \"M1\", \"N1\", \"M0\", \"N0\"])\n" + \
        "C_M2N2M1N1M0N0 = C_N2N1N0M2M1M0.swizzleRanks(rank_ids=[\"M2\", \"N2\", \"M1\", \"N1\", \"M0\", \"N0\"])\n" + \
        "t1_m2 = T1_M2N2M1N1M0N0.getRoot()\n" + \
        "c_m2 = C_M2N2M1N1M0N0.getRoot()\n" + \
        "for m2, (z_n2, (_, t1_n2, c_n2)) in z_m2 << (t1_m2 | c_m2):\n" + \
        "    for n2, (z_m1, (_, t1_m1, c_m1)) in z_n2 << (t1_n2 | c_n2):\n" + \
        "        for m1, (z_n1, (_, t1_n1, c_n1)) in z_m1 << (t1_m1 | c_m1):\n" + \
        "            for n1, (z_m0, (_, t1_m0, c_m0)) in z_n1 << (t1_n1 | c_n1):\n" + \
        "                for m0, (z_n0, (_, t1_n0, c_n0)) in z_m0 << (t1_m0 | c_m0):\n" + \
        "                    for n0, (z_ref, (_, t1_val, c_val)) in z_n0 << (t1_n0 | c_n0):\n" + \
        "                        z_ref += t1_val + c_val\n" + \
        "tmp14 = Z_M2N2M1N1M0N0\n" + \
        "tmp15 = tmp14.swizzleRanks(rank_ids=[\"N2\", \"N1\", \"N0\", \"M2\", \"M1\", \"M0\"])\n" + \
        "tmp16 = tmp15.mergeRanks(depth=3, levels=2, coord_style=\"absolute\")\n" + \
        "tmp17 = tmp16.mergeRanks(depth=0, levels=2, coord_style=\"absolute\")\n" + \
        "tmp17.setRankIds(rank_ids=[\"N\", \"M\"])\n" + \
        "Z_NM = tmp17"

    result = str(HiFiber(einsum, mapping))
    assert result in [
        hifiber_option1,
        hifiber_option2,
        hifiber_option3,
        hifiber_option4]


def test_hifiber_dyn_part():
    yaml = """
    einsum:
        declaration:
            A: [K, M]
            B: [K, N]
            Z: [M, N]
        expressions:
            - Z[m, n] = A[k, m] * B[k, n]
    mapping:
        partitioning:
            Z:
                K: [uniform_occupancy(A.6), uniform_occupancy(A.3)]
                N: [uniform_occupancy(B.5)]
        loop-order:
            Z: [K2, M, N1, K1, N0, K0]
    """

    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)

    hifiber = "Z_MN1N0 = Tensor(rank_ids=[\"M\", \"N1\", \"N0\"])\n" + \
        "z_m = Z_MN1N0.getRoot()\n" + \
        "a_k = A_KM.getRoot()\n" + \
        "b_k = B_KN.getRoot()\n" + \
        "A_KM = Tensor.fromFiber(rank_ids=[\"K\", \"M\"], fiber=a_k)\n" + \
        "B_KN = Tensor.fromFiber(rank_ids=[\"K\", \"N\"], fiber=b_k)\n" + \
        "tmp0 = A_KM\n" + \
        "tmp1 = tmp0.splitEqual(6)\n" + \
        "A_K2K1IM = tmp1\n" + \
        "A_K2K1IM.setRankIds(rank_ids=[\"K2\", \"K1I\", \"M\"])\n" + \
        "A_K2MK1I = A_K2K1IM.swizzleRanks(rank_ids=[\"K2\", \"M\", \"K1I\"])\n" + \
        "a_k2 = A_K2MK1I.getRoot()\n" + \
        "tmp2 = B_KN\n" + \
        "tmp3 = tmp2.splitNonUniform(a_k2)\n" + \
        "B_K2K1IN = tmp3\n" + \
        "B_K2K1IN.setRankIds(rank_ids=[\"K2\", \"K1I\", \"N\"])\n" + \
        "B_K2NK1I = B_K2K1IN.swizzleRanks(rank_ids=[\"K2\", \"N\", \"K1I\"])\n" + \
        "b_k2 = B_K2NK1I.getRoot()\n" + \
        "for k2, (a_m, b_n) in a_k2 & b_k2:\n" + \
        "    B_NK1I = Tensor.fromFiber(rank_ids=[\"N\", \"K1I\"], fiber=b_n)\n" + \
        "    tmp4 = B_NK1I\n" + \
        "    tmp5 = tmp4.splitEqual(5)\n" + \
        "    B_N1N0K1I = tmp5\n" + \
        "    B_N1N0K1I.setRankIds(rank_ids=[\"N1\", \"N0\", \"K1I\"])\n" + \
        "    B_N1K1IN0 = B_N1N0K1I.swizzleRanks(rank_ids=[\"N1\", \"K1I\", \"N0\"])\n" + \
        "    b_n1 = B_N1K1IN0.getRoot()\n" + \
        "    for m, (z_n1, a_k1i) in z_m << a_m:\n" + \
        "        A_K1I = Tensor.fromFiber(rank_ids=[\"K1I\"], fiber=a_k1i)\n" + \
        "        tmp6 = A_K1I\n" + \
        "        tmp7 = tmp6.splitEqual(3)\n" + \
        "        A_K1K0 = tmp7\n" + \
        "        A_K1K0.setRankIds(rank_ids=[\"K1\", \"K0\"])\n" + \
        "        a_k1 = A_K1K0.getRoot()\n" + \
        "        for n1, (z_n0, b_k1i) in z_n1 << b_n1:\n" + \
        "            B_K1IN0 = Tensor.fromFiber(rank_ids=[\"K1I\", \"N0\"], fiber=b_k1i)\n" + \
        "            tmp8 = B_K1IN0\n" + \
        "            tmp9 = tmp8.splitNonUniform(a_k1)\n" + \
        "            B_K1K0N0 = tmp9\n" + \
        "            B_K1K0N0.setRankIds(rank_ids=[\"K1\", \"K0\", \"N0\"])\n" + \
        "            B_K1N0K0 = B_K1K0N0.swizzleRanks(rank_ids=[\"K1\", \"N0\", \"K0\"])\n" + \
        "            b_k1 = B_K1N0K0.getRoot()\n" + \
        "            for k1, (a_k0, b_n0) in a_k1 & b_k1:\n" + \
        "                for n0, (z_ref, b_k0) in z_n0 << b_n0:\n" + \
        "                    for k0, (a_val, b_val) in a_k0 & b_k0:\n" + \
        "                        z_ref += a_val * b_val\n" + \
        "tmp10 = Z_MN1N0\n" + \
        "tmp11 = tmp10.mergeRanks(depth=1, levels=1, coord_style=\"absolute\")\n" + \
        "tmp11.setRankIds(rank_ids=[\"M\", \"N\"])\n" + \
        "Z_MN = tmp11"

    assert str(HiFiber(einsum, mapping)) == hifiber


def test_hifiber_index_math_no_halo():
    yaml = """
    einsum:
      declaration:
        A: [K]
        Z: [M]
      expressions:
        - Z[m] = A[2 * m]
    mapping:
        partitioning:
            Z:
                M: [uniform_shape(10), uniform_shape(5)]
                K: [follow(M)]
    """

    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)

    hifiber = "Z_M2M1M0 = Tensor(rank_ids=[\"M2\", \"M1\", \"M0\"])\n" + \
        "tmp0 = A_K\n" + \
        "tmp1 = tmp0.splitUniform(2 * 10, depth=0)\n" + \
        "tmp2 = tmp1.splitUniform(2 * 5, depth=1)\n" + \
        "A_K2K1K0 = tmp2\n" + \
        "A_K2K1K0.setRankIds(rank_ids=[\"K2\", \"K1\", \"K0\"])\n" + \
        "z_m2 = Z_M2M1M0.getRoot()\n" + \
        "a_k2 = A_K2K1K0.getRoot()\n" + \
        "for m2, (z_m1, a_k1) in z_m2 << a_k2.project(trans_fn=lambda k2: 1 / 2 * k2):\n" + \
        "    inputs_m1 = Fiber.fromLazy(a_k1.project(trans_fn=lambda k1: 1 / 2 * k1))\n" + \
        "    for m1_pos, (m1, (z_m0, a_k0)) in enumerate(z_m1 << a_k1.project(trans_fn=lambda k1: 1 / 2 * k1)):\n" + \
        "        if m1_pos == 0:\n" + \
        "            m0_start = 0\n" + \
        "        else:\n" + \
        "            m0_start = m1\n" + \
        "        if m1_pos + 1 < len(inputs_m1):\n" + \
        "            m0_end = inputs_m1.getCoords()[m1_pos + 1]\n" + \
        "        else:\n" + \
        "            m0_end = M\n" + \
        "        for m0, (z_ref, a_val) in z_m0 << a_k0.project(trans_fn=lambda k0: 1 / 2 * k0, interval=(m0_start, m0_end)).prune(trans_fn=lambda i, c, p: c % 1 == 0):\n" + \
        "            z_ref += a_val\n" + \
        "tmp3 = Z_M2M1M0\n" + \
        "tmp4 = tmp3.mergeRanks(depth=0, levels=2, coord_style=\"absolute\")\n" + \
        "tmp4.setRankIds(rank_ids=[\"M\"])\n" + \
        "Z_M = tmp4"

    assert str(HiFiber(einsum, mapping)) == hifiber


def test_hifiber_conv():
    yaml = """
    einsum:
        declaration:
            F: [S]
            I: [W]
            O: [Q]
        expressions:
            - O[q] = I[q + s] * F[s]
    mapping:
        partitioning:
            O:
                Q: [uniform_shape(10)]
                W: [follow(Q)]
        loop-order:
            O: [Q1, W0, Q0]
    """

    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)

    hifiber = "O_Q1Q0 = Tensor(rank_ids=[\"Q1\", \"Q0\"])\n" + \
        "tmp0 = I_W\n" + \
        "tmp1 = tmp0.splitUniform(10, depth=0, post_halo=-1 + S)\n" + \
        "I_W1W0 = tmp1\n" + \
        "I_W1W0.setRankIds(rank_ids=[\"W1\", \"W0\"])\n" + \
        "o_q1 = O_Q1Q0.getRoot()\n" + \
        "f_s = F_S.getRoot()\n" + \
        "i_w1 = I_W1W0.getRoot()\n" + \
        "inputs_q1 = Fiber.fromLazy(i_w1.project(trans_fn=lambda w1: w1))\n" + \
        "for q1_pos, (q1, (o_q0, i_w0)) in enumerate(o_q1 << i_w1.project(trans_fn=lambda w1: w1)):\n" + \
        "    if q1_pos == 0:\n" + \
        "        q0_start = 0\n" + \
        "    else:\n" + \
        "        q0_start = q1\n" + \
        "    if q1_pos + 1 < len(inputs_q1):\n" + \
        "        q0_end = inputs_q1.getCoords()[q1_pos + 1]\n" + \
        "    else:\n" + \
        "        q0_end = Q\n" + \
        "    for w0, i_val in i_w0:\n" + \
        "        for q0, (o_ref, f_val) in o_q0 << f_s.project(trans_fn=lambda s: w0 + -1 * s, interval=(q0_start, q0_end)):\n" + \
        "            o_ref += i_val * f_val\n" + \
        "tmp2 = O_Q1Q0\n" + \
        "tmp3 = tmp2.mergeRanks(depth=0, levels=1, coord_style=\"absolute\")\n" + \
        "tmp3.setRankIds(rank_ids=[\"Q\"])\n" + \
        "O_Q = tmp3"

    assert str(HiFiber(einsum, mapping)) == hifiber


def test_hifiber_static_flattening():
    yaml = """
    einsum:
        declaration:
            A: [K, M]
            B: [K, N]
            Z: [M, N]
        expressions:
            - Z[m, n] = A[k, m] * B[k, n]
    mapping:
        partitioning:
            Z:
                K: [uniform_shape(4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
        loop-order:
            Z: [K1, MK01, N, MK00]
    """
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)

    hifiber = "Z_NM = Tensor(rank_ids=[\"N\", \"M\"])\n" + \
        "tmp0 = A_KM\n" + \
        "tmp1 = tmp0.splitUniform(4, depth=0)\n" + \
        "A_K1K0M = tmp1\n" + \
        "A_K1K0M.setRankIds(rank_ids=[\"K1\", \"K0\", \"M\"])\n" + \
        "tmp2 = B_KN\n" + \
        "tmp3 = tmp2.splitUniform(4, depth=0)\n" + \
        "B_K1K0N = tmp3\n" + \
        "B_K1K0N.setRankIds(rank_ids=[\"K1\", \"K0\", \"N\"])\n" + \
        "z_n = Z_NM.getRoot()\n" + \
        "A_K1MK0 = A_K1K0M.swizzleRanks(rank_ids=[\"K1\", \"M\", \"K0\"])\n" + \
        "B_K1NK0 = B_K1K0N.swizzleRanks(rank_ids=[\"K1\", \"N\", \"K0\"])\n" + \
        "tmp4 = A_K1MK0\n" + \
        "tmp5 = tmp4.flattenRanks(depth=1, levels=1, coord_style=\"tuple\")\n" + \
        "A_K1MK0_flat = tmp5\n" + \
        "A_K1MK0_flat.setRankIds(rank_ids=[\"K1\", \"MK0\"])\n" + \
        "b_k1 = B_K1NK0.getRoot()\n" + \
        "a_k1 = A_K1MK0_flat.getRoot()\n" + \
        "for k1, (a_mk0, b_n) in a_k1 & b_k1:\n" + \
        "    A_MK0 = Tensor.fromFiber(rank_ids=[\"MK0\"], fiber=a_mk0)\n" + \
        "    tmp6 = A_MK0\n" + \
        "    tmp7 = tmp6.splitEqual(5)\n" + \
        "    A_MK01MK00 = tmp7\n" + \
        "    A_MK01MK00.setRankIds(rank_ids=[\"MK01\", \"MK00\"])\n" + \
        "    a_mk01 = A_MK01MK00.getRoot()\n" + \
        "    for mk01, a_mk00 in a_mk01:\n" + \
        "        for n, (z_m, b_k0) in z_n << b_n:\n" + \
        "            for (m, k0), a_val in a_mk00:\n" + \
        "                z_ref = z_m.getPayloadRef(m)\n" + \
        "                b_val = b_k0.getPayload(k0)\n" + \
        "                z_ref += a_val * b_val\n" + \
        "tmp8 = Z_NM\n" + \
        "tmp9 = tmp8.swizzleRanks(rank_ids=[\"M\", \"N\"])\n" + \
        "Z_MN = tmp9"

    assert str(HiFiber(einsum, mapping)) == hifiber


def test_hifiber_dyn_flattening():
    yaml = """
    einsum:
        declaration:
            A: [K, M]
            B: [K, N]
            Z: [M, N]
        expressions:
            - Z[m, n] = A[k, m] * B[k, n]
    mapping:
        partitioning:
            Z:
                M: [uniform_shape(6)]
                K: [uniform_occupancy(A.4)]
                (M0, K0): [flatten()]
                M0K0: [uniform_occupancy(A.5)]
        loop-order:
            Z: [M1, K1, M0K01, N, M0K00]
    """
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)

    hifiber = "Z_M1NM0 = Tensor(rank_ids=[\"M1\", \"N\", \"M0\"])\n" + \
        "tmp0 = A_KM\n" + \
        "tmp1 = tmp0.splitUniform(6, depth=1)\n" + \
        "A_KM1M0 = tmp1\n" + \
        "A_KM1M0.setRankIds(rank_ids=[\"K\", \"M1\", \"M0\"])\n" + \
        "z_m1 = Z_M1NM0.getRoot()\n" + \
        "A_M1KM0 = A_KM1M0.swizzleRanks(rank_ids=[\"M1\", \"K\", \"M0\"])\n" + \
        "b_k = B_KN.getRoot()\n" + \
        "a_m1 = A_M1KM0.getRoot()\n" + \
        "B_KN = Tensor.fromFiber(rank_ids=[\"K\", \"N\"], fiber=b_k)\n" + \
        "for m1, (z_n, a_k) in z_m1 << a_m1:\n" + \
        "    A_KM0 = Tensor.fromFiber(rank_ids=[\"K\", \"M0\"], fiber=a_k)\n" + \
        "    tmp2 = A_KM0\n" + \
        "    tmp3 = tmp2.splitEqual(4)\n" + \
        "    A_K1K0M0 = tmp3\n" + \
        "    A_K1K0M0.setRankIds(rank_ids=[\"K1\", \"K0\", \"M0\"])\n" + \
        "    A_K1M0K0 = A_K1K0M0.swizzleRanks(rank_ids=[\"K1\", \"M0\", \"K0\"])\n" + \
        "    tmp4 = A_K1M0K0\n" + \
        "    tmp5 = tmp4.flattenRanks(depth=1, levels=1, coord_style=\"tuple\")\n" + \
        "    A_K1M0K0_flat = tmp5\n" + \
        "    A_K1M0K0_flat.setRankIds(rank_ids=[\"K1\", \"M0K0\"])\n" + \
        "    a_k1 = A_K1M0K0_flat.getRoot()\n" + \
        "    tmp6 = B_KN\n" + \
        "    tmp7 = tmp6.splitNonUniform(a_k1)\n" + \
        "    B_K1K0N = tmp7\n" + \
        "    B_K1K0N.setRankIds(rank_ids=[\"K1\", \"K0\", \"N\"])\n" + \
        "    B_K1NK0 = B_K1K0N.swizzleRanks(rank_ids=[\"K1\", \"N\", \"K0\"])\n" + \
        "    b_k1 = B_K1NK0.getRoot()\n" + \
        "    for k1, (a_m0k0, b_n) in a_k1 & b_k1:\n" + \
        "        A_M0K0 = Tensor.fromFiber(rank_ids=[\"M0K0\"], fiber=a_m0k0)\n" + \
        "        tmp8 = A_M0K0\n" + \
        "        tmp9 = tmp8.splitEqual(5)\n" + \
        "        A_M0K01M0K00 = tmp9\n" + \
        "        A_M0K01M0K00.setRankIds(rank_ids=[\"M0K01\", \"M0K00\"])\n" + \
        "        a_m0k01 = A_M0K01M0K00.getRoot()\n" + \
        "        for m0k01, a_m0k00 in a_m0k01:\n" + \
        "            for n, (z_m0, b_k0) in z_n << b_n:\n" + \
        "                for (m0, k0), a_val in a_m0k00:\n" + \
        "                    z_ref = z_m0.getPayloadRef(m0)\n" + \
        "                    b_val = b_k0.getPayload(k0)\n" + \
        "                    z_ref += a_val * b_val\n" + \
        "tmp10 = Z_M1NM0\n" + \
        "tmp11 = tmp10.swizzleRanks(rank_ids=[\"M1\", \"M0\", \"N\"])\n" + \
        "tmp12 = tmp11.mergeRanks(depth=0, levels=1, coord_style=\"absolute\")\n" + \
        "tmp12.setRankIds(rank_ids=[\"M\", \"N\"])\n" + \
        "Z_MN = tmp12"

    assert str(HiFiber(einsum, mapping)) == hifiber


def test_hifiber_hardware():
    yaml = """
    einsum:
      declaration:
        A: [K, M]
        B: [K, M]
        C: [K]
        Z: [M]
      expressions:
      - Z[m] = A[k, m] * B[k, m] * C[k]

    architecture:
      accel:
      - name: level0
        local:
        - name: DRAM
          class: DRAM
          attributes:
            bandwidth: 512

        subtree:
        - name: level1
          local:
          - name: L2Cache
            class: Cache
            attributes:
              width: 64
              depth: 1024
              bandwidth: 2048

    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
      - component: DRAM
        bindings:
        - tensor: Z
          rank: M
          type: elem
          format: default
      - component: L2Cache
        bindings:
        - tensor: Z
          rank: M
          type: elem
          format: default
    format:
      Z:
        default:
          rank-order: [M]
          M:
            format: C
            cbits: 32
            pbits: 64
    """
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    format_ = Format.from_str(yaml)

    hifiber = "Z_M = Tensor(rank_ids=[\"M\"], shape=[M])\n" + \
        "A_MK = A_KM.swizzleRanks(rank_ids=[\"M\", \"K\"])\n" + \
        "B_MK = B_KM.swizzleRanks(rank_ids=[\"M\", \"K\"])\n" + \
        "z_m = Z_M.getRoot()\n" + \
        "a_m = A_MK.getRoot()\n" + \
        "b_m = B_MK.getRoot()\n" + \
        "c_k = C_K.getRoot()\n" + \
        "Metrics.beginCollect(\"tmp/Z\")\n" + \
        "Metrics.trace(\"M\", type_=\"populate_read_0\", consumable=False)\n" + \
        "Metrics.trace(\"M\", type_=\"populate_write_0\", consumable=False)\n" + \
        "for m, (z_ref, (a_k, b_k)) in z_m << (a_m & b_m):\n" + \
        "    for k, (a_val, (b_val, c_val)) in a_k & (b_k & c_k):\n" + \
        "        z_ref += a_val * b_val * c_val\n" + \
        "Metrics.endCollect()\n" + \
        "metrics = {}\n" + \
        "metrics[\"Z\"] = {}\n" + \
        "formats = {\"Z\": Format(Z_M, {\"rank-order\": [\"M\"], \"M\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}})}\n" + \
        "bindings = [{\"tensor\": \"Z\", \"rank\": \"M\", \"type\": \"elem\", \"format\": \"default\"}]\n" + \
        "traces = {(\"Z\", \"M\", \"elem\", \"read\"): \"tmp/Z-M-populate_read_0.csv\", (\"Z\", \"M\", \"elem\", \"write\"): \"tmp/Z-M-populate_write_0.csv\"}\n" + \
        "traffic = Traffic.cacheTraffic(bindings, formats, traces, 65536, 64)\n" + \
        "metrics[\"Z\"][\"DRAM\"] = {}\n" + \
        "metrics[\"Z\"][\"DRAM\"][\"Z\"] = {}\n" + \
        "metrics[\"Z\"][\"DRAM\"][\"Z\"][\"read\"] = 0\n" + \
        "metrics[\"Z\"][\"DRAM\"][\"Z\"][\"write\"] = 0\n" + \
        "metrics[\"Z\"][\"DRAM\"][\"Z\"][\"read\"] += traffic[0][\"Z\"][\"read\"]\n" + \
        "metrics[\"Z\"][\"DRAM\"][\"Z\"][\"write\"] += traffic[0][\"Z\"][\"write\"]"

    assert str(HiFiber(einsum, mapping, arch, bindings, format_)) == hifiber


def test_hifiber_gamma_no_errors():
    # There is too much variation in the Gamma spec to test if the HiFiber
    # remains unchanged
    fname = "tests/integration/gamma.yaml"
    einsum = Einsum.from_file(fname)
    mapping = Mapping.from_file(fname)
    arch = Architecture.from_file(fname)
    bindings = Bindings.from_file(fname)
    format_ = Format.from_file(fname)

    print(HiFiber(einsum, mapping, arch, bindings, format_))
