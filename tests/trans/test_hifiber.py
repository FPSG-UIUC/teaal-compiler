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

    hifiber = "T1_NM = Tensor(rank_ids=[\"N\", \"M\"])\n" + \
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
        "T1_MN = T1_NM.swizzleRanks(rank_ids=[\"M\", \"N\"])\n" + \
        "displayCanvas(canvas)\n" + \
        "Z_M2N2M1N1M0N0 = Tensor(rank_ids=[\"M2\", \"N2\", \"M1\", \"N1\", \"M0\", \"N0\"])\n" + \
        "tmp0 = T1_MN\n" + \
        "tmp1 = tmp0.splitUniform(4, depth=0)\n" + \
        "tmp2 = tmp1.splitUniform(2, depth=1)\n" + \
        "T1_M2M1M0N = tmp2\n" + \
        "T1_M2M1M0N.setRankIds(rank_ids=[\"M2\", \"M1\", \"M0\", \"N\"])\n" + \
        "tmp3 = T1_M2M1M0N\n" + \
        "tmp4 = tmp3.splitUniform(6, depth=3)\n" + \
        "tmp5 = tmp4.splitUniform(3, depth=4)\n" + \
        "T1_M2M1M0N2N1N0 = tmp5\n" + \
        "T1_M2M1M0N2N1N0.setRankIds(rank_ids=[\"M2\", \"M1\", \"M0\", \"N2\", \"N1\", \"N0\"])\n" + \
        "tmp6 = C_NM\n" + \
        "tmp7 = tmp6.splitUniform(6, depth=0)\n" + \
        "tmp8 = tmp7.splitUniform(3, depth=1)\n" + \
        "C_N2N1N0M = tmp8\n" + \
        "C_N2N1N0M.setRankIds(rank_ids=[\"N2\", \"N1\", \"N0\", \"M\"])\n" + \
        "tmp9 = C_N2N1N0M\n" + \
        "tmp10 = tmp9.splitUniform(4, depth=3)\n" + \
        "tmp11 = tmp10.splitUniform(2, depth=4)\n" + \
        "C_N2N1N0M2M1M0 = tmp11\n" + \
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
        "Z_N2N1N0M2M1M0 = Z_M2N2M1N1M0N0.swizzleRanks(rank_ids=[\"N2\", \"N1\", \"N0\", \"M2\", \"M1\", \"M0\"])\n" + \
        "tmp12 = Z_N2N1N0M2M1M0\n" + \
        "tmp13 = tmp12.flattenRanks(depth=0, levels=2, coord_style=\"absolute\")\n" + \
        "tmp14 = tmp13.flattenRanks(depth=1, levels=2, coord_style=\"absolute\")\n" + \
        "Z_NM = tmp14\n" + \
        "Z_NM.setRankIds(rank_ids=[\"N\", \"M\"])"

    assert str(HiFiber(einsum, mapping)) == hifiber


def test_hifiber_dyn_part():
    yaml = """
    einsum:
        declaration:
            A: [K, M]
            B: [K, N]
            Z: [M, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
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
        "tmp11 = tmp10.flattenRanks(depth=1, levels=1, coord_style=\"absolute\")\n" + \
        "Z_MN = tmp11\n" + \
        "Z_MN.setRankIds(rank_ids=[\"M\", \"N\"])"

    assert str(HiFiber(einsum, mapping)) == hifiber


def test_hifiber_conv():
    yaml = """
    einsum:
        declaration:
            F: [S]
            I: [W]
            O: [Q]
        expressions:
            - O[q] = sum(S).(I[q + s] * F[s])
    mapping:
        partitioning:
            O:
                Q: [uniform_shape(10)]
        loop-order:
            O: [Q1, W0, Q0]
    """

    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)

    hifiber = "O_Q1Q0 = Tensor(rank_ids=[\"Q1\", \"Q0\"])\n" + \
        "tmp0 = I_W\n" + \
        "tmp1 = tmp0.splitUniform(10, depth=0, halo=-1 + S)\n" + \
        "I_Q1W0 = tmp1\n" + \
        "I_Q1W0.setRankIds(rank_ids=[\"Q1\", \"W0\"])\n" + \
        "o_q1 = O_Q1Q0.getRoot()\n" + \
        "f_s = F_S.getRoot()\n" + \
        "i_q1 = I_Q1W0.getRoot()\n" + \
        "inputs_q1 = i_q1\n" + \
        "for q1_pos, (q1, (o_q0, i_w0)) in enumerate(o_q1 << i_q1):\n" + \
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
        "tmp3 = tmp2.flattenRanks(depth=0, levels=1, coord_style=\"absolute\")\n" + \
        "O_Q = tmp3\n" + \
        "O_Q.setRankIds(rank_ids=[\"Q\"])"

    assert str(HiFiber(einsum, mapping)) == hifiber


def test_hifiber_hardware():
    fname = "tests/integration/gamma.yaml"
    einsum = Einsum.from_file(fname)
    mapping = Mapping.from_file(fname)
    arch = Architecture.from_file(fname)
    bindings = Bindings.from_file(fname)
    format_ = Format.from_file(fname)

    hifiber = "T_MKN = Tensor(rank_ids=[\"M\", \"K\", \"N\"])\n" + \
        "t_m = T_MKN.getRoot()\n" + \
        "a_m = A_MK.getRoot()\n" + \
        "b_k = B_KN.getRoot()\n" + \
        "B_KN.setCollecting(\"K\", True)\n" + \
        "Metrics.beginCollect([\"M\", \"K\", \"N\"])\n" + \
        "for m, (t_k, a_k) in t_m << a_m:\n" + \
        "    for k, (t_n, (a_val, b_n)) in t_k << (a_k & b_k):\n" + \
        "        for n, (t_ref, b_val) in t_n << b_n:\n" + \
        "            t_ref += b_val\n" + \
        "Metrics.endCollect()\n" + \
        "metrics = {}\n" + \
        "metrics[\"T\"] = {}\n" + \
        "metrics[\"T\"][\"T footprint\"] = 0\n" + \
        "metrics[\"T\"][\"T traffic\"] = 0\n" + \
        "A_MK_format = Format(A_MK, {\"M\": {\"format\": \"U\", \"rhbits\": 32, \"pbits\": 32}, \"K\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}})\n" + \
        "metrics[\"T\"][\"A footprint\"] = A_MK_format.getTensor()\n" + \
        "metrics[\"T\"][\"A traffic\"] = metrics[\"T\"][\"A footprint\"]\n" + \
        "B_KN_format = Format(B_KN, {\"K\": {\"format\": \"U\", \"rhbits\": 32, \"pbits\": 32}, \"N\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}})\n" + \
        "metrics[\"T\"][\"B footprint\"] = B_KN_format.getTensor()\n" + \
        "metrics[\"T\"][\"B traffic\"] = Traffic.cacheTraffic(B_KN, \"K\", B_KN_format, 25165824) + B_KN_format.getRank(\"K\")\n" + \
        "metrics[\"T\"][\"K intersections\"] = Compute.lfCount(Metrics.dump(), \"K\", 0)\n" + \
        "Z_MN = Tensor(rank_ids=[\"M\", \"N\"])\n" + \
        "T_MNK = T_MKN.swizzleRanks(rank_ids=[\"M\", \"N\", \"K\"])\n" + \
        "z_m = Z_MN.getRoot()\n" + \
        "t_m = T_MNK.getRoot()\n" + \
        "a_m = A_MK.getRoot()\n" + \
        "Metrics.beginCollect([\"M\", \"N\", \"K\"])\n" + \
        "for m, (z_n, (t_n, a_k)) in z_m << (t_m & a_m):\n" + \
        "    for n, (z_ref, t_k) in z_n << t_n:\n" + \
        "        for k, (t_val, a_val) in t_k & a_k:\n" + \
        "            z_ref += t_val * a_val\n" + \
        "Metrics.endCollect()\n" + \
        "metrics[\"Z\"] = {}\n" + \
        "Z_MN_format = Format(Z_MN, {\"M\": {\"format\": \"U\", \"rhbits\": 32, \"pbits\": 32}, \"N\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}})\n" + \
        "metrics[\"Z\"][\"Z footprint\"] = Z_MN_format.getTensor()\n" + \
        "metrics[\"Z\"][\"Z traffic\"] = metrics[\"Z\"][\"Z footprint\"]\n" + \
        "metrics[\"Z\"][\"T footprint\"] = 0\n" + \
        "metrics[\"Z\"][\"T traffic\"] = 0\n" + \
        "A_MK_format = Format(A_MK, {\"M\": {\"format\": \"U\", \"rhbits\": 32, \"pbits\": 32}, \"K\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}})\n" + \
        "metrics[\"Z\"][\"A footprint\"] = A_MK_format.getTensor()\n" + \
        "metrics[\"Z\"][\"A traffic\"] = metrics[\"Z\"][\"A footprint\"]\n" + \
        "metrics[\"Z\"][\"mul\"] = Compute.opCount(Metrics.dump(), \"mul\")\n" + \
        "metrics[\"Z\"][\"add\"] = Compute.opCount(Metrics.dump(), \"add\")\n" + \
        "metrics[\"Z\"][\"T_MKN merge ops\"] = Compute.swapCount(T_MKN, 1, 64, 1)"

    assert str(HiFiber(einsum, mapping, arch, bindings, format_)) == hifiber