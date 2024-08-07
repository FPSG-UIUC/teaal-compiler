G0_II_ = Tensor(rank_ids=["I", "I_"], name="G0")
g0_i = G0_II_.getRoot()
a_i = A_IJK.getRoot()
b0_i_ = B0_I_JK.getRoot()
for i, (g0_i_, a_j) in g0_i << a_i:
    for i_, (g0_ref, b0_j) in g0_i_ << b0_i_:
        for j, (a_k, b0_k) in a_j & b0_j:
            for k, (a_val, b0_val) in a_k & b0_k:
                g0_ref += a_val * b0_val
G1_JJ_ = Tensor(rank_ids=["J", "J_"], name="G1")
A_JIK = A_IJK.swizzleRanks(rank_ids=["J", "I", "K"])
B1_J_IK = B1_IJ_K.swizzleRanks(rank_ids=["J_", "I", "K"])
g1_j = G1_JJ_.getRoot()
a_j = A_JIK.getRoot()
b1_j_ = B1_J_IK.getRoot()
for j, (g1_j_, a_i) in g1_j << a_j:
    for j_, (g1_ref, b1_i) in g1_j_ << b1_j_:
        for i, (a_k, b1_k) in a_i & b1_i:
            for k, (a_val, b1_val) in a_k & b1_k:
                g1_ref += a_val * b1_val
G2_KK_ = Tensor(rank_ids=["K", "K_"], name="G2")
A_KIJ = A_IJK.swizzleRanks(rank_ids=["K", "I", "J"])
B2_K_IJ = B2_IJK_.swizzleRanks(rank_ids=["K_", "I", "J"])
g2_k = G2_KK_.getRoot()
a_k = A_KIJ.getRoot()
b2_k_ = B2_K_IJ.getRoot()
for k, (g2_k_, a_i) in g2_k << a_k:
    for k_, (g2_ref, b2_i) in g2_k_ << b2_k_:
        for i, (a_j, b2_j) in a_i & b2_i:
            for j, (a_val, b2_val) in a_j & b2_j:
                g2_ref += a_val * b2_val