Z_IK = Tensor(rank_ids=["I", "K"], name="Z")
B_KJ = B_JK.swizzleRanks(rank_ids=["K", "J"])
z_i = Z_IK.getRoot()
a_i = A_IJ.getRoot()
b_k = B_KJ.getRoot()
for i, (z_k, a_j) in z_i << a_i:
    for k, (z_ref, b_j) in z_k << b_k:
        for j, (a_val, b_val) in a_j & b_j:
            z_ref += a_val * b_val