Z_MN = Tensor(rank_ids=["M", "N"], name="Z")
B_NK = B_KN.swizzleRanks(rank_ids=["N", "K"])
C_NJ = C_JN.swizzleRanks(rank_ids=["N", "J"])
z_m = Z_MN.getRoot()
a_m = A_MKJ.getRoot()
b_n = B_NK.getRoot()
c_n = C_NJ.getRoot()
for m, (z_n, a_k) in z_m << a_m:
    for n, (z_ref, (b_k, c_j)) in z_n << (b_n & c_n):
        for k, (a_j, b_val) in a_k & b_k:
            for j, (a_val, c_val) in a_j & c_j:
                z_ref += a_val * b_val * c_val