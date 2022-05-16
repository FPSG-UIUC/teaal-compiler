Z_MN = Tensor(rank_ids=["M", "N"])
A_MK = A_KM.swizzleRanks(rank_ids=["M", "K"])
B_NK = B_KN.swizzleRanks(rank_ids=["N", "K"])
z_m = Z_MN.getRoot()
a_m = A_MK.getRoot()
b_n = B_NK.getRoot()
for m, (z_n, a_k) in z_m << a_m:
    for n, (z_ref, b_k) in z_n << b_n:
        for k, (a_val, b_val) in a_k & b_k:
            z_ref += a_val * b_val