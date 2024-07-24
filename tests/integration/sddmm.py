T1_MN = Tensor(rank_ids=["M", "N"], name="T1")
B_NK = B_KN.swizzleRanks(rank_ids=["N", "K"])
t1_m = T1_MN.getRoot()
a_m = A_MK.getRoot()
b_n = B_NK.getRoot()
for m, (t1_n, a_k) in t1_m << a_m:
    for n, (t1_ref, b_k) in t1_n << b_n:
        for k, (a_val, b_val) in a_k & b_k:
            t1_ref += a_val * b_val
Z_MN = Tensor(rank_ids=["M", "N"], name="Z")
z_m = Z_MN.getRoot()
c_m = C_MN.getRoot()
t1_m = T1_MN.getRoot()
for m, (z_n, (c_n, t1_n)) in z_m << (c_m & t1_m):
    for n, (z_ref, (c_val, t1_val)) in z_n << (c_n & t1_n):
        z_ref += c_val * t1_val