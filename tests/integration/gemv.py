T1_M = Tensor(rank_ids=["M"], name="T1")
B_MK = B_KM.swizzleRanks(rank_ids=["M", "K"])
t1_m = T1_M.getRoot()
a_k = A_K.getRoot()
b_m = B_MK.getRoot()
for m, (t1_ref, b_k) in t1_m << b_m:
    for k, (a_val, b_val) in a_k & b_k:
        t1_ref += a_val * b_val
Z_M = Tensor(rank_ids=["M"], name="Z")
z_m = Z_M.getRoot()
t1_m = T1_M.getRoot()
c_m = C_M.getRoot()
for m, (z_ref, (_, t1_val, c_val)) in z_m << (t1_m | c_m):
    z_ref <<= a * t1_val + b * c_val