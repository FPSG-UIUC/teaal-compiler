Z_M = Tensor(rank_ids=["M"])
z_m = Z_M.getRoot()
a_k = A_K.getRoot()
b_m = B_MK.getRoot()
for m, (z_ref, b_k) in z_m << b_m:
    for k, (a_val, b_val) in a_k & b_k:
        z_ref += a_val * b_val