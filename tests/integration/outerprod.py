Z_MN = Tensor(rank_ids=["M", "N"])
z_m = Z_MN.getRoot()
a_m = A_M.getRoot()
b_n = B_N.getRoot()
for m, (z_n, a_val) in z_m << a_m:
    for n, (z_ref, b_val) in z_n << b_n:
        z_ref += a_val * b_val