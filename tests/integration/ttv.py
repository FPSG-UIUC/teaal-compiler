Z_MN = Tensor(rank_ids=["M", "N"], name="Z")
z_m = Z_MN.getRoot()
a_m = A_MNK.getRoot()
b_k = B_K.getRoot()
for m, (z_n, a_n) in z_m << a_m:
    for n, (z_ref, a_k) in z_n << a_n:
        for k, (a_val, b_val) in a_k & b_k:
            z_ref += a_val * b_val