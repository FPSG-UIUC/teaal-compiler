Z_MNO = Tensor(rank_ids=["M", "N", "O"], name="Z")
B_OK = B_KO.swizzleRanks(rank_ids=["O", "K"])
z_m = Z_MNO.getRoot()
a_m = A_MNK.getRoot()
b_o = B_OK.getRoot()
for m, (z_n, a_n) in z_m << a_m:
    for n, (z_o, a_k) in z_n << a_n:
        for o, (z_ref, b_k) in z_o << b_o:
            for k, (a_val, b_val) in a_k & b_k:
                z_ref += a_val * b_val