D_I = Tensor(rank_ids=["I"], name="D")
d_i = D_I.getRoot()
c_i = C_IJ.getRoot()
for i, (d_ref, c_j) in d_i << c_i:
    for j, c_val in c_j:
        d_ref += c_val