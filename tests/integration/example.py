T1_IJ = Tensor(rank_ids=["I", "J"], name="T1")
t1_i = T1_IJ.getRoot()
a_i = A_IJK.getRoot()
b_k = B_KL.getRoot()
for i, (t1_j, a_j) in t1_i << a_i:
    for j, (t1_ref, a_k) in t1_j << a_j:
        for k, (a_val, b_l) in a_k & b_k:
            for l, b_val in b_l:
                t1_ref += a_val * b_val
D_I = Tensor(rank_ids=["I"], name="D")
d_i = D_I.getRoot()
c_i = C_IJ.getRoot()
t1_i = T1_IJ.getRoot()
for i, (d_ref, (_, c_j, t1_j)) in d_i << (c_i | t1_i):
    for j, (_, c_val, t1_val) in c_j | t1_j:
        d_ref += c_val + t1_val