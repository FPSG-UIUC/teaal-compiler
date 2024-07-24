D_ = Tensor(rank_ids=[], name="D")
B_IJK = B_JKI.swizzleRanks(rank_ids=["I", "J", "K"])
C_IJK = C_JKI.swizzleRanks(rank_ids=["I", "J", "K"])
d_ref = D_.getRoot()
a_i = A_IJK.getRoot()
b_i = B_IJK.getRoot()
c_i = C_IJK.getRoot()
for i, (a_j, (b_j, c_j)) in a_i & (b_i & c_i):
    for j, (a_k, (b_k, c_k)) in a_j & (b_j & c_j):
        for k, (a_val, (b_val, c_val)) in a_k & (b_k & c_k):
            d_ref += a_val * b_val * c_val