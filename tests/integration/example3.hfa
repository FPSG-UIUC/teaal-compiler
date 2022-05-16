D_ = Tensor(rank_ids=[])
B_IJK = B_JKI.swizzleRanks(rank_ids=["I", "J", "K"])
d_ref = D_.getRoot()
a_i = A_IJK.getRoot()
b_i = B_IJK.getRoot()
for i, (a_j, b_j) in a_i & b_i:
    for j, (a_k, b_k) in a_j & b_j:
        for k, (a_val, b_val) in a_k & b_k:
            d_ref += a_val * b_val