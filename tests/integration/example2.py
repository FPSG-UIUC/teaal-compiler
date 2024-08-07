D_ = Tensor(rank_ids=[], name="D")
B_IJ = B_JI.swizzleRanks(rank_ids=["I", "J"])
d_ref = D_.getRoot()
a_i = A_IJ.getRoot()
b_i = B_IJ.getRoot()
for i, (a_j, b_j) in a_i & b_i:
    for j, (a_val, b_val) in a_j & b_j:
        d_ref += a_val * b_val