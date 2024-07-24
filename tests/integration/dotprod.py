Z_ = Tensor(rank_ids=[], name="Z")
z_ref = Z_.getRoot()
a_k = A_K.getRoot()
b_k = B_K.getRoot()
for k, (a_val, b_val) in a_k & b_k:
    z_ref += a_val * b_val