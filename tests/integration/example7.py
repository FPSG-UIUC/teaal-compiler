Z_MN = Tensor(rank_ids=["M", "N"], name="Z")
z_m = Z_MN.getRoot()
a_m = A_MN.getRoot()
b_m = B_MN.getRoot()
for m, (z_n, (_, a_n, b_n)) in z_m << (a_m | b_m):
    for n, (z_ref, (_, a_val, b_val)) in z_n << (a_n | b_n):
        z_ref += a_val + b_val
Z_MN = Tensor(rank_ids=["M", "N"], name="Z")
z_m = Z_MN.getRoot()
a_m = A_MN.getRoot()
b_m = B_MN.getRoot()
for m, (z_n, (a_n, b_n)) in z_m << (a_m & b_m):
    for n, (z_ref, (a_val, b_val)) in z_n << (a_n & b_n):
        z_ref += a_val * b_val