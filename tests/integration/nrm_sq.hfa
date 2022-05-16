T_ABIJ = Tensor(rank_ids=["A", "B", "I", "J"])
t_a = T_ABIJ.getRoot()
v_a = V_ABIJ.getRoot()
for a, (t_b, v_b) in t_a << v_a:
    for b, (t_i, v_i) in t_b << v_b:
        for i, (t_j, v_j) in t_i << v_i:
            for j, (t_ref, v_val) in t_j << v_j:
                t_ref += v_val
Q_ = Tensor(rank_ids=[])
q_ref = Q_.getRoot()
v_a = V_ABIJ.getRoot()
t_a = T_ABIJ.getRoot()
for a, (v_b, t_b) in v_a & t_a:
    for b, (v_i, t_i) in v_b & t_b:
        for i, (v_j, t_j) in v_i & t_i:
            for j, (v_val, t_val) in v_j & t_j:
                q_ref += v_val * t_val