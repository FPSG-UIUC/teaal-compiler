Convention for tensors:

---------------------------------



Variables that are tensor names are always capital letters. List tensors in alphabetical order on the RHS of the einsum (to the extent possible).



Input tensors:

Start at A, and go forward to B, C, D....



Output tensors:

Start at Z, and go backward to Y, X, W....



Reasoning:

if you name your output C/D/E then it changes between einsums and you always have to recalibrate your brain
Don't use O for outputs because it looks like zero and also big-O notation
If you add a new input tensor (e.g. for filtering) then you don't ever have to rename your outputs (or recalibrate your brain)


Conventions for indices:

---------------------------------



Variables that are indices (coordinates) are always lowercase letters. The mode/rank variable is always the same letter but Uppercase (i.e. for j in range(J)). Indices are always written in alphabetical order (NOT IN MEMORY LAYOUT ORDER)



Uncontracted variables:

Start at m, go forward n, o, p......



Contracted variables:

Start at k, go backward to j, i, h.....



Notes:

Contracted variables should never be used to index output tensors (e.g., never on the LHS of the einsum)
Technically, uncontracted covers both "expansion" and "merge" variables (as in Cartesian and Elementwise-mul respectively) but we do not distinguish between these two cases, though they are somewhat different in practice. Maybe rethink this in the future?


Reasoning:

Allows us to quickly look at complex einsums and figure out what is happening more quickly
Matches up perfectly with Nvidia GEMM notation (not an accident)
Because these variables start in the middle of the alphabet they are unlikely to clash with tensor names (A/B/C... and Z/Y/X...) even when written uppercase
Works great when translating to HFA (e.g. "Hey that's a rank "K" so I bet I need an intersection")


Serendipitous Bonuses:

It is a nice bonus that l is not used since it looks like 1 and I (depending on font)
It is a nice bonus that that this scheme divides the alphabet into 4 roughly equal, non-overlapping partitioned "namespaces"
k, the most common contracted variable, sounds like "contract" so that is a nice mnemonic


Examples:

-------------



Dot-product:

Z = A_k * B_k

GOOD



Cartesian Multiply:

Z_mn = A_m * B_n

GOOD



Elementwise Multiply:

Z_m = A_m * B_m

GOOD



Matrix-Vector Multiply:

Z_m = A_km * B_k

GOOD



Anti-Examples:

-------------------

Dot-product:

Z = A_i * B_i

BAD: start contracted indices at k



Cartesian Multiply:

C_mn = A_m * B_n

BAD: start output tensor at Z



Elementwise Multiply:

Z_k = A_k * B_k

BAD: k on the left-hand-side



Matrix-Vector Multiply:

Z_m = A_mk * B_k

BAD: indices should be in alphabetical order



Complex Examples:

--------------------------

SDDMM:

Z_mn = A_km * B_kn * C_mn

GOOD



MTTKRP:

Z_mn =  A_jkm * B_kn * C_jn

GOOD



Randomly generated:

Z_mnop = A_ijkmp * B_jkno * C_i

GOOD
