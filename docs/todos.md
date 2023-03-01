# To-Dos

## Priority: 1

[x] Support flattening
[ ] Allow for variable-named partition sizes
[ ] Correctly flatten together output after occupancy-space partitioning
[ ] Support the new architecture/binding specification

## Priority: 2

[ ] Remove the `sum(...).(...)` from the specification
[ ] Correctly manage Einsums with both multiplication and addition
[ ] Report which Einsums are fused together
[ ] Disallow partitioning on tensors with expressions as indices

## Priority: 3

[ ] Support index expressions that are a single integer
[ ] Support multi-cast/broadcast

## Priority: 4+

[ ] Correctly support partitioning on tensors with expressions as indices
[ ] Add a way to generate `displayTensor()` calls
[ ] Allow the formatted arrays for all fibers of a rank to be flattened together
[ ] On-chip memory traffic modeling
[ ] Additional swizzling algorithms (e.g. SpArch)
[ ] More control over Einsum pipelining

## Future Meeting Agendas

[ ] Discuss how trace generation works
[ ] Discuss Joel's Isosceles paper
