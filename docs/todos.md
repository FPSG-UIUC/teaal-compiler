# To-Dos

## Bug Fixes

## Minor Changes

- [ ] Add tests to make sure that `get_all_parts` is used in the correct locations (vs `get_static_parts` and `get_dyn_parts`)
- [ ] Add public interface to top-level `es2hfa` import
- [ ] Clean up tests
    - Which example tests actually test something unique?
    - All tests should use the [tensor naming conventions](./tensor_naming.md)
    - Delete all unused tests except for cnn and conv2d
- [ ] Introduce .coord in spacetime notebook before .pos/default
- [ ] Rename HFA -> HiFiber and es2hfa -> teaal

## Major Changes

- [ ] Add occupancy-based partitioning
    - [x] Add single-level occupancy-based partitioning
    - [ ] Add multi-level occupancy-based partitioning
    - [ ] Add n-way occupancy-based partitioning
- [ ] Allow dynamic partitioning in the coordinate space
- [ ] Switch to comma-separated partitioning fields
- [ ] Update Tensor.fromFiber to copy if it already has an owning rank
- [ ] Add display specification to specify displaying tensors (which ones) or canvas
- [ ] Allow `uniform_occupancy`-split tensors to be merged with `flattenRanks`

## Planning

- [ ] How should we specify multiple levels of occupancy-based partitioning

## Other
