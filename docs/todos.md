# To-Dos

## Bug Fixes

## Minor Changes

- [ ] Add all other integration tests
- [ ] Clean up tests
    - Which example tests actually test something unique?
    - All tests should use the [tensor naming conventions](./tensor_naming.md)
    - Delete all unused tests except for cnn and conv2d

## Major Changes

- [ ] Integration with the HFA docker environment
    - [ ] Figure out how updated Dockerfile should be committed
- [ ] Add display specification to specify displaying tensors (which ones) or canvas
- [ ] Fix `SAssign` in `make_body`
    - [ ] Make an assignable type
    - [ ] Transition all SAssign and SIAssign to use the new type
    - [ ] Switch `Graphics.make_body()` to use a dictionary access

## Planning

- [ ] Plan other types of partitioning

## Other

