# To-Dos

## Bug Fixes

## Minor Changes

- [ ] Fix `SAssign` in `make_body`
    - [ ] Make an assignable type
    - [ ] Transition all SAssign and SIAssign to use the new type
    - [ ] Switch `Graphics.make_body()` to use a dictionary access
- [ ] Fix `hfa-compiler-notebooks` repo to use new interface
    - [ ] Change notebooks
    - [ ] Change prelude.py
- [ ] Use partitioning parser instead of `make_uniform_shape` for all relevant tests
- [ ] Add all other integration tests
- [ ] Clean up tests
    - Which example tests actually test something unique?
    - All tests should use the [tensor naming conventions](./tensor_naming.md)
    - Delete all unused tests except for cnn and conv2d

## Major Changes

- [ ] Use Joel's Javascript code for filling cells
- [ ] Integration with the HFA docker environment
    - [ ] Clone docker repository and figure out how to build
    - [ ] Figure out how to remove existing containers
    - [ ] Figure out how to check if you have access to a Github repository
    - [ ] Add pip install command (`RUN access='command' && pip install ...`)

## Planning

- [ ] Think about how to represent slip
- [ ] Plan other types of partitioning

## Other

