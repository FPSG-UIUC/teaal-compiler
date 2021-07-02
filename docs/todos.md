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
    - [ ] Clone docker repository and figure out how to build
    - [ ] Figure out how to remove existing containers
    - [ ] Figure out how to check if you have access to a Github repository
    - [ ] Add pip install command (`RUN access='command' && pip install ...`)
- [ ] Separate Input into Einsum and Mapping
    - Rename current IR Mapping :(
    - Refactor to manage the separation
- [ ] Refactor Translator to HFA object with `__str__()` method
- [ ] Make display style per-index with default as position space

## Planning

- [ ] Plan other types of partitioning

## Other

