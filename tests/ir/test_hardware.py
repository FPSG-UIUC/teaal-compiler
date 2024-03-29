import pytest

from teaal.ir.component import *
from teaal.ir.hardware import Hardware
from teaal.ir.level import Level
from teaal.ir.program import Program
from teaal.parse import *


def build_outerspace_yaml():
    with open("tests/integration/outerspace.yaml", "r") as f:
        return f.read()


def parse_yamls(yaml):
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)
    program.add_einsum(0)

    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)

    return Hardware(arch, bindings, program)


def test_no_arch():
    yaml = """
    einsum:
      declaration:
        Z: [M]
      expressions:
      - Z[m] = a
    bindings:
      Z:
      - config: arch
        prefix: tmp/Z
    """
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))

    with pytest.raises(ValueError) as excinfo:
        Hardware(arch, bindings, program)
    assert str(excinfo.value) == "Empty architecture specification"


def test_bad_arch():
    yaml = """
    einsum:
      declaration:
        Z: [M]
      expressions:
      - Z[m] = a
    architecture:
      config0:
      -   name: foo
      - name: bar
    bindings:
      Z:
      - config: config0
        prefix: tmp/Z
    """
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))

    with pytest.raises(ValueError) as excinfo:
        Hardware(arch, bindings, program)
    assert str(
        excinfo.value) == "Configuration config0 must have a single root level"


def test_bad_component():
    yaml = """
    einsum:
      declaration:
        Z: [M]
      expressions:
      - Z[m] = a
    architecture:
      accel:
      - name: System
        local:
        - name: BAD
          class: foo
    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
    """
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))

    with pytest.raises(ValueError) as excinfo:
        Hardware(arch, bindings, program)
    assert str(excinfo.value) == "Unknown class: foo"


def test_bad_intersector():
    yaml = """
    einsum:
      declaration:
        Z: [M]
      expressions:
      - Z[m] = a
    architecture:
      accel:
      - name: System
        local:
        - name: BAD
          class: Intersector
          attributes:
            type: foo
    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
    """
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))

    with pytest.raises(ValueError) as excinfo:
        Hardware(arch, bindings, program)
    assert str(excinfo.value) == "Unknown intersection type: foo"


def test_no_binding():
    yaml = """
    einsum:
      declaration:
        Z: [M]
      expressions:
      - Z[m] = a
    architecture:
      arch:
      - name: System
        local:
        - name: Cache
          class: Cache
    bindings:
      Z:
      - config: arch
        prefix: tmp/Z
    """
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    hardware = Hardware(arch, bindings, program)

    cache = CacheComponent("Cache", 1, {}, {})
    assert hardware.get_component("Cache") == cache


def test_get_component():
    yaml = """
    einsum:
      declaration:
        A: [K, M]
        B: [K, N]
        T: [K, M, N]
        Z: [M, N]
      expressions:
        - T[k,m,n] = take(A[k,m], B[k,n], 1)
        - Z[m,n] = T[k,m,n] * A[k,m]
    mapping:
      rank-order:
        A: [M, K]
        B: [K, N]
        T: [M, K, N]
        Z: [M, N]
      partitioning:
        T:
          M: [uniform_occupancy(A.32)]
          K: [uniform_occupancy(A.64)]
        Z:
          M: [uniform_occupancy(A.32)]
          K: [uniform_occupancy(A.64)]
      loop-order:
        T: [M1, M0, K1, K0, N]
        Z: [M1, M0, K1, N, K0]
      spacetime:
        T:
          space: [M0, K1]
          time: [M1, K0, N]
        Z:
          space: [M0, K1]
          time: [M1, N, K0]
    architecture:
      Accelerator:
      - name: Base
        local:
        - name: LLB
          class: Buffet
          attributes:
            width: 64
            depth: 3145728

        - name: FiberCache
          class: Cache
          attributes:
            width: 64
            depth: 3145728

        - name: Compute
          class: Compute
          attributes:
            type: mul

        - name: Memory
          class: DRAM
          attributes:
            bandwidth: 128

        - name: LFIntersect
          class: Intersector
          attributes:
            type: leader-follower

        - name: HighRadixMerger
          class: Merger
          attributes:
            inputs: 64
            comparator_radix: 64
            outputs: 1
            order: fifo
            reduce: False

        - name: TopSequencer
          class: Sequencer
          attributes:
            num_ranks: 3

        - name: SAIntersect
          class: Intersector
          attributes:
            type: skip-ahead

        - name: TFIntersect
          class: Intersector
          attributes:
            type: two-finger

    bindings:
      T:
      - config: Accelerator
        prefix: tmp/T
      - component: LLB
        bindings:
        - tensor: A
          rank: K2
          format: default
          type: payload
          evict-on: root
        - tensor: B
          rank: K2
          format: default
          type: payload
          evict-on: root

      - component: FiberCache
        bindings:
        - tensor: B
          rank: K
          format: default
          type: payload

      - component: Memory
        bindings:
        - tensor: A
          rank: K2
          format: default
          type: payload
        - tensor: B
          rank: K2
          format: default
          type: payload

      - component: LFIntersect
        bindings:
        - rank: K
          leader: A

      Z:
      - config: Accelerator
        prefix: tmp/Z
      - component: LLB
        bindings:
        - tensor: Z
          rank: N2
          format: default
          type: payload
          evict-on: root

      - component: Compute
        bindings:
        - op: mul

      - component: Memory
        bindings:
        - tensor: Z
          rank: N2
          format: default
          type: payload

      - component: HighRadixMerger
        bindings:
        - tensor: T
          init-ranks: [M, K, N]
          final-ranks: [M, N, K]

      - component: TopSequencer
        bindings:
        - rank: M2
        - rank: K2
        - rank: N1

      - component: SAIntersect
        bindings:
        - rank: K2

      - component: TFIntersect
        bindings:
        - rank: K1
    """
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    hardware = Hardware(arch, bindings, program)

    def assert_component(type_, name, attrs):
        binding = bindings.get_component(name)
        component = type_(name, 1, attrs, binding)

        assert hardware.get_component(name) == component

    attrs = {"width": 64, "depth": 3145728}
    assert_component(BuffetComponent, "LLB", attrs)

    attrs = {"width": 64, "depth": 3145728}
    assert_component(CacheComponent, "FiberCache", attrs)

    assert_component(ComputeComponent, "Compute", {"type": "mul"})

    attrs = {"datawidth": 8, "bandwidth": 128}
    assert_component(DRAMComponent, "Memory", attrs)

    assert_component(LeaderFollowerComponent, "LFIntersect", {})

    attrs = {
        "inputs": 64,
        "comparator_radix": 64,
        "outputs": 1,
        "order": "fifo",
        "reduce": False
    }
    assert_component(MergerComponent, "HighRadixMerger", attrs)

    attrs = {"num_ranks": 3}
    assert_component(SequencerComponent, "TopSequencer", attrs)

    assert_component(SkipAheadComponent, "SAIntersect", {})

    assert_component(TwoFingerComponent, "TFIntersect", {})


def test_get_components():
    yaml = """
    einsum:
      declaration:
        Z: [M]
        X: [M]
        A: [K, M]
        D: [J, M]
      expressions:
      - Z[m] = A[k, m]
      - X[m] = D[j, m]

    architecture:
      accel:
      - name: System

        local:
        - name: Intersect0
          class: Intersector
          attributes:
            type: skip-ahead

        subtree:
        - name: PE

          local:
          - name: Intersect1
            class: Intersector
            attributes:
              type: skip-ahead

          - name: MAC
            class: compute
            attributes:
              type: add

    bindings:
      Z:
      - config: accel
        prefix: tmp/Z

      - component: Intersect0
        bindings:
        - rank: K

      - component: MAC
        bindings:
        - op: add

      X:
      - config: accel
        prefix: tmp/X
      - component: Intersect1
        bindings:
        - rank: J

    """
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    hardware = Hardware(arch, bindings, program)

    intersect = SkipAheadComponent(
        "Intersect0", 1, {}, bindings.get_component("Intersect0"))
    mac = ComputeComponent("MAC", 1,
                           {"type": "add"},
                           bindings.get_component("MAC"))

    assert hardware.get_components(
        "Z", FunctionalComponent) == [
        intersect, mac]


def test_get_config():
    yaml = build_outerspace_yaml()
    hardware = parse_yamls(yaml)

    assert hardware.get_config("T0") == "MultiplyPhase"
    assert hardware.get_config("T1") == "MergePhase"
    assert hardware.get_config("Z") == "MergePhase"


def test_get_frequency_unspecified():
    yaml = """
    einsum:
      declaration:
        Z: [M]
      expressions:
      - Z[m] = a
    architecture:
      accel:
      - name: System
    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
    """
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    hardware = Hardware(arch, bindings, program)

    with pytest.raises(ValueError) as excinfo:
        hardware.get_frequency("Z")
    assert str(excinfo.value) == "Unspecified clock frequency for config accel"


def test_get_frequency_bad():
    yaml = """
    einsum:
      declaration:
        Z: [M]
      expressions:
      - Z[m] = a
    architecture:
      accel:
      - name: System
        attributes:
          clock_frequency: foo
    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
    """
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    hardware = Hardware(arch, bindings, program)

    with pytest.raises(ValueError) as excinfo:
        hardware.get_frequency("Z")
    assert str(excinfo.value) == "Bad clock frequency for config accel"


def test_get_frequency():
    yaml = build_outerspace_yaml()
    hardware = parse_yamls(yaml)

    assert hardware.get_frequency("Z") == 1500000000


def test_get_traffic_path_multiple_bindings():
    yaml = """
    einsum:
      declaration:
        Z: [M]
        A: [M]
      expressions:
      - Z[m] = A[m]

    architecture:
      accel:
      - name: BAD

        local:
        - name: Memory0
          class: DRAM

        - name: Memory1
          class: DRAM

        - name: Compute
          class: compute
          attributes:
            type: add

    bindings:
      Z:
      - config: accel
        prefix: tmp/Z

      - component: Memory0
        bindings:
        - tensor: A
          rank: M
          type: payload
          format: default

      - component: Memory1
        bindings:
        - tensor: A
          rank: M
          type: payload
          format: default

      - component: Compute
        bindings:
        - op: add
    """
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)
    hardware = Hardware(arch, bindings, program)

    with pytest.raises(ValueError) as excinfo:
        hardware.get_traffic_path("A", "M", "payload", "default")
    assert str(excinfo.value) == "Multiple traffic paths for tensor A in Einsum Z"


def test_get_traffic_path():
    yaml = """
    einsum:
      declaration:
        A: [M]
        B: [M, K]
        X: [M]
        Z: [M]
      expressions:
      - X[m] = A[m] * B[m, k]
      - Z[m] = A[m] + B[m]
    architecture:
      accel:
      - name: System

        local:
        - name: Memory
          class: DRAM

        subtree:
        - name: Stages
          local:
          - name: Intersection
            class: Intersector
            attributes:
              type: skip-ahead

          - name: LLB
            class: Buffet

          subtree:
          - name: Stage0
            local:
            - name: S0B
              class: Buffet

            - name: MAC0
              class: compute
              attributes:
                type: mul

          - name: Stage1
            local:
            - name: S1B
              class: Buffet

            - name: MAC1
              class: compute
              attributes:
                type: mul

          - name: Stage2
            local:
            - name: S2B
              class: Buffet

            - name: MAC2
              class: compute
              attributes:
                type: mul

    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
      - component: Memory
        bindings:
        - tensor: A
          rank: M
          format: default
          type: payload
          evict-on: root

        - tensor: Z
          rank: M
          format: default
          type: payload
          evict-on: root

      - component: S0B
        bindings:
        - tensor: A
          rank: M
          format: default
          type: payload
          evict-on: root
        - tensor: Z
          rank: M
          format: default
          type: payload
          evict-on: root

      - component: MAC0
        bindings:
        - op: mul

      - component: S1B
        bindings:
        - tensor: Z
          rank: M
          format: default
          type: coord
          evict-on: root

      X:
      - config: accel
        prefix: tmp/X
      - component: MAC1
        bindings:
        - op: add

      - component: S2B
        bindings:
        - tensor: A
          rank: M
          format: default
          type: payload
          evict-on: root
        - tensor: X
          rank: M
          format: default
          type: payload
          evict-on: root
    """
    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(1)
    hardware = Hardware(arch, bindings, program)

    mem = DRAMComponent("Memory", 1, {}, bindings.get_component("Memory"))
    s0b = BuffetComponent("S0B", 1, {}, bindings.get_component("S0B"))
    s1b = BuffetComponent("S1B", 1, {}, bindings.get_component("S1B"))
    s2b = BuffetComponent("S2B", 1, {}, bindings.get_component("S2B"))

    assert hardware.get_traffic_path(
        "A", "M", "payload", "default") == [(mem, "lazy"), (s0b, "lazy")]
    assert hardware.get_traffic_path(
        "Z", "M", "payload", "default") == [(mem, "lazy"), (s0b, "lazy")]
    assert hardware.get_traffic_path(
        "Z", "M", "coord", "default") == [(s1b, "lazy")]

    program.add_einsum(0)
    assert hardware.get_traffic_path("B", "M", "payload", "default") == []


def test_get_traffic_eager():
    extensor = "tests/integration/extensor.yaml"
    arch = Architecture.from_file(extensor)
    bindings = Bindings.from_file(extensor)
    program = Program(Einsum.from_file(extensor), Mapping.from_file(extensor))
    program.add_einsum(0)
    hardware = Hardware(arch, bindings, program)

    dram = hardware.get_component("MainMemory")
    llb = hardware.get_component("LLB")

    ranks = ["K2", "M2", "M1", "K1", "M0", "K0"]
    types = [[], [], [], ["coord"], ["coord", "payload"], ["coord", "payload"]]
    llb.expand_eager("Z", "A", "default", ranks, types)

    assert hardware.get_traffic_path(
        "A", "K1", "coord", "default") == [
        (dram, "lazy"), (llb, "lazy")]
    assert hardware.get_traffic_path(
        "A", "K0", "coord", "default") == [
        (dram, "lazy"), (llb, "M0")]


def test_get_prefix():
    gamma = "tests/integration/gamma.yaml"
    arch = Architecture.from_file(gamma)
    bindings = Bindings.from_file(gamma)
    program = Program(Einsum.from_file(gamma), Mapping.from_file(gamma))
    hardware = Hardware(arch, bindings, program)

    assert hardware.get_prefix("T") == "tmp/gamma_T"
    assert hardware.get_prefix("Z") == "tmp/gamma_Z"


def test_get_tree():
    yaml = """
    einsum:
      declaration:
        A: [M]
        Z: [M]
      expressions:
      - Z[m] = A[m]
    """
    arch = Architecture.from_file("tests/integration/test_arch.yaml")
    bindings = Bindings.from_file("tests/integration/test_bindings.yaml")
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    hardware = Hardware(arch, bindings, program)

    regs = BuffetComponent(
        "Registers", 8,
        {},
        bindings.get_component("Registers"))
    mac = ComputeComponent("MAC", 8,
                           {"type": "mul"},
                           bindings.get_component("MAC"))
    pe = Level("PE", 8, {}, [regs, mac], [])

    mem_attrs = {"datawidth": 8, "bandwidth": 128}
    mem = DRAMComponent(
        "Memory",
        1,
        mem_attrs,
        bindings.get_component("Memory"))
    attrs = {"clock_frequency": 10 ** 9}

    tree = Level("System", 1, attrs, [mem], [pe])

    program.add_einsum(0)
    assert hardware.get_tree() == tree
