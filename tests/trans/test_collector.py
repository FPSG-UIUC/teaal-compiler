import pytest

from es2hfa.ir.hardware import Hardware
from es2hfa.ir.metrics import Metrics
from es2hfa.ir.program import Program
from es2hfa.parse import *
from es2hfa.trans.collector import Collector


def build_gamma_yaml():
    with open("tests/integration/gamma.yaml", "r") as f:
        return f.read()


def build_collector(yaml, i):
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)

    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    hardware = Hardware(arch, bindings)

    format_ = Format.from_str(yaml)

    program.add_einsum(i)
    metrics = Metrics(program, hardware, format_)
    return Collector(program, metrics)


def test_dump():
    yaml = build_gamma_yaml()
    collector = build_collector(yaml, 0)
    hfa = "metrics = {}\n" + \
          "metrics[\"T\"] = {}\n" + \
          "metrics[\"T\"][\"T footprint\"] = 0\n" + \
          "metrics[\"T\"][\"T traffic\"] = 0\n" + \
          "A_MK_format = Format(A_MK, {\"M\": {\"format\": \"U\", \"rhbits\": 32, \"pbits\": 32}, \"K\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}})\n" + \
          "metrics[\"T\"][\"A footprint\"] = A_MK_format.getTensor()\n" + \
          "metrics[\"T\"][\"A traffic\"] = metrics[\"T\"][\"A footprint\"]\n" + \
          "B_KN_format = Format(B_KN, {\"K\": {\"format\": \"U\", \"rhbits\": 32, \"pbits\": 32}, \"N\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}})\n" + \
          "metrics[\"T\"][\"B footprint\"] = B_KN_format.getTensor()\n" + \
          "metrics[\"T\"][\"B traffic\"] = Traffic.cacheTraffic(B_KN, \"K\", B_KN_format, 25165824) + B_KN_format.getRank(\"K\")\n" + \
          "metrics[\"T\"][\"K intersections\"] = Compute.lfCount(Metrics.dump(), \"K\", 0)"

    assert collector.dump().gen(0) == hfa

    collector = build_collector(yaml, 1)
    hfa = "metrics[\"Z\"] = {}\n" + \
          "Z_MN_format = Format(Z_MN, {\"M\": {\"format\": \"U\", \"rhbits\": 32, \"pbits\": 32}, \"N\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}})\n" + \
          "metrics[\"Z\"][\"Z footprint\"] = Z_MN_format.getTensor()\n" + \
          "metrics[\"Z\"][\"Z traffic\"] = metrics[\"Z\"][\"Z footprint\"]\n" + \
          "metrics[\"Z\"][\"T footprint\"] = 0\n" + \
          "metrics[\"Z\"][\"T traffic\"] = 0\n" + \
          "A_MK_format = Format(A_MK, {\"M\": {\"format\": \"U\", \"rhbits\": 32, \"pbits\": 32}, \"K\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}})\n" + \
          "metrics[\"Z\"][\"A footprint\"] = A_MK_format.getTensor()\n" + \
          "metrics[\"Z\"][\"A traffic\"] = metrics[\"Z\"][\"A footprint\"]\n" + \
          "metrics[\"Z\"][\"mul\"] = Compute.opCount(Metrics.dump(), \"mul\")\n" + \
          "metrics[\"Z\"][\"add\"] = Compute.opCount(Metrics.dump(), \"add\")\n" + \
          "metrics[\"Z\"][\"T_MKN merge ops\"] = Compute.swapCount(T_MKN, 1, 64, 1)"

    assert collector.dump().gen(0) == hfa


def test_dump_buffet():
    yaml = """
    einsum:
      declaration:
        A: [M]
        Z: [M, N]
      expressions:
        - Z[m, n] = A[m]

    mapping:
      loop-order:
        Z: [N, M]

    architecture:
      subtree:
      - name: System
        local:
        - name: Memory
          class: DRAM

        subtree:
        - name: PE

          local:
          - name: RegFile
            class: Buffet

          - name: Compute
            class: Compute

    bindings:
    - name: Memory
      bindings:
      - tensor: A
        rank: root

    - name: RegFile
      bindings:
      - tensor: A
        rank: M

    - name: Compute
      bindings:
      - einsum: Z
        op: add

    format:
      A:
        M:
          format: C
          cbits: 32
          pbits: 64
    """
    collector = build_collector(yaml, 0)
    hfa = "metrics = {}\n" + \
          "metrics[\"Z\"] = {}\n" + \
          "metrics[\"Z\"][\"Z footprint\"] = 0\n" + \
          "metrics[\"Z\"][\"Z traffic\"] = 0\n" + \
          "A_M_format = Format(A_M, {\"M\": {\"format\": \"C\", \"cbits\": 32, \"pbits\": 64}})\n" + \
          "metrics[\"Z\"][\"A footprint\"] = A_M_format.getTensor()\n" + \
          "metrics[\"Z\"][\"A traffic\"] = Traffic.buffetTraffic(A_M, \"M\", A_M_format) + A_M_format.getRank(\"M\")\n" + \
          "metrics[\"Z\"][\"add\"] = Compute.opCount(Metrics.dump(), \"add\")"

    assert collector.dump().gen(0) == hfa

def test_dump_leader_follower_bad_rank():
    yaml = """
    einsum:
      declaration:
        A: [K, M]
        B: [K]
        Z: [M]
      expressions:
        - Z[m] = sum(K).(A[k, m] * B[m])

    architecture:
      subtree:
      - name: System
        local:
        - name: Intersect
          class: LeaderFollower

    bindings:
    - name: Intersect
      bindings:
      - einsum: Z
        rank: P
        leader: B
    """
    collector = build_collector(yaml, 0)

    with pytest.raises(ValueError) as excinfo:
        collector.dump()
    assert str(excinfo.value) == "Tensor B has no rank P"

def test_dump_leader_follower():
    yaml = """
    einsum:
      declaration:
        A: [K, M]
        B: [K]
        Z: [M]
      expressions:
        - Z[m] = sum(K).(A[k, m] * B[m])

    architecture:
      subtree:
      - name: System
        local:
        - name: Intersect
          class: LeaderFollower

    bindings:
    - name: Intersect
      bindings:
      - einsum: Z
        rank: K
        leader: B
    """
    collector = build_collector(yaml, 0)

    hfa = "metrics = {}\n" + \
          "metrics[\"Z\"] = {}\n" + \
          "metrics[\"Z\"][\"Z footprint\"] = 0\n" + \
          "metrics[\"Z\"][\"Z traffic\"] = 0\n" + \
          "metrics[\"Z\"][\"A footprint\"] = 0\n" + \
          "metrics[\"Z\"][\"A traffic\"] = 0\n" + \
          "metrics[\"Z\"][\"B footprint\"] = 0\n" + \
          "metrics[\"Z\"][\"B traffic\"] = 0\n" + \
          "metrics[\"Z\"][\"K intersections\"] = Compute.lfCount(Metrics.dump(), \"K\", 1)"

    assert collector.dump().gen(0) == hfa

def test_dump_skip_ahead():
    yaml = """
    einsum:
      declaration:
        A: [K, M]
        B: [K]
        Z: [M]
      expressions:
        - Z[m] = sum(K).(A[k, m] * B[m])

    architecture:
      subtree:
      - name: System
        local:
        - name: Intersect
          class: SkipAhead

    bindings:
    - name: Intersect
      bindings:
      - einsum: Z
        rank: K
    """
    collector = build_collector(yaml, 0)

    hfa = "metrics = {}\n" + \
          "metrics[\"Z\"] = {}\n" + \
          "metrics[\"Z\"][\"Z footprint\"] = 0\n" + \
          "metrics[\"Z\"][\"Z traffic\"] = 0\n" + \
          "metrics[\"Z\"][\"A footprint\"] = 0\n" + \
          "metrics[\"Z\"][\"A traffic\"] = 0\n" + \
          "metrics[\"Z\"][\"B footprint\"] = 0\n" + \
          "metrics[\"Z\"][\"B traffic\"] = 0\n" + \
          "metrics[\"Z\"][\"K intersections\"] = Compute.skipCount(Metrics.dump(), \"K\")"

    assert collector.dump().gen(0) == hfa

# def test_dump_leader_follower_not_intersected():
#     yaml = """
#     einsum:
#       declaration:
#         A: [M]
#         B: [K]
#         C: [K, M]
#         Z: [M]
#       expressions:
#         - Z[m] = sum(K).(A[k, m] * B[m] * C[k, m])
#
#     architecture:
#       subtree:
#       - name: System
#         local:
#         - name: Intersect
#           class: LeaderFollower
#
#     bindings:
#     - name: Intersect
#       bindings:
#       - einsum: Z
#         rank: K
#         leader: C
#     """
#     collector = build_collector(yaml, 0)
#
#     hfa = "metrics = {}\n" + \
#           "metrics[\"Z\"] = {}\n" + \
#           "metrics[\"Z\"][\"Z footprint\"] = 0\n" + \
#           "metrics[\"Z\"][\"Z traffic\"] = 0\n" + \
#           "metrics[\"Z\"][\"A footprint\"] = 0\n" + \
#           "metrics[\"Z\"][\"A traffic\"] = 0\n" + \
#           "metrics[\"Z\"][\"B footprint\"] = 0\n" + \
#           "metrics[\"Z\"][\"B traffic\"] = 0\n" + \
#           "metrics[\"Z\"][\"C footprint\"] = 0\n" + \
#           "metrics[\"Z\"][\"C traffic\"] = 0\n" + \
#           "metrics[\"Z\"][\"K intersections\"] = Compute.lfCount(Metrics.dump(), \"K\", 1)"
#
#     assert collector.dump().gen(0) == hfa


def test_end():
    hfa = "Metrics.endCollect()"

    assert Collector.end().gen(0) == hfa


def test_set_collecting():
    yaml = build_gamma_yaml()
    collector = build_collector(yaml, 0)
    hfa = "B_KN.setCollecting(\"K\", True)"

    assert collector.set_collecting("B", "K").gen(0) == hfa


def test_start():
    yaml = build_gamma_yaml()
    collector = build_collector(yaml, 0)
    hfa = "Metrics.beginCollect([\"M\", \"K\", \"N\"])"

    assert collector.start().gen(0) == hfa
