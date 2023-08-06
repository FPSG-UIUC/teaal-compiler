import pytest

from teaal.ir.hardware import Hardware
from teaal.ir.iter_graph import IterationGraph
from teaal.ir.metrics import Metrics
from teaal.ir.program import Program
from teaal.ir.tensor import Tensor
from teaal.parse import *
from teaal.trans.header import Header
from teaal.trans.partitioner import Partitioner
from teaal.trans.utils import TransUtils


def build_header(exprs, mapping):
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
            C: [M, N]
        expressions:
    """ + exprs + """
    mapping:
    """ + mapping

    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    header = Header(program, None, Partitioner(program, TransUtils(program)))

    return header


def build_header_conv(loop_order):
    yaml = """
    einsum:
        declaration:
            F: [S]
            I: [W]
            O: [Q]
        expressions:
            - O[q] = I[q + s] * F[s]
    mapping:
        loop-order:
            O: """ + loop_order

    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    header = Header(program, None, Partitioner(program, TransUtils(program)))

    return header


def build_header_gamma():
    fname = "tests/integration/gamma.yaml"
    einsum = Einsum.from_file(fname)
    mapping = Mapping.from_file(fname)
    arch = Architecture.from_file(fname)
    bindings = Bindings.from_file(fname)
    format_ = Format.from_file(fname)

    program = Program(einsum, mapping)
    hardware = Hardware(arch, bindings, program)

    program.add_einsum(0)
    metrics = Metrics(program, hardware, format_)

    header = Header(
        program,
        metrics,
        Partitioner(
            program,
            TransUtils(program)))
    return header


def build_matmul_header(mapping):
    exprs = """
            - Z[m, n] = A[k, m] * B[k, n]
    """
    return build_header(exprs, mapping)


def test_make_get_payload():
    header = build_matmul_header("")
    tensor = Tensor("A", ["M", "K"])

    hifiber = "a_val = a_m.getPayload(m, k)"
    assert header.make_get_payload(tensor, ["M", "K"]).gen(0) == hifiber


def test_make_get_payload_output():
    header = build_matmul_header("")
    tensor = Tensor("Z", ["M", "N"])
    tensor.set_is_output(True)

    hifiber = "z_n = z_m.getPayloadRef(m)"
    assert header.make_get_payload(tensor, ["M"]).gen(0) == hifiber


def test_make_get_payload_metrics():
    header = build_header_gamma()
    tensor = Tensor("A", ["M", "K"])

    hifiber = "a_k = a_m.getPayload(m, trace=\"get_payload_A\")"
    assert header.make_get_payload(tensor, ["M"]).gen(0) == hifiber


def test_make_get_root():
    hifiber = "a_m = A_MK.getRoot()"

    tensor = Tensor("A", ["M", "K"])
    assert Header.make_get_root(tensor).gen(depth=0) == hifiber


def test_make_output():
    mapping = """
        partitioning:
            Z:
                M: [uniform_occupancy(A.6)]
        loop-order:
            Z: [M1, K, N, M0]
    """

    hifiber = "Z_M1NM0 = Tensor(rank_ids=[\"M1\", \"N\", \"M0\"])"

    header = build_matmul_header(mapping)
    assert header.make_output().gen(depth=0) == hifiber


def test_make_output_shape():
    exprs = """
            - Z[m, n] = A[k, m]
    """

    hifiber = "Z_MN = Tensor(rank_ids=[\"M\", \"N\"], shape=[M, N])"

    header = build_header(exprs, "")
    assert header.make_output().gen(depth=0) == hifiber


def test_make_output_no_shape_flattening():
    exprs = """
            - Z[m, n] = C[m, n]
    """

    mapping = """
        partitioning:
            Z:
                (M, N): [flatten()]
    """

    hifiber = "Z_MN = Tensor(rank_ids=[\"MN\"])"
    header = build_header(exprs, mapping)
    assert header.make_output().gen(depth=0) == hifiber


def test_make_output_conv_no_shape():
    hifiber = "O_Q = Tensor(rank_ids=[\"Q\"])"
    header = build_header_conv("[S, Q]")

    assert header.make_output().gen(0) == hifiber


def test_make_output_conv_shape():
    hifiber = "O_Q = Tensor(rank_ids=[\"Q\"], shape=[Q])"
    header = build_header_conv("[Q, S]")

    assert header.make_output().gen(0) == hifiber


def test_make_output_metrics_shape():
    hifiber = "T_MKN = Tensor(rank_ids=[\"M\", \"K\", \"N\"], shape=[M, K, N])"
    header = build_header_gamma()

    assert header.make_output().gen(0) == hifiber


def test_make_swizzle_bad():
    header = build_matmul_header("")
    tensor = Tensor("A", ["K", "M"])
    with pytest.raises(ValueError) as excinfo:
        header.make_swizzle(tensor, [], "foo")

    assert str(
        excinfo.value) == "Unknown swizzling reason: foo"


def test_make_swizzle_loop_order():
    hifiber = "A_MK = A_KM.swizzleRanks(rank_ids=[\"M\", \"K\"])"

    header = build_matmul_header("")
    tensor = Tensor("A", ["K", "M"])
    assert header.make_swizzle(
        tensor, ["M", "K"], "loop-order").gen(depth=0) == hifiber


def test_make_swizzle_none():
    hifiber = ""

    mapping = """
      rank-order:
        A: [M, K]
    """

    header = build_matmul_header(mapping)
    tensor = Tensor("A", ["M", "K"])
    assert header.make_swizzle(
        tensor, ["M", "K"], "loop-order").gen(depth=0) == hifiber


def test_make_swizzle_partitioning():
    hifiber = "A_K1MK0 = A_K1K0M.swizzleRanks(rank_ids=[\"K1\", \"M\", \"K0\"])"

    mapping = """
        partitioning:
            Z:
                K: [uniform_shape(4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
    """

    header = build_matmul_header(mapping)
    tensor = Tensor("A", ["K1", "K0", "M"])
    assert header.make_swizzle(
        tensor, [
            "M", "K0"], "partitioning").gen(
        depth=0) == hifiber


def test_make_swizzle_metrics():
    hifiber = "A_KM = A_MK.swizzleRanks(rank_ids=[\"K\", \"M\"])"

    header = build_matmul_header("")
    tensor = Tensor("A", ["M", "K"])
    assert header.make_swizzle(
        tensor, [
            "K", "M"], "metrics").gen(
        depth=0) == hifiber


def test_make_tensor_from_fiber():
    hifiber = "A_KM = Tensor.fromFiber(rank_ids=[\"K\", \"M\"], fiber=a_k)"
    tensor = Tensor("A", ["K", "M"])
    assert Header.make_tensor_from_fiber(tensor).gen(depth=0) == hifiber
