import pytest

from teaal.ir.iter_graph import IterationGraph
from teaal.ir.program import Program
from teaal.ir.tensor import Tensor
from teaal.parse.einsum import Einsum
from teaal.parse.mapping import Mapping
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

    header = Header(program, Partitioner(program, TransUtils()))

    return header


def build_header_conv(loop_order):
    yaml = """
    einsum:
        declaration:
            F: [S]
            I: [W]
            O: [Q]
        expressions:
            - O[q] = sum(S).(I[q + s] * F[s])
    mapping:
        loop-order:
            O: """ + loop_order

    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    header = Header(program, Partitioner(program, TransUtils()))

    return header


def build_matmul_header(mapping):
    exprs = """
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    """
    return build_header(exprs, mapping)


def test_make_get_payload():
    hifiber = "a_val = a_m.getPayload((m, k))"

    tensor = Tensor("A", ["M", "K"])
    assert Header.make_get_payload(tensor, ["M", "K"]).gen(0) == hifiber


def test_make_get_payload_output():
    hifiber = "z_n = z_m.getPayloadRef((m,))"

    tensor = Tensor("Z", ["M", "N"])
    tensor.set_is_output(True)
    assert Header.make_get_payload(tensor, ["M"]).gen(0) == hifiber


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
            - Z[m, n] = sum(K).(A[k, m])
    """

    hifiber = "Z_MN = Tensor(rank_ids=[\"M\", \"N\"], shape=[M, N])"

    header = build_header(exprs, "")
    assert header.make_output().gen(depth=0) == hifiber


def test_make_output_shape_input_flattening():
    mapping = """
        partitioning:
            Z:
                K: [uniform_shape(4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
        loop-order:
            Z: [K1, MK01, N, MK00]
    """

    hifiber = "Z_NM = Tensor(rank_ids=[\"N\", \"M\"], shape=[N, M])"
    header = build_matmul_header(mapping)
    assert header.make_output().gen(depth=0) == hifiber


def test_make_output_conv_no_shape():
    hifiber = "O_Q = Tensor(rank_ids=[\"Q\"])"
    header = build_header_conv("[S, Q]")

    assert header.make_output().gen(0) == hifiber


def test_make_output_conv_shape():
    hifiber = "O_Q = Tensor(rank_ids=[\"Q\"], shape=[Q])"
    header = build_header_conv("[Q, S]")

    assert header.make_output().gen(0) == hifiber


def test_make_swizzle_bad():
    header = build_matmul_header("")
    tensor = Tensor("A", ["K", "M"])
    with pytest.raises(ValueError) as excinfo:
        header.make_swizzle(tensor, "foo")

    assert str(
        excinfo.value) == "Unknown swizzling reason: foo"


def test_make_swizzle_loop_order():
    hifiber = "A_MK = A_KM.swizzleRanks(rank_ids=[\"M\", \"K\"])"

    header = build_matmul_header("")
    tensor = Tensor("A", ["K", "M"])
    assert header.make_swizzle(tensor, "loop-order").gen(depth=0) == hifiber


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
    assert header.make_swizzle(tensor, "partitioning").gen(depth=0) == hifiber


def test_make_tensor_from_fiber():
    hifiber = "A_KM = Tensor.fromFiber(rank_ids=[\"K\", \"M\"], fiber=a_k)"
    tensor = Tensor("A", ["K", "M"])
    assert Header.make_tensor_from_fiber(tensor).gen(depth=0) == hifiber
