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


def test_make_get_root():
    hfa = "a_m = A_MK.getRoot()"

    tensor = Tensor("A", ["M", "K"])
    assert Header.make_get_root(tensor).gen(depth=0) == hfa


def test_make_output():
    mapping = """
        partitioning:
            Z:
                M: [uniform_occupancy(A.6)]
        loop-order:
            Z: [M1, K, N, M0]
    """

    hfa = "Z_M1NM0 = Tensor(rank_ids=[\"M1\", \"N\", \"M0\"])"

    header = build_matmul_header(mapping)
    assert header.make_output().gen(depth=0) == hfa


def test_make_output_shape():
    exprs = """
            - Z[m, n] = sum(K).(A[k, m])
    """

    hfa = "Z_MN = Tensor(rank_ids=[\"M\", \"N\"], shape=[M, N])"

    header = build_header(exprs, "")
    assert header.make_output().gen(depth=0) == hfa


def test_make_output_conv_no_shape():
    hfa = "O_Q = Tensor(rank_ids=[\"Q\"])"
    header = build_header_conv("[S, Q]")

    assert header.make_output().gen(0) == hfa


def test_make_output_conv_shape():
    hfa = "O_Q = Tensor(rank_ids=[\"Q\"], shape=[Q])"
    header = build_header_conv("[Q, S]")

    assert header.make_output().gen(0) == hfa


def test_make_swizzle():
    hfa = "A_MK = A_KM.swizzleRanks(rank_ids=[\"M\", \"K\"])"

    header = build_matmul_header("")
    tensor = Tensor("A", ["K", "M"])
    assert header.make_swizzle(tensor).gen(depth=0) == hfa


def test_make_tensor_from_fiber():
    hfa = "A_KM = Tensor.fromFiber(rank_ids=[\"K\", \"M\"], fiber=a_k)"
    tensor = Tensor("A", ["K", "M"])
    assert Header.make_tensor_from_fiber(tensor).gen(depth=0) == hfa
