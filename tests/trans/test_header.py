from es2hfa.ir.iter_graph import IterationGraph
from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.einsum import Einsum
from es2hfa.parse.mapping import Mapping
from es2hfa.trans.header import Header
from es2hfa.trans.partitioner import Partitioner
from es2hfa.trans.utils import TransUtils


def build_header(mapping):
    yaml = """
    einsum:
        declaration:
            Z: [M, N]
            A: [K, M]
            B: [K, N]
            C: [M, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
    """ + mapping

    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    header = Header(program, Partitioner(program, TransUtils()))

    return header, program


def test_make_output():
    mapping = """
        partitioning:
            Z:
                M: [uniform_occupancy(A.6)]
        loop-order:
            Z: [M1, K, N, M0]
    """

    hfa = "Z_M1NM0 = Tensor(rank_ids=[\"M1\", \"N\", \"M0\"])"

    header, program = build_header(mapping)
    assert header.make_output().gen(depth=0) == hfa


def test_make_swizzle_root():
    hfa = "A_MK = A_KM.swizzleRanks(rank_ids=[\"M\", \"K\"])\n" + \
          "a_m = A_MK.getRoot()"

    header, _ = build_header("")
    tensor = Tensor("A", ["K", "M"])
    assert header.make_swizzle_root(tensor).gen(depth=0) == hfa


def test_make_tensor_from_fiber():
    hfa = "A_KM = Tensor.fromFiber(rank_ids=[\"K\", \"M\"], fiber=a_k)"
    tensor = Tensor("A", ["K", "M"])
    assert Header.make_tensor_from_fiber(tensor).gen(depth=0) == hfa
