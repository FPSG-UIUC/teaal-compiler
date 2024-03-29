import pytest

from teaal.ir.iter_graph import IterationGraph
from teaal.ir.program import Program
from teaal.ir.tensor import Tensor
from teaal.parse.einsum import Einsum
from teaal.parse.mapping import Mapping


def test_peek_rank0():
    yaml = """
    einsum:
        declaration:
            A: []
        expressions:
            - A[] = b
    """
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)
    graph = IterationGraph(program)

    tensor = Tensor("A", [])
    tensor.set_is_output(True)

    assert graph.peek_concord() == (None, [tensor])


def test_peek_default():
    yaml = """
    einsum:
        declaration:
            A: [I, J]
            B: [I, K]
            C: [J, K]
        expressions:
            - A[i, j] = B[i, k] * C[j, k]
    """
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)
    graph = IterationGraph(program)

    results = [Tensor("A", ["I", "J"]), Tensor("B", ["I", "K"])]
    results[0].set_is_output(True)

    assert graph.peek_concord() == ("I", results)


def test_peek_order():
    yaml = """
    einsum:
        declaration:
            A: [I, J]
            B: [I, K]
            C: [J, K]
        expressions:
            - A[i, j] = B[i, k] * C[j, k]
    mapping:
        loop-order:
            A: [J, K, I]
    """
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    for tensor in program.get_equation().get_tensors():
        program.get_loop_order().apply(tensor)
    graph = IterationGraph(program)

    results = [Tensor("A", ["J", "I"]), Tensor("C", ["J", "K"])]
    results[0].set_is_output(True)

    assert graph.peek_concord() == ("J", results)


def test_peek_discord_bad():
    yaml = """
    einsum:
        declaration:
            A: [K, M]
            B: [K, N]
            Z: [M, N]
        expressions:
            - Z[m, n] = A[k, m] * B[k, n]
    """
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)
    graph = IterationGraph(program)

    with pytest.raises(ValueError) as excinfo:
        graph.peek_discord()
    assert str(
        excinfo.value) == "Can only perform a discordant traversal inside the loop nest"


def test_peek_discord_none():
    yaml = """
    einsum:
        declaration:
            A: [K, M]
            B: [K, N]
            Z: [M, N]
        expressions:
            - Z[m, n] = A[k, m] * B[k, n]
    """
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)
    graph = IterationGraph(program)

    for tensor in program.get_equation().get_tensors():
        program.get_loop_order().apply(tensor)

    graph.pop_concord()
    assert graph.peek_discord() == []

    graph.pop_concord()
    assert graph.peek_discord() == []

    graph.pop_concord()
    assert graph.peek_discord() == []


def test_peek_discord_flatten():
    yaml = """
    einsum:
        declaration:
            A: [K, M]
            B: [J, K, N]
            Z: [M, N]
        expressions:
            - Z[m, n] = A[k, m] * B[j, k, n]
    mapping:
        partitioning:
            Z:
                K: [uniform_shape(4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
        loop-order:
            Z: [K1, MK01, N, MK00, J]
    """
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)
    graph = IterationGraph(program)

    # Apply the partitioning
    A = program.get_equation().get_tensor("A")
    program.apply_all_partitioning(A)
    program.apply_partition_swizzling(A)
    program.apply_all_partitioning(A)
    program.get_loop_order().apply(A)

    B = program.get_equation().get_tensor("B")
    program.apply_all_partitioning(B)
    program.get_loop_order().apply(B)

    Z = program.get_equation().get_tensor("Z")
    program.get_loop_order().apply(Z)

    graph.pop_concord()
    assert graph.peek_discord() == []

    graph.pop_concord()
    assert graph.peek_discord() == []

    graph.pop_concord()
    assert graph.peek_discord() == []

    graph.pop_concord()
    assert graph.peek_discord() == [(["M"], Z), (["K0"], B)]


def test_pop_default():
    yaml = """
    einsum:
        declaration:
            A: [I, J]
            B: [I, K]
            C: [J, K]
        expressions:
            - A[i, j] = B[i, k] * C[j, k]
    """
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)
    graph = IterationGraph(program)

    A = Tensor("A", ["I", "J"])
    A.set_is_output(True)
    B = Tensor("B", ["I", "K"])
    C = Tensor("C", ["J", "K"])

    A.pop()
    B.pop()
    assert graph.pop_concord() == ("I", [A, B])

    A.pop()
    C.pop()
    assert graph.pop_concord() == ("J", [A, C])

    B.pop()
    C.pop()
    assert graph.pop_concord() == ("K", [B, C])

    assert graph.peek_concord() == (None, [A, B, C])


def test_pop_order():
    yaml = """
    einsum:
        declaration:
            A: [I, J]
            B: [I, K]
            C: [J, K]
        expressions:
            - A[i, j] = B[i, k] * C[j, k]
    mapping:
        loop-order:
            A: [J, K, I]
    """
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    for tensor in program.get_equation().get_tensors():
        program.get_loop_order().apply(tensor)
    graph = IterationGraph(program)

    A = Tensor("A", ["J", "I"])
    A.set_is_output(True)
    B = Tensor("B", ["K", "I"])
    C = Tensor("C", ["J", "K"])

    A.pop()
    C.pop()
    assert graph.pop_concord() == ("J", [A, C])

    B.pop()
    C.pop()
    assert graph.pop_concord() == ("K", [B, C])

    A.pop()
    B.pop()
    assert graph.pop_concord() == ("I", [A, B])

    assert graph.peek_concord() == (None, [A, B, C])


def test_pop_occupancy_partitioning():
    yaml = """
    einsum:
        declaration:
            A: [K, M]
            B: [K, N]
            Z: [M, N]
        expressions:
            - Z[m, n] = A[k, m] * B[k, n]
    mapping:
        partitioning:
            Z:
                M: [uniform_occupancy(A.5)]
                N: [uniform_occupancy(B.6)]
        loop-order:
            Z: [M1, N1, K, M0, N0]
    """
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    for tensor in program.get_equation().get_tensors():
        program.get_loop_order().apply(tensor)
    graph = IterationGraph(program)

    # First apply all partitioning/loop order to the output tensor
    output = program.get_equation().get_output()
    program.apply_all_partitioning(output)
    program.get_loop_order().apply(output)

    # Apply the partitioning to the A tensor
    A = program.get_equation().get_tensor("A")
    A.from_fiber()
    program.apply_partitioning(A, ("M",))

    # Make sure that there are no errors on pop
    assert graph.pop_concord()[0] == "M1"


def test_peek_pop_coord_math():
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
            O: [W, Q]
    """

    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)
    graph = IterationGraph(program)

    I = Tensor("I", ["W"])
    F = Tensor("F", ["S"])
    O = Tensor("O", ["Q"])
    O.set_is_output(True)

    assert graph.peek_concord() == ("W", [I])

    I.pop()
    assert graph.pop_concord() == ("W", [I])

    assert graph.peek_concord() == ("Q", [O, F])

    F.pop()
    O.pop()
    assert graph.pop_concord() == ("Q", [O, F])

    assert graph.peek_concord() == (None, [O, I, F])


def test_pop_discord_none():
    yaml = """
    einsum:
        declaration:
            A: [K, M]
            B: [K, N]
            Z: [M, N]
        expressions:
            - Z[m, n] = A[k, m] * B[k, n]
    """
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)
    graph = IterationGraph(program)

    for tensor in program.get_equation().get_tensors():
        program.get_loop_order().apply(tensor)

    graph.pop_concord()
    assert graph.pop_discord() == []

    graph.pop_concord()
    assert graph.pop_discord() == []

    graph.pop_concord()
    assert graph.pop_discord() == []


def test_pop_discord_flatten():
    yaml = """
    einsum:
        declaration:
            A: [K, M]
            B: [J, K, N]
            Z: [M, N]
        expressions:
            - Z[m, n] = A[k, m] * B[j, k, n]
    mapping:
        partitioning:
            Z:
                K: [uniform_shape(4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
        loop-order:
            Z: [K1, MK01, N, MK00, J]
    """
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)
    graph = IterationGraph(program)

    # Apply the partitioning
    A = program.get_equation().get_tensor("A")
    program.apply_all_partitioning(A)
    program.apply_partition_swizzling(A)
    program.apply_all_partitioning(A)
    program.get_loop_order().apply(A)

    B = program.get_equation().get_tensor("B")
    program.apply_all_partitioning(B)
    program.get_loop_order().apply(B)

    Z = program.get_equation().get_tensor("Z")
    program.get_loop_order().apply(Z)

    graph.pop_concord()
    assert graph.pop_discord() == []

    graph.pop_concord()
    assert graph.pop_discord() == []

    graph.pop_concord()
    assert graph.pop_discord() == []

    graph.pop_concord()
    assert Z.peek_rest() == ["M"]
    assert B.peek_rest() == ["K0", "J"]
    assert graph.pop_discord() == [(["M"], Z), (["K0"], B)]
    assert Z.peek_rest() == []
    assert B.peek_rest() == ["J"]
