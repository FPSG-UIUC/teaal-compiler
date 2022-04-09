import pytest

from es2hfa.ir.iter_graph import IterationGraph
from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.einsum import Einsum
from es2hfa.parse.mapping import Mapping


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

    assert graph.peek() == (None, [tensor])


def test_peek_default():
    yaml = """
    einsum:
        declaration:
            A: [I, J]
            B: [I, K]
            C: [J, K]
        expressions:
            - A[i, j] = sum(K).(B[i, k] * C[j, k])
    """
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)
    graph = IterationGraph(program)

    results = [Tensor("A", ["I", "J"]), Tensor("B", ["I", "K"])]
    results[0].set_is_output(True)

    assert graph.peek() == ("I", results)


def test_peek_order():
    yaml = """
    einsum:
        declaration:
            A: [I, J]
            B: [I, K]
            C: [J, K]
        expressions:
            - A[i, j] = sum(K).(B[i, k] * C[j, k])
    mapping:
        loop-order:
            A: [J, K, I]
    """
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    for tensor in program.get_tensors():
        program.get_loop_order().apply(tensor)
    graph = IterationGraph(program)

    results = [Tensor("A", ["J", "I"]), Tensor("C", ["J", "K"])]
    results[0].set_is_output(True)

    assert graph.peek() == ("J", results)


def test_pop_default():
    yaml = """
    einsum:
        declaration:
            A: [I, J]
            B: [I, K]
            C: [J, K]
        expressions:
            - A[i, j] = sum(K).(B[i, k] * C[j, k])
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
    assert graph.pop() == ("I", [A, B])

    A.pop()
    C.pop()
    assert graph.pop() == ("J", [A, C])

    B.pop()
    C.pop()
    assert graph.pop() == ("K", [B, C])

    assert graph.peek() == (None, [A, B, C])


def test_pop_order():
    yaml = """
    einsum:
        declaration:
            A: [I, J]
            B: [I, K]
            C: [J, K]
        expressions:
            - A[i, j] = sum(K).(B[i, k] * C[j, k])
    mapping:
        loop-order:
            A: [J, K, I]
    """
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    for tensor in program.get_tensors():
        program.get_loop_order().apply(tensor)
    graph = IterationGraph(program)

    A = Tensor("A", ["J", "I"])
    A.set_is_output(True)
    B = Tensor("B", ["K", "I"])
    C = Tensor("C", ["J", "K"])

    A.pop()
    C.pop()
    assert graph.pop() == ("J", [A, C])

    B.pop()
    C.pop()
    assert graph.pop() == ("K", [B, C])

    A.pop()
    B.pop()
    assert graph.pop() == ("I", [A, B])

    assert graph.peek() == (None, [A, B, C])


def test_pop_occupancy_partitioning():
    yaml = """
    einsum:
        declaration:
            A: [K, M]
            B: [K, N]
            Z: [M, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
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

    for tensor in program.get_tensors():
        program.get_loop_order().apply(tensor)
    graph = IterationGraph(program)

    # First apply all partitioning/loop order to the output tensor
    output = program.get_output()
    program.apply_all_partitioning(output)
    program.get_loop_order().apply(output)

    # Apply the partitioning to the A tensor
    A = program.get_tensor("A")
    A.from_fiber()
    program.apply_partitioning(A, "M")

    # Make sure that there are no errors on pop
    assert graph.pop()[0] == "M1"


def test_peek_pop_index_math():
    yaml = """
    einsum:
        declaration:
            F: [S]
            I: [W]
            O: [Q]
        expressions:
            - O[q] = sum(S).(I[q + s] + F[s])
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

    assert graph.peek() == ("W", [I])

    I.pop()
    assert graph.pop() == ("W", [I])

    assert graph.peek() == ("Q", [O, F])

    F.pop()
    O.pop()
    assert graph.pop() == ("Q", [O, F])

    assert graph.peek() == (None, [O, I, F])
