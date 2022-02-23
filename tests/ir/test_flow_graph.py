from es2hfa.ir.flow_graph import FlowGraph
from es2hfa.ir.nodes import *
from es2hfa.ir.program import Program
from es2hfa.parse.einsum import Einsum
from es2hfa.parse.mapping import Mapping


def build_program(mapping):
    yaml = """
    einsum:
        declaration:
            A: [K, M]
            B: [K, N]
            Z: [M, N]
        expressions:
            - Z[M, N] = sum(K).(A[K, M] * B[K, N])
    mapping:
    """ + mapping
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    return program


def test_graph():
    program = build_program("")
    graph = FlowGraph(program)
    sort = graph.sort()

    # We need SRNodes for each tensor
    assert SRNode("A", {"K", "M"}) in sort
    assert SRNode("B", {"K", "N"}) in sort
    assert SRNode("Z", {"M", "N"}) in sort

    # The loop nodes must be last
    assert sort[3] == LoopNode("M")
    assert sort[4] == LoopNode("N")
    assert sort[5] == LoopNode("K")
