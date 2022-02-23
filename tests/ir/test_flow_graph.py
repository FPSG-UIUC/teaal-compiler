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
    assert SRNode("A", ["M", "K"]) in sort
    assert SRNode("B", ["N", "K"]) in sort
    assert SRNode("Z", ["M", "N"]) in sort

    # The loop nodes must be last
    assert sort[3] == LoopNode("M")
    assert sort[4] == LoopNode("N")
    assert sort[5] == LoopNode("K")


def test_graph_loop_order():
    spec = """
        loop-order:
            Z: [K, M, N]
    """
    program = build_program(spec)
    graph = FlowGraph(program)
    sort = graph.sort()

    # We need SRNodes for each tensor
    assert SRNode("A", ["K", "M"]) in sort
    assert SRNode("B", ["K", "N"]) in sort
    assert SRNode("Z", ["M", "N"]) in sort

    # The loop nodes must be last
    assert sort[3] == LoopNode("K")
    assert sort[4] == LoopNode("M")
    assert sort[5] == LoopNode("N")


def test_graph_static_parts():
    spec = """
        partitioning:
            Z:
                K: [uniform_shape(6), uniform_shape(3)]
        loop-order:
            Z: [K2, M, K1, N, K0]
    """
    program = build_program(spec)
    graph = FlowGraph(program)
    sort = graph.sort()

    # The PartNode must be before the SRNode
    assert sort.index(
        PartNode(
            "A", ("K",))) < sort.index(
        SRNode(
            "A", [
                "K2", "M", "K1", "K0"]))
    assert sort.index(
        PartNode(
            "B", ("K",))) < sort.index(
        SRNode(
            "B", [
                "K2", "K1", "N", "K0"]))
    assert SRNode("Z", ["M", "N"]) in sort

    # The loop nodes must be last
    assert sort[5] == LoopNode("K2")
    assert sort[6] == LoopNode("M")
    assert sort[7] == LoopNode("K1")
    assert sort[8] == LoopNode("N")
    assert sort[9] == LoopNode("K0")


def test_graph_dyn_parts():
    spec = """
        partitioning:
            Z:
                K: [uniform_occupancy(A.6), uniform_occupancy(A.3)]
                N: [uniform_occupancy(B.5)]
        loop-order:
            Z: [K2, M, N1, K1, N0, K0]
    """
    program = build_program(spec)
    graph = FlowGraph(program)
    sort = graph.sort()

    # This ordering has been checked, but is subject to change
    corr = [
        SRNode(
            "B", [
                'K', 'N']), SRNode(
            "A", [
                'K', 'M']), PartNode(
            "A", ('K',)), SRNode(
            "A", [
                'K2', 'M', 'K1I']), PartNode(
            "B", ('K',)), SRNode(
            "B", [
                'K2', 'N', 'K1I']), SRNode(
            "Z", [
                'M', 'N']), LoopNode("K2"), PartNode(
            "B", ('N',)), SRNode(
            "B", [
                'N1', 'K1I', 'N0']), LoopNode("M"), PartNode(
            "A", ('K1I',)), SRNode(
            "A", [
                'K1', 'K0']), PartNode(
            "Z", ('N',)), SRNode(
            "Z", [
                'N1', 'N0']), LoopNode("N1"), PartNode(
            "B", ('K1I',)), SRNode(
            "B", [
                'K1', 'N0', 'K0']), LoopNode("K1"), LoopNode("N0"), LoopNode("K0")]
    assert sort == corr
