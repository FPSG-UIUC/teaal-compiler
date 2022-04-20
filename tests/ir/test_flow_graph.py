import networkx as nx
import pytest

from es2hfa.ir.flow_graph import FlowGraph
from es2hfa.ir.iter_graph import IterationGraph
from es2hfa.ir.nodes import *
from es2hfa.ir.program import Program
from es2hfa.parse.einsum import Einsum
from es2hfa.parse.mapping import Mapping


def build_program_no_loops():
    einsum = Einsum.from_file("tests/integration/test_translate_no_loops.yaml")
    mapping = Mapping.from_file(
        "tests/integration/test_translate_no_loops.yaml")

    program = Program(einsum, mapping)
    program.add_einsum(0)

    return program


def build_program_matmul(mapping):
    yaml = """
    einsum:
        declaration:
            A: [K, M]
            B: [K, N]
            Z: [M, N]
        expressions:
            - Z[m, n] = sum(K).(A[k, m] * B[k, n])
    mapping:
    """ + mapping
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    return program


def build_program_conv(mapping):
    yaml = """
    einsum:
        declaration:
            F: [S]
            I: [W]
            O: [Q]
        expressions:
            - O[q] = sum(S).(I[q + s] * F[s])
    mapping:
    """ + mapping
    program = Program(Einsum.from_str(yaml), Mapping.from_str(yaml))
    program.add_einsum(0)

    return program


def test_graph_no_loops():
    program = build_program_no_loops()
    graph = FlowGraph(program).get_graph()

    corr = nx.DiGraph()
    corr.add_edge(OtherNode("Output"), SRNode("A", []))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Graphics"), OtherNode("Body"))
    corr.add_edge(SRNode("A", []), OtherNode("Body"))
    corr.add_edge(OtherNode("Body"), OtherNode("Footer"))

    assert nx.is_isomorphic(graph, corr)


def test_graph():
    program = build_program_matmul("")
    graph = FlowGraph(program).get_graph()

    corr = nx.DiGraph()
    corr.add_edge(OtherNode("Output"), SRNode("Z", ["M", "N"]))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Graphics"), LoopNode("M"))
    corr.add_edge(SRNode("Z", ["M", "N"]), LoopNode("M"))
    corr.add_edge(SRNode("A", ["M", "K"]), LoopNode("M"))
    corr.add_edge(SRNode("B", ["N", "K"]), LoopNode("N"))
    corr.add_edge(LoopNode("M"), LoopNode("N"))
    corr.add_edge(LoopNode("M"), LoopNode("K"))
    corr.add_edge(LoopNode("N"), LoopNode("K"))
    corr.add_edge(LoopNode("N"), OtherNode("Body"))
    corr.add_edge(LoopNode("K"), OtherNode("Body"))
    corr.add_edge(OtherNode("Body"), OtherNode("Footer"))

    assert nx.is_isomorphic(graph, corr)


def test_graph_loop_order():
    spec = """
        loop-order:
            Z: [K, M, N]
    """
    program = build_program_matmul(spec)
    graph = FlowGraph(program).get_graph()

    corr = nx.DiGraph()
    corr.add_edge(OtherNode("Output"), SRNode("Z", ["M", "N"]))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Graphics"), LoopNode("K"))
    corr.add_edge(SRNode("Z", ["M", "N"]), LoopNode("M"))
    corr.add_edge(SRNode("A", ["K", "M"]), LoopNode("K"))
    corr.add_edge(SRNode("B", ["K", "N"]), LoopNode("K"))
    corr.add_edge(LoopNode("K"), LoopNode("M"))
    corr.add_edge(LoopNode("K"), LoopNode("N"))
    corr.add_edge(LoopNode("M"), LoopNode("N"))
    corr.add_edge(LoopNode("M"), OtherNode("Body"))
    corr.add_edge(LoopNode("N"), OtherNode("Body"))
    corr.add_edge(OtherNode("Body"), OtherNode("Footer"))

    assert nx.is_isomorphic(graph, corr)


def test_graph_static_parts():
    spec = """
        partitioning:
            Z:
                K: [uniform_shape(6), uniform_shape(3)]
                N: [uniform_shape(5)]
        loop-order:
            Z: [K2, M, N1, K1, N0, K0]
    """
    program = build_program_matmul(spec)
    graph = FlowGraph(program).get_graph()

    corr = nx.DiGraph()
    corr.add_edge(LoopNode("K2"), LoopNode("M"))
    corr.add_edge(LoopNode("K2"), LoopNode("N1"))
    corr.add_edge(LoopNode("M"), LoopNode("N1"))
    corr.add_edge(LoopNode("M"), LoopNode("K1"))
    corr.add_edge(LoopNode("N1"), LoopNode("K1"))
    corr.add_edge(LoopNode("N1"), LoopNode("N0"))
    corr.add_edge(LoopNode("K1"), LoopNode("N0"))
    corr.add_edge(LoopNode("K1"), LoopNode("K0"))
    corr.add_edge(LoopNode("N0"), LoopNode("K0"))
    corr.add_edge(LoopNode("N0"), OtherNode("Body"))
    corr.add_edge(LoopNode("K0"), OtherNode("Body"))
    corr.add_edge(OtherNode("Graphics"), LoopNode("K2"))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Output"), SRNode("Z", ['M', 'N1', 'N0']))
    corr.add_edge(OtherNode("Body"), OtherNode("Footer"))
    corr.add_edge(SRNode("Z", ['M', 'N1', 'N0']), LoopNode("M"))
    corr.add_edge(PartNode("A", ('K',)), OtherNode("Graphics"))
    corr.add_edge(PartNode("A", ('K',)), SRNode("A", ['K2', 'M', 'K1', 'K0']))
    corr.add_edge(SRNode("A", ['K2', 'M', 'K1', 'K0']), LoopNode("K2"))
    corr.add_edge(PartNode("B", ('K',)), OtherNode("Graphics"))
    corr.add_edge(
        PartNode(
            "B", ('K',)), SRNode(
            "B", [
                'K2', 'N1', 'K1', 'N0', 'K0']))
    corr.add_edge(PartNode("B", ('N',)), OtherNode("Graphics"))
    corr.add_edge(
        PartNode(
            "B", ('N',)), SRNode(
            "B", [
                'K2', 'N1', 'K1', 'N0', 'K0']))
    corr.add_edge(SRNode("B", ['K2', 'N1', 'K1', 'N0', 'K0']), LoopNode("K2"))

    assert nx.is_isomorphic(graph, corr)


def test_graph_dyn_parts():
    spec = """
        partitioning:
            Z:
                K: [uniform_occupancy(A.6), uniform_occupancy(A.3)]
                N: [uniform_occupancy(B.5)]
        loop-order:
            Z: [K2, M, N1, K1, N0, K0]
    """
    program = build_program_matmul(spec)
    graph = FlowGraph(program).get_graph()

    corr = nx.DiGraph()

    corr.add_edge(LoopNode("K2"), LoopNode("M"))
    corr.add_edge(LoopNode("K2"), FromFiberNode("B", "N"))
    corr.add_edge(FromFiberNode("B", "N"), PartNode("B", ('N',)))
    corr.add_edge(LoopNode("M"), LoopNode("N1"))
    corr.add_edge(LoopNode("M"), FromFiberNode("A", "K1I"))
    corr.add_edge(FromFiberNode("A", "K1I"), PartNode("A", ('K1I',)))
    corr.add_edge(LoopNode("N1"), LoopNode("K1"))
    corr.add_edge(LoopNode("N1"), LoopNode("N0"))
    corr.add_edge(LoopNode("N1"), FromFiberNode("B", "K1I"))
    corr.add_edge(FromFiberNode("B", "K1I"), PartNode("B", ('K1I',)))
    corr.add_edge(LoopNode("K1"), LoopNode("N0"))
    corr.add_edge(LoopNode("K1"), LoopNode("K0"))
    corr.add_edge(LoopNode("N0"), LoopNode("K0"))
    corr.add_edge(LoopNode("N0"), OtherNode("Body"))
    corr.add_edge(LoopNode("K0"), OtherNode("Body"))
    corr.add_edge(OtherNode("Graphics"), LoopNode("K2"))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Output"), SRNode("Z", ['M', 'N1', 'N0']))
    corr.add_edge(OtherNode("Body"), OtherNode("Footer"))
    corr.add_edge(SRNode("Z", ['M', 'N1', 'N0']), LoopNode("M"))
    corr.add_edge(PartNode("A", ('K',)), SRNode("A", ['K2', 'M', 'K1I']))
    corr.add_edge(PartNode("A", ('K',)), PartNode("A", ('K1I',)))
    corr.add_edge(PartNode("A", ('K1I',)), SRNode("A", ['K1', 'K0']))
    corr.add_edge(SRNode("A", ['K', 'M']), FromFiberNode("A", "K"))
    corr.add_edge(FromFiberNode("A", "K"), PartNode("A", ('K',)))
    corr.add_edge(SRNode("A", ['K2', 'M', 'K1I']), LoopNode("K2"))
    corr.add_edge(SRNode("A", ['K2', 'M', 'K1I']), PartNode("B", ('K',)))
    corr.add_edge(SRNode("A", ['K1', 'K0']), LoopNode("K1"))
    corr.add_edge(SRNode("A", ['K1', 'K0']), PartNode("B", ('K1I',)))
    corr.add_edge(PartNode("B", ('K',)), SRNode("B", ['K2', 'N', 'K1I']))
    corr.add_edge(PartNode("B", ('K',)), PartNode("B", ('K1I',)))
    corr.add_edge(PartNode("B", ('K1I',)), SRNode("B", ['K1', 'N0', 'K0']))
    corr.add_edge(PartNode("B", ('N',)), SRNode("B", ['N1', 'K1I', 'N0']))
    corr.add_edge(PartNode("B", ('K',)), SRNode("B", ['N1', 'K1I', 'N0']))
    corr.add_edge(PartNode("B", ('N',)), SRNode("B", ['K1', 'N0', 'K0']))
    corr.add_edge(SRNode("B", ['K', 'N']), FromFiberNode("B", "K"))
    corr.add_edge(FromFiberNode("B", "K"), PartNode("B", ('K',)))
    corr.add_edge(SRNode("B", ['K2', 'N', 'K1I']), LoopNode("K2"))
    corr.add_edge(SRNode("B", ['N1', 'K1I', 'N0']), LoopNode("N1"))
    corr.add_edge(SRNode("B", ['K1', 'N0', 'K0']), LoopNode("K1"))

    assert nx.is_isomorphic(graph, corr)


def test_graph_conv():
    spec = """
        loop-order:
            O: [W, Q]
    """
    program = build_program_conv(spec)
    graph = FlowGraph(program).get_graph()

    corr = nx.DiGraph()
    corr.add_edge(LoopNode("W"), LoopNode("Q"))
    corr.add_edge(LoopNode("W"), OtherNode("Body"))
    corr.add_edge(LoopNode("Q"), OtherNode("Body"))
    corr.add_edge(OtherNode("Graphics"), LoopNode("W"))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Output"), SRNode("O", ['Q']))
    corr.add_edge(OtherNode("Body"), OtherNode("Footer"))
    corr.add_edge(SRNode("O", ['Q']), LoopNode("Q"))
    corr.add_edge(SRNode("I", ['W']), LoopNode("W"))
    corr.add_edge(SRNode("F", ['S']), LoopNode("Q"))

    assert nx.is_isomorphic(graph, corr)


def test_build_fiber_nodes_empty_graph():
    program = build_program_no_loops()
    flow_graph = FlowGraph(program)
    iter_graph = IterationGraph(program)

    with pytest.raises(ValueError) as excinfo:
        flow_graph._FlowGraph__build_fiber_nodes(iter_graph)

    assert str(excinfo.value) == "No loop node to connect"
