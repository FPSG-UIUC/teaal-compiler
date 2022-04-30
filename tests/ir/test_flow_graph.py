import networkx as nx
import pytest

from es2hfa.ir.flow_graph import FlowGraph
from es2hfa.ir.hardware import Hardware
from es2hfa.ir.iter_graph import IterationGraph
from es2hfa.ir.metrics import Metrics
from es2hfa.ir.nodes import *
from es2hfa.ir.program import Program
from es2hfa.parse.arch import Architecture
from es2hfa.parse.bindings import Bindings
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


def build_gamma():
    with open("tests/integration/gamma.yaml", "r") as f:
        yaml = f.read()

    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)

    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    hardware = Hardware(arch, bindings)

    return program, hardware


def test_graph_no_loops():
    program = build_program_no_loops()
    graph = FlowGraph(program, None).get_graph()

    corr = nx.DiGraph()
    corr.add_edge(OtherNode("Output"), GetRootNode("A", []))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Graphics"), OtherNode("Body"))
    corr.add_edge(GetRootNode("A", []), OtherNode("Body"))
    corr.add_edge(OtherNode("Body"), OtherNode("Footer"))

    assert nx.is_isomorphic(graph, corr)


def test_graph():
    program = build_program_matmul("")
    graph = FlowGraph(program, None).get_graph()

    corr = nx.DiGraph()
    corr.add_edge(OtherNode("Output"), GetRootNode("Z", ["M", "N"]))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Graphics"), LoopNode("M"))
    corr.add_edge(GetRootNode("Z", ["M", "N"]), LoopNode("M"))
    corr.add_edge(GetRootNode("A", ["M", "K"]), LoopNode("M"))
    corr.add_edge(GetRootNode("B", ["N", "K"]), LoopNode("N"))
    corr.add_edge(LoopNode("M"), LoopNode("N"))
    corr.add_edge(LoopNode("M"), LoopNode("K"))
    corr.add_edge(LoopNode("N"), LoopNode("K"))
    corr.add_edge(LoopNode("N"), OtherNode("Body"))
    corr.add_edge(LoopNode("K"), OtherNode("Body"))
    corr.add_edge(OtherNode("Body"), OtherNode("Footer"))
    corr.add_edge(SwizzleNode("A", ["M", "K"]), GetRootNode("A", ["M", "K"]))
    corr.add_edge(SwizzleNode("B", ["N", "K"]), GetRootNode("B", ["N", "K"]))

    assert nx.is_isomorphic(graph, corr)


def test_graph_loop_order():
    spec = """
        loop-order:
            Z: [K, M, N]
    """
    program = build_program_matmul(spec)
    graph = FlowGraph(program, None).get_graph()

    corr = nx.DiGraph()
    corr.add_edge(OtherNode("Output"), GetRootNode("Z", ["M", "N"]))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Graphics"), LoopNode("K"))
    corr.add_edge(GetRootNode("Z", ["M", "N"]), LoopNode("M"))
    corr.add_edge(GetRootNode("A", ["K", "M"]), LoopNode("K"))
    corr.add_edge(GetRootNode("B", ["K", "N"]), LoopNode("K"))
    corr.add_edge(LoopNode("K"), LoopNode("M"))
    corr.add_edge(LoopNode("K"), LoopNode("N"))
    corr.add_edge(LoopNode("M"), LoopNode("N"))
    corr.add_edge(LoopNode("M"), OtherNode("Body"))
    corr.add_edge(LoopNode("N"), OtherNode("Body"))
    corr.add_edge(OtherNode("Body"), OtherNode("Footer"))
    corr.add_edge(SwizzleNode("A", ["K", "M"]), GetRootNode("A", ["K", "M"]))
    corr.add_edge(SwizzleNode("B", ["K", "N"]), GetRootNode("B", ["K", "N"]))

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
    graph = FlowGraph(program, None).get_graph()

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
    corr.add_edge(OtherNode("Output"), GetRootNode("Z", ['M', 'N1', 'N0']))
    corr.add_edge(OtherNode("Body"), OtherNode("Footer"))
    corr.add_edge(GetRootNode("Z", ['M', 'N1', 'N0']), LoopNode("M"))
    corr.add_edge(PartNode("A", ('K',)), OtherNode("Graphics"))
    corr.add_edge(
        PartNode(
            "A", ('K',)), SwizzleNode(
            "A", [
                'K2', 'M', 'K1', 'K0']))
    corr.add_edge(GetRootNode("A", ['K2', 'M', 'K1', 'K0']), LoopNode("K2"))
    corr.add_edge(PartNode("B", ('K',)), OtherNode("Graphics"))
    corr.add_edge(
        PartNode(
            "B", ('K',)), SwizzleNode(
            "B", [
                'K2', 'N1', 'K1', 'N0', 'K0']))
    corr.add_edge(PartNode("B", ('N',)), OtherNode("Graphics"))
    corr.add_edge(
        PartNode(
            "B", ('N',)), SwizzleNode(
            "B", [
                'K2', 'N1', 'K1', 'N0', 'K0']))
    corr.add_edge(
        GetRootNode(
            "B", [
                'K2', 'N1', 'K1', 'N0', 'K0']), LoopNode("K2"))
    corr.add_edge(
        SwizzleNode(
            "A", [
                "K2", "M", "K1", "K0"]), GetRootNode(
            "A", [
                "K2", "M", "K1", "K0"]))
    corr.add_edge(
        SwizzleNode(
            "B", [
                "K2", "N1", "K1", "N0", "K0"]), GetRootNode(
            "B", [
                "K2", "N1", "K1", "N0", "K0"]))

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
    graph = FlowGraph(program, None).get_graph()

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
    corr.add_edge(OtherNode("Output"), GetRootNode("Z", ['M', 'N1', 'N0']))
    corr.add_edge(OtherNode("Body"), OtherNode("Footer"))
    corr.add_edge(GetRootNode("Z", ['M', 'N1', 'N0']), LoopNode("M"))
    corr.add_edge(PartNode("A", ('K',)), SwizzleNode("A", ['K2', 'M', 'K1I']))
    corr.add_edge(PartNode("A", ('K',)), PartNode("A", ('K1I',)))
    corr.add_edge(PartNode("A", ('K1I',)), SwizzleNode("A", ['K1', 'K0']))
    corr.add_edge(GetRootNode("A", ['K', 'M']), FromFiberNode("A", "K"))
    corr.add_edge(FromFiberNode("A", "K"), PartNode("A", ('K',)))
    corr.add_edge(GetRootNode("A", ['K2', 'M', 'K1I']), LoopNode("K2"))
    corr.add_edge(GetRootNode("A", ['K2', 'M', 'K1I']), PartNode("B", ('K',)))
    corr.add_edge(GetRootNode("A", ['K1', 'K0']), LoopNode("K1"))
    corr.add_edge(GetRootNode("A", ['K1', 'K0']), PartNode("B", ('K1I',)))
    corr.add_edge(PartNode("B", ('K',)), SwizzleNode("B", ['K2', 'N', 'K1I']))
    corr.add_edge(PartNode("B", ('K',)), PartNode("B", ('K1I',)))
    corr.add_edge(
        PartNode(
            "B", ('K1I',)), SwizzleNode(
            "B", [
                'K1', 'N0', 'K0']))
    corr.add_edge(PartNode("B", ('N',)), SwizzleNode("B", ['N1', 'K1I', 'N0']))
    corr.add_edge(PartNode("B", ('K',)), SwizzleNode("B", ['N1', 'K1I', 'N0']))
    corr.add_edge(PartNode("B", ('N',)), SwizzleNode("B", ['K1', 'N0', 'K0']))
    corr.add_edge(GetRootNode("B", ['K', 'N']), FromFiberNode("B", "K"))
    corr.add_edge(FromFiberNode("B", "K"), PartNode("B", ('K',)))
    corr.add_edge(GetRootNode("B", ['K2', 'N', 'K1I']), LoopNode("K2"))
    corr.add_edge(GetRootNode("B", ['N1', 'K1I', 'N0']), LoopNode("N1"))
    corr.add_edge(GetRootNode("B", ['K1', 'N0', 'K0']), LoopNode("K1"))
    corr.add_edge(SwizzleNode("A", ["K", "M"]), GetRootNode("A", ["K", "M"]))
    corr.add_edge(SwizzleNode("B", ["K", "N"]), GetRootNode("B", ["K", "N"]))
    corr.add_edge(
        SwizzleNode(
            "A", [
                "K2", "M", "K1I"]), GetRootNode(
            "A", [
                "K2", "M", "K1I"]))
    corr.add_edge(
        SwizzleNode(
            "B", [
                "K2", "N", "K1I"]), GetRootNode(
            "B", [
                "K2", "N", "K1I"]))
    corr.add_edge(
        SwizzleNode(
            "B", [
                "N1", "K1I", "N0"]), GetRootNode(
            "B", [
                "N1", "K1I", "N0"]))
    corr.add_edge(
        SwizzleNode(
            "A", [
                "K1", "K0"]), GetRootNode(
            "A", [
                "K1", "K0"]))
    corr.add_edge(
        SwizzleNode(
            "B", [
                "K1", "N0", "K0"]), GetRootNode(
            "B", [
                "K1", "N0", "K0"]))

    assert nx.is_isomorphic(graph, corr)


def test_graph_mixed_parts():
    spec = """
        partitioning:
            Z:
                K: [uniform_shape(20), uniform_occupancy(A.6), uniform_occupancy(A.3)]
        loop-order:
            Z: [K3, M, K2, K1, N, K0]
    """
    program = build_program_matmul(spec)
    graph = FlowGraph(program, None).get_graph()

    corr = nx.DiGraph()

    corr.add_edge(LoopNode("K3"), LoopNode("M"))
    corr.add_edge(LoopNode("K3"), FromFiberNode("B", "K2I"))
    corr.add_edge(LoopNode("M"), LoopNode("K2"))
    corr.add_edge(LoopNode("M"), LoopNode("N"))
    corr.add_edge(LoopNode("M"), FromFiberNode("A", "K2I"))
    corr.add_edge(LoopNode("K2"), LoopNode("K1"))
    corr.add_edge(LoopNode("K2"), FromFiberNode("A", "K1I"))
    corr.add_edge(LoopNode("K2"), FromFiberNode("B", "K1I"))
    corr.add_edge(LoopNode("K1"), LoopNode("N"))
    corr.add_edge(LoopNode("K1"), LoopNode("K0"))
    corr.add_edge(LoopNode("N"), LoopNode("K0"))
    corr.add_edge(LoopNode("N"), OtherNode("Body"))
    corr.add_edge(LoopNode("K0"), OtherNode("Body"))
    corr.add_edge(OtherNode("Graphics"), LoopNode("K3"))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Output"), GetRootNode("Z", ["M", "N"]))
    corr.add_edge(OtherNode("Body"), OtherNode("Footer"))
    corr.add_edge(GetRootNode("Z", ["M", "N"]), LoopNode("M"))
    corr.add_edge(PartNode("A", ("K",)), OtherNode("Graphics"))
    corr.add_edge(PartNode("A", ("K",)), PartNode("A", ("K2I",)))
    corr.add_edge(PartNode("A", ("K",)), SwizzleNode("A", ["K3", "M", "K2I"]))
    corr.add_edge(PartNode("A", ("K2I",)), PartNode("A", ("K1I",)))
    corr.add_edge(PartNode("A", ("K2I",)), SwizzleNode("A", ["K2", "K1I"]))
    corr.add_edge(PartNode("A", ("K1I",)), SwizzleNode("A", ["K1", "K0"]))
    corr.add_edge(GetRootNode("A", ["K3", "M", "K2I"]), LoopNode("K3"))
    corr.add_edge(PartNode("B", ("K",)), OtherNode("Graphics"))
    corr.add_edge(PartNode("B", ("K",)), PartNode("B", ("K2I",)))
    corr.add_edge(PartNode("B", ("K",)), SwizzleNode("B", ["K3", "K2I", "N"]))
    corr.add_edge(PartNode("B", ("K2I",)), PartNode("B", ("K1I",)))
    corr.add_edge(
        PartNode(
            "B", ("K2I",)), SwizzleNode(
            "B", [
                "K2", "K1I", "N"]))
    corr.add_edge(PartNode("B", ("K1I",)), SwizzleNode("B", ["K1", "N", "K0"]))
    corr.add_edge(GetRootNode("B", ["K3", "K2I", "N"]), LoopNode("K3"))
    corr.add_edge(FromFiberNode("B", "K2I"), PartNode("B", ("K2I",)))
    corr.add_edge(GetRootNode("B", ["K2", "K1I", "N"]), LoopNode("K2"))
    corr.add_edge(FromFiberNode("A", "K2I"), PartNode("A", ("K2I",)))
    corr.add_edge(GetRootNode("A", ["K2", "K1I"]), PartNode("B", ("K2I",)))
    corr.add_edge(GetRootNode("A", ["K2", "K1I"]), LoopNode("K2"))
    corr.add_edge(FromFiberNode("A", "K1I"), PartNode("A", ("K1I",)))
    corr.add_edge(GetRootNode("A", ["K1", "K0"]), PartNode("B", ("K1I",)))
    corr.add_edge(GetRootNode("A", ["K1", "K0"]), LoopNode("K1"))
    corr.add_edge(FromFiberNode("B", "K1I"), PartNode("B", ("K1I",)))
    corr.add_edge(GetRootNode("B", ["K1", "N", "K0"]), LoopNode("K1"))
    corr.add_edge(
        SwizzleNode(
            "A", [
                "K3", "M", "K2I"]), GetRootNode(
            "A", [
                "K3", "M", "K2I"]))
    corr.add_edge(
        SwizzleNode(
            "B", [
                "K3", "K2I", "N"]), GetRootNode(
            "B", [
                "K3", "K2I", "N"]))
    corr.add_edge(
        SwizzleNode(
            "B", [
                "K2", "K1I", "N"]), GetRootNode(
            "B", [
                "K2", "K1I", "N"]))
    corr.add_edge(
        SwizzleNode(
            "A", [
                "K2", "K1I"]), GetRootNode(
            "A", [
                "K2", "K1I"]))
    corr.add_edge(
        SwizzleNode(
            "A", [
                "K1", "K0"]), GetRootNode(
            "A", [
                "K1", "K0"]))
    corr.add_edge(
        SwizzleNode(
            "B", [
                "K1", "N", "K0"]), GetRootNode(
            "B", [
                "K1", "N", "K0"]))

    assert nx.is_isomorphic(graph, corr)


def test_graph_conv():
    spec = """
        loop-order:
            O: [W, Q]
    """
    program = build_program_conv(spec)
    graph = FlowGraph(program, None).get_graph()

    corr = nx.DiGraph()
    corr.add_edge(LoopNode("W"), LoopNode("Q"))
    corr.add_edge(LoopNode("W"), OtherNode("Body"))
    corr.add_edge(LoopNode("Q"), OtherNode("Body"))
    corr.add_edge(OtherNode("Graphics"), LoopNode("W"))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Output"), GetRootNode("O", ['Q']))
    corr.add_edge(OtherNode("Body"), OtherNode("Footer"))
    corr.add_edge(GetRootNode("O", ['Q']), LoopNode("Q"))
    corr.add_edge(GetRootNode("I", ['W']), LoopNode("W"))
    corr.add_edge(GetRootNode("F", ['S']), LoopNode("Q"))
    corr.add_edge(SwizzleNode("I", ["W"]), GetRootNode("I", ["W"]))
    corr.add_edge(SwizzleNode("F", ["S"]), GetRootNode("F", ["S"]))

    assert nx.is_isomorphic(graph, corr)


def test_graph_conv_part():
    spec = """
        partitioning:
            O:
                Q: [uniform_shape(20), uniform_occupancy(I.10)]
        loop-order:
            O: [Q2, Q1, S, Q0]
    """
    program = build_program_conv(spec)
    graph = FlowGraph(program, None).get_graph()

    corr = nx.DiGraph()

    corr.add_edge(LoopNode("Q2"), LoopNode("Q1"))
    corr.add_edge(LoopNode("Q2"), FromFiberNode("I", "W1I"))
    corr.add_edge(LoopNode("Q1"), LoopNode("S"))
    corr.add_edge(LoopNode("Q1"), LoopNode("Q0"))
    corr.add_edge(LoopNode("S"), LoopNode("Q0"))
    corr.add_edge(LoopNode("S"), OtherNode("Body"))
    corr.add_edge(LoopNode("Q0"), OtherNode("Body"))
    corr.add_edge(OtherNode("Graphics"), LoopNode("Q2"))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Output"), GetRootNode("O", ["Q2", "Q1", "Q0"]))
    corr.add_edge(OtherNode("Body"), OtherNode("Footer"))
    corr.add_edge(GetRootNode("O", ["Q2", "Q1", "Q0"]), LoopNode("Q2"))
    corr.add_edge(PartNode("I", ("W",)), OtherNode("Graphics"))
    corr.add_edge(PartNode("I", ("W",)), PartNode("I", ("W1I",)))
    corr.add_edge(PartNode("I", ("W",)), SwizzleNode("I", ["Q2", "W1I"]))
    corr.add_edge(PartNode("I", ("W1I",)), SwizzleNode("I", ["Q1", "W0"]))
    corr.add_edge(GetRootNode("I", ["Q2", "W1I"]), LoopNode("Q2"))
    corr.add_edge(GetRootNode("F", ["S"]), LoopNode("S"))
    corr.add_edge(FromFiberNode("I", "W1I"), PartNode("I", ("W1I",)))
    corr.add_edge(GetRootNode("I", ["Q1", "W0"]), LoopNode("Q1"))
    corr.add_edge(LoopNode("Q1"), IntervalNode("Q0"))
    corr.add_edge(IntervalNode("Q0"), LoopNode("Q0"))
    corr.add_edge(GetRootNode("I", ["Q1", "W0"]), EagerInputNode("Q1", ["I"]))
    corr.add_edge(EagerInputNode("Q1", ["I"]), IntervalNode("Q0"))
    corr.add_edge(
        SwizzleNode(
            "I", [
                "Q2", "W1I"]), GetRootNode(
            "I", [
                "Q2", "W1I"]))
    corr.add_edge(SwizzleNode("F", ["S"]), GetRootNode("F", ["S"]))
    corr.add_edge(
        SwizzleNode(
            "I", [
                "Q1", "W0"]), GetRootNode(
            "I", [
                "Q1", "W0"]))

    assert nx.is_isomorphic(graph, corr)


def test_graph_metrics():
    program, hardware = build_gamma()
    program.add_einsum(0)
    metrics = Metrics(program, hardware)
    graph = FlowGraph(program, metrics).get_graph()

    corr = nx.DiGraph()

    corr.add_edge(LoopNode("M"), LoopNode("K"))
    corr.add_edge(LoopNode("K"), LoopNode("N"))
    corr.add_edge(LoopNode("K"), OtherNode("Body"))
    corr.add_edge(LoopNode("N"), OtherNode("Body"))
    corr.add_edge(OtherNode("Graphics"), LoopNode("M"))
    corr.add_edge(OtherNode("Graphics"), MetricsNode("Start"))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Output"), GetRootNode("T", ["M", "K", "N"]))
    corr.add_edge(OtherNode("Body"), OtherNode("Footer"))
    corr.add_edge(OtherNode("Body"), MetricsNode("End"))
    corr.add_edge(OtherNode("Footer"), MetricsNode("Dump"))
    corr.add_edge(MetricsNode("Start"), LoopNode("M"))
    corr.add_edge(MetricsNode("End"), OtherNode("Footer"))
    corr.add_edge(GetRootNode("T", ["M", "K", "N"]), LoopNode("M"))
    corr.add_edge(SwizzleNode("A", ["M", "K"]), GetRootNode("A", ["M", "K"]))
    corr.add_edge(GetRootNode("A", ["M", "K"]), LoopNode("M"))
    corr.add_edge(SwizzleNode("B", ["K", "N"]), GetRootNode("B", ["K", "N"]))
    corr.add_edge(GetRootNode("B", ["K", "N"]), LoopNode("K"))
    corr.add_edge(SwizzleNode("B", ['K', 'N']), CollectingNode("B", "K"))
    corr.add_edge(CollectingNode("B", "K"), MetricsNode("Start"))

    assert nx.is_isomorphic(graph, corr)


def test_build_fiber_nodes_empty_graph():
    program = build_program_no_loops()
    flow_graph = FlowGraph(program, None)
    iter_graph = IterationGraph(program)

    with pytest.raises(ValueError) as excinfo:
        flow_graph._FlowGraph__build_fiber_nodes(iter_graph)

    assert str(excinfo.value) == "No loop node to connect"
