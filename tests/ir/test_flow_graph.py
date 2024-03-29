import networkx as nx
import pytest

from teaal.ir.flow_graph import FlowGraph
from teaal.ir.flow_nodes import *
from teaal.ir.hardware import Hardware
from teaal.ir.iter_graph import IterationGraph
from teaal.ir.metrics import Metrics
from teaal.ir.program import Program
from teaal.parse import *


def print_errs(graph, corr):
    print("In Graph")
    for edge in graph.edges:
        if edge not in corr.edges:
            print("    corr.add_edge", end="(")
            print(type(edge[0]).__name__, end="(")
            print(str(list(edge[0]._Node__key()))[1:-1], end="), ")
            print(type(edge[1]).__name__, end="(")
            print(str(list(edge[1]._Node__key()))[1:-1], end="))\n")

    print("In Corr")
    for edge in corr.edges:
        if edge not in graph.edges:
            print(edge)

    print("---")


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
            - Z[m, n] = A[k, m] * B[k, n]
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
            - O[q] = I[q + s] * F[s]
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
    hardware = Hardware(arch, bindings, program)

    format_ = Format.from_str(yaml)

    return program, hardware, format_


def build_extensor():
    with open("tests/integration/extensor.yaml", "r") as f:
        yaml = f.read()

    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)

    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    hardware = Hardware(arch, bindings, program)

    format_ = Format.from_str(yaml)

    return program, hardware, format_


def build_extensor_energy():
    with open("tests/integration/extensor-energy.yaml", "r") as f:
        yaml = f.read()

    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)

    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    hardware = Hardware(arch, bindings, program)

    format_ = Format.from_str(yaml)

    return program, hardware, format_


def test_graph_no_loops():
    program = build_program_no_loops()
    graph = FlowGraph(program, None, []).get_graph()

    corr = nx.DiGraph()
    corr.add_edge(OtherNode("Output"), GetRootNode("A", []))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Graphics"), OtherNode("Body"))
    corr.add_edge(GetRootNode("A", []), OtherNode("Body"))
    corr.add_edge(OtherNode("Body"), OtherNode("Footer"))

    print_errs(graph, corr)

    assert nx.is_isomorphic(graph, corr)


def test_graph():
    program = build_program_matmul("")
    graph = FlowGraph(program, None, []).get_graph()

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
    corr.add_edge(OtherNode("Body"), EndLoopNode("K"))
    corr.add_edge(EndLoopNode("K"), EndLoopNode("N"))
    corr.add_edge(EndLoopNode("N"), EndLoopNode("M"))
    corr.add_edge(EndLoopNode("M"), OtherNode("Footer"))
    corr.add_edge(SwizzleNode(
        "A", ["M", "K"], "loop-order"), GetRootNode("A", ["M", "K"]))
    corr.add_edge(SwizzleNode(
        "B", ["N", "K"], "loop-order"), GetRootNode("B", ["N", "K"]))
    corr.add_edge(SwizzleNode(
        "A", ["M", "K"], "loop-order"), OtherNode("Graphics"))
    corr.add_edge(SwizzleNode(
        "B", ["N", "K"], "loop-order"), OtherNode("Graphics"))

    print_errs(graph, corr)

    assert nx.is_isomorphic(graph, corr)


def test_graph_loop_order():
    spec = """
        loop-order:
            Z: [K, M, N]
    """
    program = build_program_matmul(spec)
    graph = FlowGraph(program, None, []).get_graph()

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
    corr.add_edge(OtherNode("Body"), EndLoopNode("N"))
    corr.add_edge(EndLoopNode("N"), EndLoopNode("M"))
    corr.add_edge(EndLoopNode("M"), EndLoopNode("K"))
    corr.add_edge(EndLoopNode("K"), OtherNode("Footer"))
    corr.add_edge(SwizzleNode(
        "A", ["K", "M"], "loop-order"), GetRootNode("A", ["K", "M"]))
    corr.add_edge(SwizzleNode(
        "B", ["K", "N"], "loop-order"), GetRootNode("B", ["K", "N"]))
    corr.add_edge(SwizzleNode(
        "A", ["K", "M"], "loop-order"), OtherNode("Graphics"))
    corr.add_edge(SwizzleNode(
        "B", ["K", "N"], "loop-order"), OtherNode("Graphics"))

    print_errs(graph, corr)

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
    graph = FlowGraph(program, None, []).get_graph()

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
    corr.add_edge(OtherNode("Body"), EndLoopNode("K0"))
    corr.add_edge(EndLoopNode("K0"), EndLoopNode("N0"))
    corr.add_edge(EndLoopNode("N0"), EndLoopNode("K1"))
    corr.add_edge(EndLoopNode("K1"), EndLoopNode("N1"))
    corr.add_edge(EndLoopNode("N1"), EndLoopNode("M"))
    corr.add_edge(EndLoopNode("M"), EndLoopNode("K2"))
    corr.add_edge(EndLoopNode("K2"), OtherNode("Footer"))
    corr.add_edge(GetRootNode("Z", ['M', 'N1', 'N0']), LoopNode("M"))
    corr.add_edge(PartNode("A", ('K',)), OtherNode("Graphics"))
    corr.add_edge(
        PartNode(
            "A", ('K',)), SwizzleNode(
            "A", [
                'K2', 'M', 'K1', 'K0'], "loop-order"))
    corr.add_edge(GetRootNode("A", ['K2', 'M', 'K1', 'K0']), LoopNode("K2"))
    corr.add_edge(PartNode("B", ('K',)), OtherNode("Graphics"))
    corr.add_edge(
        PartNode(
            "B", ('K',)), SwizzleNode(
            "B", [
                'K2', 'N1', 'K1', 'N0', 'K0'], "loop-order"))
    corr.add_edge(PartNode("B", ('N',)), OtherNode("Graphics"))
    corr.add_edge(
        PartNode(
            "B", ('N',)), SwizzleNode(
            "B", [
                'K2', 'N1', 'K1', 'N0', 'K0'], "loop-order"))
    corr.add_edge(
        GetRootNode(
            "B", [
                'K2', 'N1', 'K1', 'N0', 'K0']), LoopNode("K2"))
    corr.add_edge(
        SwizzleNode(
            "A", [
                "K2", "M", "K1", "K0"], "loop-order"), GetRootNode(
            "A", [
                "K2", "M", "K1", "K0"]))
    corr.add_edge(
        SwizzleNode(
            "B", [
                "K2", "N1", "K1", "N0", "K0"], "loop-order"), GetRootNode(
            "B", [
                "K2", "N1", "K1", "N0", "K0"]))
    corr.add_edge(
        SwizzleNode(
            "A", [
                "K2", "M", "K1", "K0"], "loop-order"), OtherNode("Graphics"))
    corr.add_edge(
        SwizzleNode(
            "B", [
                "K2", "N1", "K1", "N0", "K0"], "loop-order"), OtherNode("Graphics"))

    print_errs(graph, corr)

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
    graph = FlowGraph(program, None, []).get_graph()

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
    corr.add_edge(OtherNode("Body"), EndLoopNode("K0"))
    corr.add_edge(EndLoopNode("K0"), EndLoopNode("N0"))
    corr.add_edge(EndLoopNode("N0"), EndLoopNode("K1"))
    corr.add_edge(EndLoopNode("K1"), EndLoopNode("N1"))
    corr.add_edge(EndLoopNode("N1"), EndLoopNode("M"))
    corr.add_edge(EndLoopNode("M"), EndLoopNode("K2"))
    corr.add_edge(EndLoopNode("K2"), OtherNode("Footer"))
    corr.add_edge(GetRootNode("Z", ['M', 'N1', 'N0']), LoopNode("M"))
    corr.add_edge(
        PartNode(
            "A", ('K',)), SwizzleNode(
            "A", [
                'K2', 'M', 'K1I'], "loop-order"))
    corr.add_edge(PartNode("A", ('K',)), PartNode("A", ('K1I',)))
    corr.add_edge(
        PartNode(
            "A", ('K1I',)), SwizzleNode(
            "A", [
                'K1', 'K0'], "loop-order"))
    corr.add_edge(GetRootNode("A", ['K', 'M']), FromFiberNode("A", "K"))
    corr.add_edge(FromFiberNode("A", "K"), PartNode("A", ('K',)))
    corr.add_edge(GetRootNode("A", ['K2', 'M', 'K1I']), LoopNode("K2"))
    corr.add_edge(GetRootNode("A", ['K2', 'M', 'K1I']), PartNode("B", ('K',)))
    corr.add_edge(GetRootNode("A", ['K1', 'K0']), LoopNode("K1"))
    corr.add_edge(GetRootNode("A", ['K1', 'K0']), PartNode("B", ('K1I',)))
    corr.add_edge(
        PartNode(
            "B", ('K',)), SwizzleNode(
            "B", [
                'K2', 'N', 'K1I'], "loop-order"))
    corr.add_edge(PartNode("B", ('K',)), PartNode("B", ('K1I',)))
    corr.add_edge(
        PartNode(
            "B", ('K1I',)), SwizzleNode(
            "B", [
                'K1', 'N0', 'K0'], "loop-order"))
    corr.add_edge(
        PartNode(
            "B", ('N',)), SwizzleNode(
            "B", [
                'N1', 'K1I', 'N0'], "loop-order"))
    corr.add_edge(
        PartNode(
            "B", ('K',)), SwizzleNode(
            "B", [
                'N1', 'K1I', 'N0'], "loop-order"))
    corr.add_edge(
        PartNode(
            "B", ('N',)), SwizzleNode(
            "B", [
                'K1', 'N0', 'K0'], "loop-order"))
    corr.add_edge(GetRootNode("B", ['K', 'N']), FromFiberNode("B", "K"))
    corr.add_edge(FromFiberNode("B", "K"), PartNode("B", ('K',)))
    corr.add_edge(GetRootNode("B", ['K2', 'N', 'K1I']), LoopNode("K2"))
    corr.add_edge(GetRootNode("B", ['N1', 'K1I', 'N0']), LoopNode("N1"))
    corr.add_edge(GetRootNode("B", ['K1', 'N0', 'K0']), LoopNode("K1"))
    corr.add_edge(SwizzleNode(
        "A", ["K", "M"], "loop-order"), GetRootNode("A", ["K", "M"]))
    corr.add_edge(SwizzleNode(
        "A", ["K", "M"], "loop-order"), OtherNode("Graphics"))
    corr.add_edge(SwizzleNode(
        "B", ["K", "N"], "loop-order"), GetRootNode("B", ["K", "N"]))
    corr.add_edge(SwizzleNode(
        "B", ["K", "N"], "loop-order"), OtherNode("Graphics"))
    corr.add_edge(
        SwizzleNode(
            "A", [
                "K2", "M", "K1I"], "loop-order"), GetRootNode(
            "A", [
                "K2", "M", "K1I"]))
    corr.add_edge(
        SwizzleNode(
            "B", [
                "K2", "N", "K1I"], "loop-order"), GetRootNode(
            "B", [
                "K2", "N", "K1I"]))
    corr.add_edge(
        SwizzleNode(
            "B", [
                "N1", "K1I", "N0"], "loop-order"), GetRootNode(
            "B", [
                "N1", "K1I", "N0"]))
    corr.add_edge(
        SwizzleNode(
            "A", [
                "K1", "K0"], "loop-order"), GetRootNode(
            "A", [
                "K1", "K0"]))
    corr.add_edge(
        SwizzleNode(
            "B", [
                "K1", "N0", "K0"], "loop-order"), GetRootNode(
            "B", [
                "K1", "N0", "K0"]))

    print_errs(graph, corr)

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
    graph = FlowGraph(program, None, []).get_graph()

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
    corr.add_edge(OtherNode("Body"), EndLoopNode("K0"))
    corr.add_edge(EndLoopNode("K0"), EndLoopNode("N"))
    corr.add_edge(EndLoopNode("N"), EndLoopNode("K1"))
    corr.add_edge(EndLoopNode("K1"), EndLoopNode("K2"))
    corr.add_edge(EndLoopNode("K2"), EndLoopNode("M"))
    corr.add_edge(EndLoopNode("M"), EndLoopNode("K3"))
    corr.add_edge(EndLoopNode("K3"), OtherNode("Footer"))
    corr.add_edge(GetRootNode("Z", ["M", "N"]), LoopNode("M"))
    corr.add_edge(PartNode("A", ("K",)), OtherNode("Graphics"))
    corr.add_edge(PartNode("A", ("K",)), PartNode("A", ("K2I",)))
    corr.add_edge(
        PartNode(
            "A", ("K",)), SwizzleNode(
            "A", [
                "K3", "M", "K2I"], "loop-order"))
    corr.add_edge(PartNode("A", ("K2I",)), PartNode("A", ("K1I",)))
    corr.add_edge(
        PartNode(
            "A", ("K2I",)), SwizzleNode(
            "A", [
                "K2", "K1I"], "loop-order"))
    corr.add_edge(
        PartNode(
            "A", ("K1I",)), SwizzleNode(
            "A", [
                "K1", "K0"], "loop-order"))
    corr.add_edge(GetRootNode("A", ["K3", "M", "K2I"]), LoopNode("K3"))
    corr.add_edge(PartNode("B", ("K",)), OtherNode("Graphics"))
    corr.add_edge(PartNode("B", ("K",)), PartNode("B", ("K2I",)))
    corr.add_edge(
        PartNode(
            "B", ("K",)), SwizzleNode(
            "B", [
                "K3", "K2I", "N"], "loop-order"))
    corr.add_edge(PartNode("B", ("K2I",)), PartNode("B", ("K1I",)))
    corr.add_edge(
        PartNode(
            "B", ("K2I",)), SwizzleNode(
            "B", [
                "K2", "K1I", "N"], "loop-order"))
    corr.add_edge(
        PartNode(
            "B", ("K1I",)), SwizzleNode(
            "B", [
                "K1", "N", "K0"], "loop-order"))
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
                "K3", "M", "K2I"], "loop-order"), GetRootNode(
            "A", [
                "K3", "M", "K2I"]))
    corr.add_edge(
        SwizzleNode(
            "A", [
                "K3", "M", "K2I"], "loop-order"), OtherNode("Graphics"))
    corr.add_edge(
        SwizzleNode(
            "B", [
                "K3", "K2I", "N"], "loop-order"), GetRootNode(
            "B", [
                "K3", "K2I", "N"]))
    corr.add_edge(
        SwizzleNode(
            "B", [
                "K3", "K2I", "N"], "loop-order"), OtherNode("Graphics"))
    corr.add_edge(
        SwizzleNode(
            "B", [
                "K2", "K1I", "N"], "loop-order"), GetRootNode(
            "B", [
                "K2", "K1I", "N"]))
    corr.add_edge(
        SwizzleNode(
            "A", [
                "K2", "K1I"], "loop-order"), GetRootNode(
            "A", [
                "K2", "K1I"]))
    corr.add_edge(
        SwizzleNode(
            "A", [
                "K1", "K0"], "loop-order"), GetRootNode(
            "A", [
                "K1", "K0"]))
    corr.add_edge(
        SwizzleNode(
            "B", [
                "K1", "N", "K0"], "loop-order"), GetRootNode(
            "B", [
                "K1", "N", "K0"]))

    print_errs(graph, corr)

    assert nx.is_isomorphic(graph, corr)


def test_graph_static_flattening():
    mapping = """
        partitioning:
            Z:
                K: [uniform_shape(4)]
                (M, K0): [flatten()]
                MK0: [uniform_occupancy(A.5)]
        loop-order:
            Z: [K1, MK01, N, MK00]
    """
    program = build_program_matmul(mapping)
    graph = FlowGraph(program, None, []).get_graph()

    corr = nx.DiGraph()

    corr.add_edge(LoopNode("K1"), LoopNode("MK01"))
    corr.add_edge(LoopNode("K1"), FromFiberNode("A", "MK0"))
    corr.add_edge(LoopNode("K1"), LoopNode("N"))
    corr.add_edge(LoopNode("MK01"), LoopNode("N"))
    corr.add_edge(LoopNode("MK01"), LoopNode("MK00"))
    corr.add_edge(LoopNode("N"), LoopNode("MK00"))
    corr.add_edge(LoopNode("N"), GetPayloadNode("Z", ['M']))
    corr.add_edge(LoopNode("N"), GetPayloadNode("B", ['K0']))
    corr.add_edge(LoopNode("MK00"), OtherNode("Body"))
    corr.add_edge(LoopNode("MK00"), GetPayloadNode("Z", ['M']))
    corr.add_edge(LoopNode("MK00"), GetPayloadNode("B", ['K0']))
    corr.add_edge(OtherNode("Graphics"), LoopNode("K1"))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Output"), GetRootNode("Z", ['N', 'M']))
    corr.add_edge(OtherNode("Body"), EndLoopNode("MK00"))
    corr.add_edge(EndLoopNode("MK00"), EndLoopNode("N"))
    corr.add_edge(EndLoopNode("N"), EndLoopNode("MK01"))
    corr.add_edge(EndLoopNode("MK01"), EndLoopNode("K1"))
    corr.add_edge(EndLoopNode("K1"), OtherNode("Footer"))
    corr.add_edge(GetRootNode("Z", ['N', 'M']), LoopNode("N"))
    corr.add_edge(PartNode("A", ('K',)), OtherNode("Graphics"))
    corr.add_edge(
        PartNode(
            "A", ('K',)), SwizzleNode(
            "A", [
                'M', 'K0'], "partitioning"))
    corr.add_edge(
        PartNode(
            "A", ('K',)), SwizzleNode(
            "A", [
                'K1', 'MK0'], "loop-order"))
    corr.add_edge(
        SwizzleNode(
            "A", [
                'M', 'K0'], "partitioning"), PartNode(
            "A", ('M', 'K0')))
    corr.add_edge(PartNode("A", ('M', 'K0')), OtherNode("Graphics"))
    corr.add_edge(PartNode("A", ('M', 'K0')), PartNode("A", ('MK0',)))
    corr.add_edge(
        PartNode(
            "A", ('M', 'K0')), SwizzleNode(
            "A", [
                'K1', 'MK0'], "loop-order"))
    corr.add_edge(
        PartNode(
            "A", ('MK0',)), SwizzleNode(
            "A", [
                'MK01', 'MK00'], "loop-order"))
    corr.add_edge(SwizzleNode(
        "A", ['K1', 'MK0'], "loop-order"), GetRootNode("A", ['K1', 'MK0']))
    corr.add_edge(SwizzleNode(
        "A", ['K1', 'MK0'], "loop-order"), OtherNode("Graphics"))
    corr.add_edge(GetRootNode("A", ['K1', 'MK0']), LoopNode("K1"))
    corr.add_edge(PartNode("B", ('K',)), OtherNode("Graphics"))
    corr.add_edge(
        PartNode(
            "B", ('K',)), SwizzleNode(
            "B", [
                'K1', 'N', 'K0'], "loop-order"))
    corr.add_edge(SwizzleNode(
        "B", ['K1', 'N', 'K0'], "loop-order"), GetRootNode("B", ['K1', 'N', 'K0']))
    corr.add_edge(SwizzleNode(
        "B", ['K1', 'N', 'K0'], "loop-order"), OtherNode("Graphics"))
    corr.add_edge(GetRootNode("B", ['K1', 'N', 'K0']), LoopNode("K1"))
    corr.add_edge(FromFiberNode("A", "MK0"), PartNode("A", ('MK0',)))
    corr.add_edge(SwizzleNode(
        "A", ['MK01', 'MK00'], "loop-order"), GetRootNode("A", ['MK01', 'MK00']))
    corr.add_edge(GetRootNode("A", ['MK01', 'MK00']), LoopNode("MK01"))
    corr.add_edge(GetPayloadNode("Z", ['M']), OtherNode("Body"))
    corr.add_edge(GetPayloadNode("B", ['K0']), OtherNode("Body"))

    print_errs(graph, corr)

    assert nx.is_isomorphic(graph, corr)


def test_graph_dyn_flattening():
    mapping = """
        partitioning:
            Z:
                M: [uniform_occupancy(A.6)]
                K: [uniform_occupancy(A.4)]
                (M0, K0): [flatten()]
                M0K0: [uniform_occupancy(A.5)]
        loop-order:
            Z: [M1, K1, M0K01, N, M0K00]
    """
    program = build_program_matmul(mapping)
    graph = FlowGraph(program, None, []).get_graph()

    corr = nx.DiGraph()
    corr.add_edge(LoopNode('M1'), LoopNode('K1'))
    corr.add_edge(LoopNode('M1'), LoopNode('N'))
    corr.add_edge(LoopNode('M1'), FromFiberNode('A', 'K'))
    corr.add_edge(LoopNode('K1'), LoopNode('M0K01'))
    corr.add_edge(LoopNode('K1'), FromFiberNode('A', 'M0K0'))
    corr.add_edge(LoopNode('K1'), LoopNode('N'))
    corr.add_edge(LoopNode('M0K01'), LoopNode('N'))
    corr.add_edge(LoopNode('M0K01'), LoopNode('M0K00'))
    corr.add_edge(LoopNode('N'), LoopNode('M0K00'))
    corr.add_edge(LoopNode('N'), GetPayloadNode('Z', ['M0']))
    corr.add_edge(LoopNode('N'), GetPayloadNode('B', ['K0']))
    corr.add_edge(LoopNode('M0K00'), OtherNode('Body'))
    corr.add_edge(LoopNode('M0K00'), GetPayloadNode('Z', ['M0']))
    corr.add_edge(LoopNode('M0K00'), GetPayloadNode('B', ['K0']))
    corr.add_edge(OtherNode('Graphics'), LoopNode('M1'))
    corr.add_edge(OtherNode('Output'), OtherNode('Graphics'))
    corr.add_edge(OtherNode('Output'), GetRootNode('Z', ['M1', 'N', 'M0']))
    corr.add_edge(OtherNode("Body"), EndLoopNode("M0K00"))
    corr.add_edge(EndLoopNode("M0K00"), EndLoopNode("N"))
    corr.add_edge(EndLoopNode("N"), EndLoopNode("M0K01"))
    corr.add_edge(EndLoopNode("M0K01"), EndLoopNode("K1"))
    corr.add_edge(EndLoopNode("K1"), EndLoopNode("M1"))
    corr.add_edge(EndLoopNode("M1"), OtherNode("Footer"))
    corr.add_edge(GetRootNode('Z', ['M1', 'N', 'M0']), LoopNode('M1'))
    corr.add_edge(
        PartNode(
            'A', ('K',)), SwizzleNode(
            'A', [
                'M0', 'K0'], 'partitioning'))
    corr.add_edge(PartNode('A', ('K',)), PartNode('A', ('M0', 'K0')))
    corr.add_edge(
        PartNode(
            'A', ('K',)), SwizzleNode(
            'A', [
                'K1', 'M0K0'], 'loop-order'))
    corr.add_edge(
        PartNode(
            'A', ('M',)), SwizzleNode(
            'A', [
                'M0', 'K0'], 'partitioning'))
    corr.add_edge(PartNode('A', ('M',)), PartNode('A', ('M0', 'K0')))
    corr.add_edge(
        PartNode(
            'A', ('M',)), SwizzleNode(
            'A', [
                'M1', 'K', 'M0'], 'loop-order'))
    corr.add_edge(
        SwizzleNode(
            'A', [
                'M0', 'K0'], 'partitioning'), PartNode(
            'A', ('M0', 'K0')))
    corr.add_edge(PartNode('A', ('M0', 'K0')), PartNode('A', ('M0K0',)))
    corr.add_edge(
        PartNode(
            'A', ('M0', 'K0')), SwizzleNode(
            'A', [
                'K1', 'M0K0'], 'loop-order'))
    corr.add_edge(
        PartNode(
            'A', ('M0K0',)), SwizzleNode(
            'A', [
                'M0K01', 'M0K00'], 'loop-order'))
    corr.add_edge(SwizzleNode(
        'A', ['M', 'K'], 'loop-order'), GetRootNode('A', ['M', 'K']))
    corr.add_edge(SwizzleNode(
        'A', ['M', 'K'], 'loop-order'), OtherNode("Graphics"))
    corr.add_edge(GetRootNode('A', ['M', 'K']), FromFiberNode('A', 'M'))
    corr.add_edge(
        PartNode(
            'B', ('K',)), SwizzleNode(
            'B', [
                'K1', 'N', 'K0'], 'loop-order'))
    corr.add_edge(SwizzleNode(
        'B', ['K', 'N'], 'loop-order'), GetRootNode('B', ['K', 'N']))
    corr.add_edge(SwizzleNode(
        'B', ['K', 'N'], 'loop-order'), OtherNode("Graphics"))
    corr.add_edge(GetRootNode('B', ['K', 'N']), FromFiberNode('B', 'K'))
    corr.add_edge(FromFiberNode('A', 'M'), PartNode('A', ('M',)))
    corr.add_edge(SwizzleNode(
        'A', ['M1', 'K', 'M0'], 'loop-order'), GetRootNode('A', ['M1', 'K', 'M0']))
    corr.add_edge(GetRootNode('A', ['M1', 'K', 'M0']), LoopNode('M1'))
    corr.add_edge(FromFiberNode('B', 'K'), PartNode('B', ('K',)))
    corr.add_edge(SwizzleNode(
        'B', ['K1', 'N', 'K0'], 'loop-order'), GetRootNode('B', ['K1', 'N', 'K0']))
    corr.add_edge(GetRootNode('B', ['K1', 'N', 'K0']), LoopNode('K1'))
    corr.add_edge(FromFiberNode('A', 'K'), PartNode('A', ('K',)))
    corr.add_edge(SwizzleNode(
        'A', ['K1', 'M0K0'], 'loop-order'), GetRootNode('A', ['K1', 'M0K0']))
    corr.add_edge(GetRootNode('A', ['K1', 'M0K0']), PartNode('B', ('K',)))
    corr.add_edge(GetRootNode('A', ['K1', 'M0K0']), LoopNode('K1'))
    corr.add_edge(FromFiberNode('A', 'M0K0'), PartNode('A', ('M0K0',)))
    corr.add_edge(SwizzleNode(
        'A', ['M0K01', 'M0K00'], 'loop-order'), GetRootNode('A', ['M0K01', 'M0K00']))
    corr.add_edge(GetRootNode('A', ['M0K01', 'M0K00']), LoopNode('M0K01'))
    corr.add_edge(GetPayloadNode('Z', ['M0']), OtherNode('Body'))
    corr.add_edge(GetPayloadNode('B', ['K0']), OtherNode('Body'))

    print_errs(graph, corr)

    assert nx.is_isomorphic(graph, corr)


def test_graph_conv():
    spec = """
        loop-order:
            O: [W, Q]
    """
    program = build_program_conv(spec)
    graph = FlowGraph(program, None, []).get_graph()

    corr = nx.DiGraph()
    corr.add_edge(LoopNode("W"), LoopNode("Q"))
    corr.add_edge(LoopNode("W"), OtherNode("Body"))
    corr.add_edge(LoopNode("Q"), OtherNode("Body"))
    corr.add_edge(OtherNode("Graphics"), LoopNode("W"))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Output"), GetRootNode("O", ['Q']))
    corr.add_edge(OtherNode("Body"), EndLoopNode("Q"))
    corr.add_edge(EndLoopNode("Q"), EndLoopNode("W"))
    corr.add_edge(EndLoopNode("W"), OtherNode("Footer"))
    corr.add_edge(GetRootNode("O", ['Q']), LoopNode("Q"))
    corr.add_edge(GetRootNode("I", ['W']), LoopNode("W"))
    corr.add_edge(GetRootNode("F", ['S']), LoopNode("Q"))
    corr.add_edge(
        SwizzleNode(
            "I",
            ["W"],
            "loop-order"),
        GetRootNode(
            "I",
            ["W"]))
    corr.add_edge(SwizzleNode("I", ["W"], "loop-order"), OtherNode("Graphics"))
    corr.add_edge(
        SwizzleNode(
            "F",
            ["S"],
            "loop-order"),
        GetRootNode(
            "F",
            ["S"]))
    corr.add_edge(SwizzleNode("F", ["S"], "loop-order"), OtherNode("Graphics"))

    print_errs(graph, corr)

    assert nx.is_isomorphic(graph, corr)


def test_graph_conv_part():
    spec = """
        partitioning:
            O:
                Q: [uniform_shape(20), uniform_occupancy(I.10)]
                W: [follow(Q)]
        loop-order:
            O: [Q2, Q1, S, Q0]
    """
    program = build_program_conv(spec)
    graph = FlowGraph(program, None, []).get_graph()

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
    corr.add_edge(OtherNode("Body"), EndLoopNode("Q0"))
    corr.add_edge(EndLoopNode("Q0"), EndLoopNode("S"))
    corr.add_edge(EndLoopNode("S"), EndLoopNode("Q1"))
    corr.add_edge(EndLoopNode("Q1"), EndLoopNode("Q2"))
    corr.add_edge(EndLoopNode("Q2"), OtherNode("Footer"))
    corr.add_edge(GetRootNode("O", ["Q2", "Q1", "Q0"]), LoopNode("Q2"))
    corr.add_edge(PartNode("I", ("W",)), OtherNode("Graphics"))
    corr.add_edge(PartNode("I", ("W",)), PartNode("I", ("W1I",)))
    corr.add_edge(
        PartNode(
            "I", ("W",)), SwizzleNode(
            "I", [
                "W2", "W1I"], "loop-order"))
    corr.add_edge(
        PartNode(
            "I", ("W1I",)), SwizzleNode(
            "I", [
                "W1", "W0"], "loop-order"))
    corr.add_edge(GetRootNode("I", ["W2", "W1I"]), LoopNode("Q2"))
    corr.add_edge(GetRootNode("F", ["S"]), LoopNode("S"))
    corr.add_edge(FromFiberNode("I", "W1I"), PartNode("I", ("W1I",)))
    corr.add_edge(GetRootNode("I", ["W1", "W0"]), LoopNode("Q1"))
    corr.add_edge(LoopNode("Q1"), IntervalNode("Q0"))
    corr.add_edge(IntervalNode("Q0"), LoopNode("Q0"))
    corr.add_edge(GetRootNode("I", ["W1", "W0"]), EagerInputNode("Q1", ["I"]))
    corr.add_edge(EagerInputNode("Q1", ["I"]), IntervalNode("Q0"))
    corr.add_edge(
        SwizzleNode(
            "I", [
                "W2", "W1I"], "loop-order"), GetRootNode(
            "I", [
                "W2", "W1I"]))
    corr.add_edge(
        SwizzleNode(
            "I", [
                "W2", "W1I"], "loop-order"), OtherNode("Graphics"))
    corr.add_edge(
        SwizzleNode(
            "F",
            ["S"],
            "loop-order"),
        GetRootNode(
            "F",
            ["S"]))
    corr.add_edge(
        SwizzleNode(
            "F",
            ["S"],
            "loop-order"), OtherNode("Graphics"))
    corr.add_edge(
        SwizzleNode(
            "I", [
                "W1", "W0"], "loop-order"), GetRootNode(
            "I", [
                "W1", "W0"]))

    print_errs(graph, corr)

    assert nx.is_isomorphic(graph, corr)


def test_graph_metrics_no_loops():
    yaml = """
    einsum:
      declaration:
        Z: []
      expressions:
        - Z[] = a
    architecture:
      accel:
      - name: empty
    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
    format:
      Z:
        default:
          rank-order: []
    """
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)

    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    hardware = Hardware(arch, bindings, program)

    format_ = Format.from_str(yaml)

    program.add_einsum(0)
    metrics = Metrics(program, hardware, format_)
    graph = FlowGraph(program, metrics, []).get_graph()

    corr = nx.DiGraph()

    corr.add_edge(OtherNode("Body"), OtherNode("Footer"))
    corr.add_edge(OtherNode("Body"), MetricsNode("End"))
    corr.add_edge(OtherNode("Graphics"), OtherNode("Body"))
    corr.add_edge(OtherNode("Graphics"), MetricsNode("Start"))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Output"), GetRootNode("Z", []))
    corr.add_edge(OtherNode("Footer"), MetricsNode("Dump"))
    corr.add_edge(MetricsNode("Start"), OtherNode("Body"))
    corr.add_edge(MetricsNode("End"), OtherNode("Footer"))
    corr.add_edge(GetRootNode("Z", []), OtherNode("Body"))
    corr.add_edge(OtherNode('Graphics'), MetricsNode('Body'))
    corr.add_edge(MetricsNode('Start'), MetricsNode('Body'))
    corr.add_edge(MetricsNode('Body'), OtherNode('Body'))
    corr.add_edge(MetricsNode('Body'), MetricsNode('End'))

    print_errs(graph, corr)

    assert nx.is_isomorphic(graph, corr)


def test_graph_metrics_T():
    program, hardware, format_ = build_gamma()
    program.add_einsum(0)
    metrics = Metrics(program, hardware, format_)
    graph = FlowGraph(program, metrics, []).get_graph()

    corr = nx.DiGraph()

    corr.add_edge(LoopNode("M"), LoopNode("K"))
    corr.add_edge(LoopNode("K"), LoopNode("N"))
    corr.add_edge(LoopNode("K"), OtherNode("Body"))
    corr.add_edge(LoopNode("N"), OtherNode("Body"))
    corr.add_edge(OtherNode("Body"), EndLoopNode("N"))
    corr.add_edge(EndLoopNode("N"), EndLoopNode("K"))
    corr.add_edge(EndLoopNode("K"), EndLoopNode("M"))
    corr.add_edge(EndLoopNode("M"), OtherNode("Footer"))
    corr.add_edge(OtherNode("Graphics"), LoopNode("M"))
    corr.add_edge(OtherNode("Graphics"), MetricsNode("Start"))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Output"), GetRootNode("T", ['M', 'K', 'N']))
    corr.add_edge(OtherNode("Footer"), MetricsNode("Dump"))
    corr.add_edge(MetricsNode("Start"), LoopNode("M"))
    corr.add_edge(MetricsNode("End"), OtherNode("Footer"))
    corr.add_edge(GetRootNode("T", ['M', 'K', 'N']), LoopNode("M"))
    corr.add_edge(
        SwizzleNode("A", ['M', 'K'], "loop-order"),
        GetRootNode("A", ['M', 'K']))
    corr.add_edge(
        SwizzleNode("A", ['M', 'K'], "loop-order"), OtherNode("Graphics"))
    corr.add_edge(GetRootNode("A", ['M', 'K']), LoopNode("M"))
    corr.add_edge(SwizzleNode(
        "B", ['K', 'N'], "loop-order"), GetRootNode("B", ['K', 'N']))
    corr.add_edge(
        SwizzleNode("B", ['K', 'N'], "loop-order"), OtherNode("Graphics"))
    corr.add_edge(GetRootNode("B", ['K', 'N']), LoopNode("K"))
    corr.add_edge(LoopNode('M'), MetricsHeaderNode('K'))
    corr.add_edge(LoopNode('K'), MetricsHeaderNode('N'))
    corr.add_edge(OtherNode('Graphics'), MetricsHeaderNode('M'))
    corr.add_edge(MetricsNode('Start'), MetricsHeaderNode('M'))
    corr.add_edge(MetricsHeaderNode('M'), LoopNode('M'))
    corr.add_edge(MetricsHeaderNode('K'), LoopNode('K'))
    corr.add_edge(MetricsHeaderNode('N'), LoopNode('N'))
    corr.add_edge(LoopNode('N'), MetricsNode('Body'))
    corr.add_edge(EndLoopNode('N'), MetricsFooterNode('N'))
    corr.add_edge(EndLoopNode('K'), MetricsFooterNode('K'))
    corr.add_edge(EndLoopNode('M'), MetricsFooterNode('M'))
    corr.add_edge(MetricsNode('Body'), OtherNode('Body'))
    corr.add_edge(MetricsFooterNode('N'), EndLoopNode('K'))
    corr.add_edge(MetricsFooterNode('K'), EndLoopNode('M'))
    corr.add_edge(MetricsFooterNode('M'), OtherNode('Footer'))
    corr.add_edge(MetricsFooterNode('M'), MetricsNode('End'))
    corr.add_edge(EndLoopNode("M"), MetricsNode("End"))

    print_errs(graph, corr)

    assert nx.is_isomorphic(graph, corr)


def test_graph_metrics_Z():
    program, hardware, format_ = build_gamma()
    program.add_einsum(1)
    metrics = Metrics(program, hardware, format_)
    graph = FlowGraph(program, metrics, []).get_graph()

    corr = nx.DiGraph()

    corr.add_edge(LoopNode("M"), LoopNode("N"))
    corr.add_edge(LoopNode("M"), LoopNode("K"))
    corr.add_edge(LoopNode("N"), LoopNode("K"))
    corr.add_edge(LoopNode("N"), OtherNode("Body"))
    corr.add_edge(LoopNode("K"), OtherNode("Body"))
    corr.add_edge(OtherNode("Body"), EndLoopNode("K"))
    corr.add_edge(EndLoopNode("K"), EndLoopNode("N"))
    corr.add_edge(EndLoopNode("N"), EndLoopNode("M"))
    corr.add_edge(EndLoopNode("M"), OtherNode("Footer"))
    corr.add_edge(OtherNode("Graphics"), LoopNode("M"))
    corr.add_edge(OtherNode("Graphics"), MetricsNode("Start"))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Output"), GetRootNode("Z", ['M', 'N']))
    corr.add_edge(OtherNode("Footer"), MetricsNode("Dump"))
    corr.add_edge(MetricsNode("Start"), LoopNode("M"))
    corr.add_edge(MetricsNode("End"), OtherNode("Footer"))
    corr.add_edge(GetRootNode("Z", ['M', 'N']), LoopNode("M"))
    corr.add_edge(SwizzleNode(
        "T", ['M', 'N', 'K'], "loop-order"), GetRootNode("T", ['M', 'N', 'K']))
    corr.add_edge(
        SwizzleNode(
            "T", [
                'M', 'N', 'K'], "loop-order"), OtherNode("Graphics"))
    corr.add_edge(GetRootNode("T", ['M', 'N', 'K']), LoopNode("M"))
    corr.add_edge(
        SwizzleNode(
            "T", [
                'M', 'K', 'N'], "metrics"), SwizzleNode(
            "T", [
                'M', 'N', 'K'], "loop-order"))
    corr.add_edge(SwizzleNode(
        "A", ['M', 'K'], "loop-order"), GetRootNode("A", ['M', 'K']))
    corr.add_edge(
        SwizzleNode(
            "A", [
                'M', 'K'], "loop-order"), OtherNode("Graphics"))
    corr.add_edge(GetRootNode("A", ['M', 'K']), LoopNode("M"))
    corr.add_edge(LoopNode('M'), MetricsHeaderNode('N'))
    corr.add_edge(LoopNode('N'), MetricsHeaderNode('K'))
    corr.add_edge(OtherNode('Graphics'), MetricsHeaderNode('M'))
    corr.add_edge(MetricsNode('Start'), MetricsHeaderNode('M'))
    corr.add_edge(MetricsHeaderNode('M'), LoopNode('M'))
    corr.add_edge(MetricsHeaderNode('N'), LoopNode('N'))
    corr.add_edge(MetricsHeaderNode('K'), LoopNode('K'))
    corr.add_edge(LoopNode('K'), MetricsNode('Body'))
    corr.add_edge(EndLoopNode('K'), MetricsFooterNode('K'))
    corr.add_edge(EndLoopNode('N'), MetricsFooterNode('N'))
    corr.add_edge(EndLoopNode('M'), MetricsFooterNode('M'))
    corr.add_edge(MetricsNode('Body'), OtherNode('Body'))
    corr.add_edge(MetricsFooterNode('K'), EndLoopNode('N'))
    corr.add_edge(MetricsFooterNode('N'), EndLoopNode('M'))
    corr.add_edge(MetricsFooterNode('M'), OtherNode('Footer'))
    corr.add_edge(MetricsFooterNode('M'), MetricsNode('End'))
    corr.add_edge(EndLoopNode("M"), MetricsNode("End"))

    print_errs(graph, corr)

    assert nx.is_isomorphic(graph, corr)


def test_graph_metrics_extensor():
    program, hardware, format_ = build_extensor()
    program.add_einsum(0)
    metrics = Metrics(program, hardware, format_)
    graph = FlowGraph(program, metrics, []).get_graph()

    corr = nx.DiGraph()

    corr.add_edge(LoopNode("N2"), LoopNode("K2"))
    corr.add_edge(LoopNode("N2"), LoopNode("M2"))
    corr.add_edge(LoopNode("K2"), LoopNode("M2"))
    corr.add_edge(LoopNode("K2"), LoopNode("N1"))
    corr.add_edge(LoopNode("M2"), LoopNode("M1"))
    corr.add_edge(LoopNode("M1"), LoopNode("N1"))
    corr.add_edge(LoopNode("M1"), LoopNode("K1"))
    corr.add_edge(LoopNode("N1"), LoopNode("K1"))
    corr.add_edge(LoopNode("N1"), LoopNode("M0"))
    corr.add_edge(LoopNode("K1"), LoopNode("M0"))
    corr.add_edge(LoopNode("K1"), LoopNode("N0"))
    corr.add_edge(LoopNode("M0"), LoopNode("N0"))
    corr.add_edge(LoopNode("M0"), LoopNode("K0"))
    corr.add_edge(LoopNode("N0"), LoopNode("K0"))
    corr.add_edge(LoopNode("N0"), OtherNode("Body"))
    corr.add_edge(LoopNode("K0"), OtherNode("Body"))
    corr.add_edge(OtherNode("Body"), EndLoopNode("K0"))
    corr.add_edge(EndLoopNode("K0"), EndLoopNode("N0"))
    corr.add_edge(EndLoopNode("N0"), EndLoopNode("M0"))
    corr.add_edge(EndLoopNode("M0"), EndLoopNode("K1"))
    corr.add_edge(EndLoopNode("K1"), EndLoopNode("N1"))
    corr.add_edge(EndLoopNode("N1"), EndLoopNode("M1"))
    corr.add_edge(EndLoopNode("M1"), EndLoopNode("M2"))
    corr.add_edge(EndLoopNode("M2"), EndLoopNode("K2"))
    corr.add_edge(EndLoopNode("K2"), EndLoopNode("N2"))
    corr.add_edge(EndLoopNode("N2"), OtherNode("Footer"))
    corr.add_edge(OtherNode("Footer"), MetricsNode("Dump"))
    corr.add_edge(OtherNode("Graphics"), LoopNode("N2"))
    corr.add_edge(OtherNode("Graphics"), MetricsNode("Start"))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Output"), GetRootNode(
        "Z", ['N2', 'M2', 'M1', 'N1', 'M0', 'N0']))
    corr.add_edge(MetricsNode("Start"), LoopNode("N2"))
    corr.add_edge(MetricsNode("End"), OtherNode("Footer"))
    corr.add_edge(
        GetRootNode(
            "Z", [
                'N2', 'M2', 'M1', 'N1', 'M0', 'N0']), LoopNode("N2"))
    corr.add_edge(PartNode("A", ('M',)), OtherNode("Graphics"))
    corr.add_edge(
        PartNode(
            "A", ('M',)), SwizzleNode(
            "A", [
                'K2', 'M2', 'M1', 'K1', 'M0', 'K0'], "loop-order"))
    corr.add_edge(PartNode("A", ('K',)), OtherNode("Graphics"))
    corr.add_edge(
        PartNode(
            "A", ('K',)), SwizzleNode(
            "A", [
                'K2', 'M2', 'M1', 'K1', 'M0', 'K0'], "loop-order"))
    corr.add_edge(SwizzleNode("A",
                              ['K2', 'M2', 'M1', 'K1', 'M0', 'K0'], "loop-order"),
                  GetRootNode("A", ['K2', 'M2', 'M1', 'K1', 'M0', 'K0']))
    corr.add_edge(
        SwizzleNode(
            "A", [
                'K2', 'M2', 'M1', 'K1', 'M0', 'K0'], "loop-order"), OtherNode("Graphics"))
    corr.add_edge(
        GetRootNode(
            "A", [
                'K2', 'M2', 'M1', 'K1', 'M0', 'K0']), LoopNode("K2"))
    corr.add_edge(PartNode("B", ('K',)), OtherNode("Graphics"))
    corr.add_edge(
        PartNode(
            "B", ('K',)), SwizzleNode(
            "B", [
                'N2', 'K2', 'N1', 'K1', 'N0', 'K0'], "loop-order"))
    corr.add_edge(PartNode("B", ('N',)), OtherNode("Graphics"))
    corr.add_edge(
        PartNode(
            "B", ('N',)), SwizzleNode(
            "B", [
                'N2', 'K2', 'N1', 'K1', 'N0', 'K0'], "loop-order"))
    corr.add_edge(SwizzleNode("B",
                              ['N2', 'K2', 'N1', 'K1', 'N0', 'K0'],
                              "loop-order"),
                  GetRootNode("B",
                              ['N2', 'K2', 'N1', 'K1', 'N0', 'K0']))
    corr.add_edge(
        SwizzleNode(
            "B", [
                'N2', 'K2', 'N1', 'K1', 'N0', 'K0'], "loop-order"), OtherNode("Graphics"))
    corr.add_edge(
        GetRootNode(
            "B", [
                'N2', 'K2', 'N1', 'K1', 'N0', 'K0']), LoopNode("N2"))
    corr.add_edge(LoopNode('N2'), MetricsHeaderNode('K2'))
    corr.add_edge(LoopNode('K2'), MetricsHeaderNode('M2'))
    corr.add_edge(LoopNode('M2'), MetricsHeaderNode('M1'))
    corr.add_edge(LoopNode('M1'), MetricsHeaderNode('N1'))
    corr.add_edge(LoopNode('N1'), MetricsHeaderNode('K1'))
    corr.add_edge(LoopNode('K1'), MetricsHeaderNode('M0'))
    corr.add_edge(LoopNode('M0'), MetricsHeaderNode('N0'))
    corr.add_edge(LoopNode('N0'), MetricsHeaderNode('K0'))
    corr.add_edge(OtherNode('Graphics'), MetricsHeaderNode('N2'))
    corr.add_edge(MetricsNode('Start'), MetricsHeaderNode('N2'))
    corr.add_edge(MetricsHeaderNode('N2'), LoopNode('N2'))
    corr.add_edge(MetricsHeaderNode('K2'), LoopNode('K2'))
    corr.add_edge(MetricsHeaderNode('M2'), LoopNode('M2'))
    corr.add_edge(MetricsHeaderNode('M1'), LoopNode('M1'))
    corr.add_edge(MetricsHeaderNode('N1'), LoopNode('N1'))
    corr.add_edge(MetricsHeaderNode('K1'), LoopNode('K1'))
    corr.add_edge(MetricsHeaderNode('M0'), LoopNode('M0'))
    corr.add_edge(MetricsHeaderNode('N0'), LoopNode('N0'))
    corr.add_edge(MetricsHeaderNode('K0'), LoopNode('K0'))
    corr.add_edge(LoopNode('K0'), MetricsNode('Body'))
    corr.add_edge(EndLoopNode('K0'), MetricsFooterNode('K0'))
    corr.add_edge(EndLoopNode('N0'), MetricsFooterNode('N0'))
    corr.add_edge(EndLoopNode('M0'), MetricsFooterNode('M0'))
    corr.add_edge(EndLoopNode('K1'), MetricsFooterNode('K1'))
    corr.add_edge(EndLoopNode('N1'), MetricsFooterNode('N1'))
    corr.add_edge(EndLoopNode('M1'), MetricsFooterNode('M1'))
    corr.add_edge(EndLoopNode('M2'), MetricsFooterNode('M2'))
    corr.add_edge(EndLoopNode('K2'), MetricsFooterNode('K2'))
    corr.add_edge(EndLoopNode('N2'), MetricsFooterNode('N2'))
    corr.add_edge(MetricsNode('Body'), OtherNode('Body'))
    corr.add_edge(MetricsFooterNode('K0'), EndLoopNode('N0'))
    corr.add_edge(MetricsFooterNode('N0'), EndLoopNode('M0'))
    corr.add_edge(MetricsFooterNode('M0'), EndLoopNode('K1'))
    corr.add_edge(MetricsFooterNode('K1'), EndLoopNode('N1'))
    corr.add_edge(MetricsFooterNode('N1'), EndLoopNode('M1'))
    corr.add_edge(MetricsFooterNode('M1'), EndLoopNode('M2'))
    corr.add_edge(MetricsFooterNode('M2'), EndLoopNode('K2'))
    corr.add_edge(MetricsFooterNode('K2'), EndLoopNode('N2'))
    corr.add_edge(MetricsFooterNode('N2'), OtherNode('Footer'))
    corr.add_edge(MetricsFooterNode('N2'), MetricsNode('End'))
    corr.add_edge(EndLoopNode("N2"), MetricsNode("End"))

    print_errs(graph, corr)

    assert nx.is_isomorphic(graph, corr)


def test_graph_metrics_extensor_energy():
    program, hardware, format_ = build_extensor_energy()
    program.add_einsum(0)
    metrics = Metrics(program, hardware, format_)
    graph = FlowGraph(program, metrics, []).get_graph()

    corr = nx.DiGraph()

    corr.add_edge(LoopNode("N2"), LoopNode("K2"))
    corr.add_edge(LoopNode("N2"), LoopNode("M2"))
    corr.add_edge(LoopNode("K2"), LoopNode("M2"))
    corr.add_edge(LoopNode("K2"), LoopNode("N1"))
    corr.add_edge(LoopNode("M2"), LoopNode("M1"))
    corr.add_edge(LoopNode("M1"), LoopNode("N1"))
    corr.add_edge(LoopNode("M1"), LoopNode("K1"))
    corr.add_edge(LoopNode("N1"), LoopNode("K1"))
    corr.add_edge(LoopNode("N1"), LoopNode("M0"))
    corr.add_edge(LoopNode("K1"), LoopNode("M0"))
    corr.add_edge(LoopNode("K1"), LoopNode("N0"))
    corr.add_edge(LoopNode("M0"), LoopNode("N0"))
    corr.add_edge(LoopNode("M0"), LoopNode("K0"))
    corr.add_edge(LoopNode("N0"), LoopNode("K0"))
    corr.add_edge(LoopNode("N0"), OtherNode("Body"))
    corr.add_edge(LoopNode("K0"), OtherNode("Body"))
    corr.add_edge(OtherNode("Body"), EndLoopNode("K0"))
    corr.add_edge(EndLoopNode("K0"), EndLoopNode("N0"))
    corr.add_edge(EndLoopNode("N0"), EndLoopNode("M0"))
    corr.add_edge(EndLoopNode("M0"), EndLoopNode("K1"))
    corr.add_edge(EndLoopNode("K1"), EndLoopNode("N1"))
    corr.add_edge(EndLoopNode("N1"), EndLoopNode("M1"))
    corr.add_edge(EndLoopNode("M1"), EndLoopNode("M2"))
    corr.add_edge(EndLoopNode("M2"), EndLoopNode("K2"))
    corr.add_edge(EndLoopNode("K2"), EndLoopNode("N2"))
    corr.add_edge(EndLoopNode("N2"), OtherNode("Footer"))
    corr.add_edge(OtherNode("Footer"), MetricsNode("Dump"))
    corr.add_edge(OtherNode("Graphics"), LoopNode("N2"))
    corr.add_edge(OtherNode("Graphics"), MetricsNode("Start"))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Output"), GetRootNode(
        "Z", ['N2', 'M2', 'M1', 'N1', 'M0', 'N0']))
    corr.add_edge(MetricsNode("Start"), LoopNode("N2"))
    corr.add_edge(MetricsNode("End"), OtherNode("Footer"))
    corr.add_edge(
        GetRootNode(
            "Z", [
                'N2', 'M2', 'M1', 'N1', 'M0', 'N0']), LoopNode("N2"))
    corr.add_edge(PartNode("A", ('M',)), OtherNode("Graphics"))
    corr.add_edge(
        PartNode(
            "A", ('M',)), SwizzleNode(
            "A", [
                'K2', 'M2', 'M1', 'K1', 'M0', 'K0'], "loop-order"))
    corr.add_edge(PartNode("A", ('K',)), OtherNode("Graphics"))
    corr.add_edge(
        PartNode(
            "A", ('K',)), SwizzleNode(
            "A", [
                'K2', 'M2', 'M1', 'K1', 'M0', 'K0'], "loop-order"))
    corr.add_edge(SwizzleNode("A",
                              ['K2', 'M2', 'M1', 'K1', 'M0', 'K0'], "loop-order"),
                  GetRootNode("A", ['K2', 'M2', 'M1', 'K1', 'M0', 'K0']))
    corr.add_edge(
        SwizzleNode(
            "A", [
                'K2', 'M2', 'M1', 'K1', 'M0', 'K0'], "loop-order"), OtherNode("Graphics"))
    corr.add_edge(
        GetRootNode(
            "A", [
                'K2', 'M2', 'M1', 'K1', 'M0', 'K0']), LoopNode("K2"))
    corr.add_edge(PartNode("B", ('K',)), OtherNode("Graphics"))
    corr.add_edge(
        PartNode(
            "B", ('K',)), SwizzleNode(
            "B", [
                'N2', 'K2', 'N1', 'K1', 'N0', 'K0'], "loop-order"))
    corr.add_edge(PartNode("B", ('N',)), OtherNode("Graphics"))
    corr.add_edge(
        PartNode(
            "B", ('N',)), SwizzleNode(
            "B", [
                'N2', 'K2', 'N1', 'K1', 'N0', 'K0'], "loop-order"))
    corr.add_edge(SwizzleNode("B",
                              ['N2', 'K2', 'N1', 'K1', 'N0', 'K0'],
                              "loop-order"),
                  GetRootNode("B",
                              ['N2', 'K2', 'N1', 'K1', 'N0', 'K0']))
    corr.add_edge(
        SwizzleNode(
            "B", [
                'N2', 'K2', 'N1', 'K1', 'N0', 'K0'], "loop-order"), OtherNode("Graphics"))
    corr.add_edge(
        GetRootNode(
            "B", [
                'N2', 'K2', 'N1', 'K1', 'N0', 'K0']), LoopNode("N2"))
    corr.add_edge(LoopNode('N2'), MetricsHeaderNode('K2'))
    corr.add_edge(LoopNode('K2'), MetricsHeaderNode('M2'))
    corr.add_edge(LoopNode('M2'), MetricsHeaderNode('M1'))
    corr.add_edge(LoopNode('M1'), MetricsHeaderNode('N1'))
    corr.add_edge(LoopNode('N1'), MetricsHeaderNode('K1'))
    corr.add_edge(LoopNode('K1'), MetricsHeaderNode('M0'))
    corr.add_edge(LoopNode('M0'), MetricsHeaderNode('N0'))
    corr.add_edge(LoopNode('N0'), MetricsHeaderNode('K0'))
    corr.add_edge(OtherNode('Graphics'), MetricsHeaderNode('N2'))
    corr.add_edge(MetricsNode('Start'), MetricsHeaderNode('N2'))
    corr.add_edge(MetricsHeaderNode('N2'), LoopNode('N2'))
    corr.add_edge(MetricsHeaderNode('K2'), LoopNode('K2'))
    corr.add_edge(MetricsHeaderNode('M2'), LoopNode('M2'))
    corr.add_edge(MetricsHeaderNode('M1'), LoopNode('M1'))
    corr.add_edge(MetricsHeaderNode('N1'), LoopNode('N1'))
    corr.add_edge(MetricsHeaderNode('K1'), LoopNode('K1'))
    corr.add_edge(MetricsHeaderNode('M0'), LoopNode('M0'))
    corr.add_edge(MetricsHeaderNode('N0'), LoopNode('N0'))
    corr.add_edge(MetricsHeaderNode('K0'), LoopNode('K0'))
    corr.add_edge(LoopNode('K0'), MetricsNode('Body'))
    corr.add_edge(EndLoopNode('K0'), MetricsFooterNode('K0'))
    corr.add_edge(EndLoopNode('N0'), MetricsFooterNode('N0'))
    corr.add_edge(EndLoopNode('M0'), MetricsFooterNode('M0'))
    corr.add_edge(EndLoopNode('K1'), MetricsFooterNode('K1'))
    corr.add_edge(EndLoopNode('N1'), MetricsFooterNode('N1'))
    corr.add_edge(EndLoopNode('M1'), MetricsFooterNode('M1'))
    corr.add_edge(EndLoopNode('M2'), MetricsFooterNode('M2'))
    corr.add_edge(EndLoopNode('K2'), MetricsFooterNode('K2'))
    corr.add_edge(EndLoopNode('N2'), MetricsFooterNode('N2'))
    corr.add_edge(MetricsNode('Body'), OtherNode('Body'))
    corr.add_edge(MetricsFooterNode('K0'), EndLoopNode('N0'))
    corr.add_edge(MetricsFooterNode('N0'), EndLoopNode('M0'))
    corr.add_edge(MetricsFooterNode('M0'), EndLoopNode('K1'))
    corr.add_edge(MetricsFooterNode('K1'), EndLoopNode('N1'))
    corr.add_edge(MetricsFooterNode('N1'), EndLoopNode('M1'))
    corr.add_edge(MetricsFooterNode('M1'), EndLoopNode('M2'))
    corr.add_edge(MetricsFooterNode('M2'), EndLoopNode('K2'))
    corr.add_edge(MetricsFooterNode('K2'), EndLoopNode('N2'))
    corr.add_edge(MetricsFooterNode('N2'), OtherNode('Footer'))
    corr.add_edge(MetricsFooterNode('N2'), MetricsNode('End'))
    corr.add_edge(EndLoopNode("N2"), MetricsNode("End"))

    print_errs(graph, corr)

    assert nx.is_isomorphic(graph, corr)


def test_graph_metrics_swizzle_for_part():
    yaml = """
    einsum:
      declaration:
        Z: []
        A: [K, M]
      expressions:
        - Z[] = A[k, m]
    mapping:
      partitioning:
        Z:
          (M, K): [flatten()]
    architecture:
      accel:
      - name: level0
        local:
        - name: Merger
          class: Merger
          attributes:
            inputs: 16
            comparator_radix: 16
    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
      - component: Merger
        bindings:
        - tensor: A
          init-ranks: [K, M]
          final-ranks: [M, K]
    format:
      A:
        default:
          rank-order: [MK]
          MK:
            format: C
            pbits: 64
      Z:
        default:
          rank-order: []
    """
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)

    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    hardware = Hardware(arch, bindings, program)

    format_ = Format.from_str(yaml)

    program.add_einsum(0)
    metrics = Metrics(program, hardware, format_)
    graph = FlowGraph(program, metrics, []).get_graph()

    corr = nx.DiGraph()

    corr.add_edge(LoopNode("MK"), OtherNode("Body"))
    corr.add_edge(OtherNode("Body"), EndLoopNode("MK"))
    corr.add_edge(EndLoopNode("MK"), OtherNode("Footer"))
    corr.add_edge(OtherNode("Graphics"), LoopNode("MK"))
    corr.add_edge(OtherNode("Graphics"), MetricsNode("Start"))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Output"), GetRootNode("Z", []))
    corr.add_edge(OtherNode("Footer"), MetricsNode("Dump"))
    corr.add_edge(MetricsNode("Start"), LoopNode("MK"))
    corr.add_edge(MetricsNode("End"), OtherNode("Footer"))
    corr.add_edge(GetRootNode("Z", []), OtherNode("Body"))
    corr.add_edge(
        SwizzleNode("A", ['M', 'K'], "partitioning"),
        PartNode("A", ('M', 'K')))
    corr.add_edge(PartNode("A", ('M', 'K')), OtherNode("Graphics"))
    corr.add_edge(
        PartNode("A", ('M', 'K')),
        SwizzleNode("A", ['MK'], "loop-order"))
    corr.add_edge(
        SwizzleNode("A", ['K', 'M'], "metrics"),
        SwizzleNode("A", ['M', 'K'], "partitioning"))
    corr.add_edge(
        SwizzleNode("A", ['MK'], "loop-order"),
        GetRootNode("A", ['MK']))
    corr.add_edge(
        SwizzleNode("A", ['MK'], "loop-order"),
        OtherNode("Graphics"))
    corr.add_edge(GetRootNode("A", ['MK']), LoopNode("MK"))
    corr.add_edge(OtherNode('Graphics'), MetricsHeaderNode('MK'))
    corr.add_edge(MetricsNode('Start'), MetricsHeaderNode('MK'))
    corr.add_edge(MetricsHeaderNode('MK'), LoopNode('MK'))
    corr.add_edge(LoopNode('MK'), MetricsNode('Body'))
    corr.add_edge(MetricsNode('Body'), OtherNode('Body'))
    corr.add_edge(MetricsFooterNode('MK'), MetricsNode('End'))
    corr.add_edge(EndLoopNode('MK'), MetricsFooterNode('MK'))
    corr.add_edge(MetricsFooterNode('MK'), OtherNode('Footer'))
    corr.add_edge(EndLoopNode("MK"), MetricsNode("End"))

    print_errs(graph, corr)

    assert nx.is_isomorphic(graph, corr)


def test_graph_metrics_trace_output():
    yaml = """
    einsum:
      declaration:
        Z: [K, M]
        A: [K, M]
      expressions:
        - Z[k, m] = A[k, m]
    architecture:
      accel:
      - name: level0
        local:
        - name: Buffer
          class: Buffet
    bindings:
      Z:
      - config: accel
        prefix: tmp/Z
      - component: Buffer
        bindings:
        - tensor: Z
          rank: K
          type: payload
          style: eager
          evict-on: root
          format: default
    format:
      Z:
        default:
          rank-order: [K, M]
          K:
            format: C
            pbits: 32
          M:
            format: C
            cbits: 32
            pbits: 64
    """
    einsum = Einsum.from_str(yaml)
    mapping = Mapping.from_str(yaml)
    program = Program(einsum, mapping)

    arch = Architecture.from_str(yaml)
    bindings = Bindings.from_str(yaml)
    hardware = Hardware(arch, bindings, program)

    format_ = Format.from_str(yaml)

    program.add_einsum(0)
    metrics = Metrics(program, hardware, format_)
    graph = FlowGraph(program, metrics, []).get_graph()

    corr = nx.DiGraph()

    corr.add_edge(LoopNode("K"), LoopNode("M"))
    corr.add_edge(LoopNode("M"), OtherNode("Body"))
    corr.add_edge(OtherNode("Body"), EndLoopNode("M"))
    corr.add_edge(EndLoopNode("M"), EndLoopNode("K"))
    corr.add_edge(EndLoopNode("K"), OtherNode("Footer"))
    corr.add_edge(OtherNode("Footer"), MetricsNode("Dump"))
    corr.add_edge(OtherNode("Graphics"), LoopNode("K"))
    corr.add_edge(OtherNode("Graphics"), MetricsNode("Start"))
    corr.add_edge(OtherNode("Output"), OtherNode("Graphics"))
    corr.add_edge(OtherNode("Output"), GetRootNode("Z", ['K', 'M']))
    corr.add_edge(MetricsNode("Start"), LoopNode("K"))
    corr.add_edge(MetricsNode("End"), OtherNode("Footer"))
    corr.add_edge(GetRootNode("Z", ['K', 'M']), LoopNode("K"))
    corr.add_edge(SwizzleNode(
        "A", ['K', 'M'], "loop-order"), GetRootNode("A", ['K', 'M']))
    corr.add_edge(
        SwizzleNode(
            "A", ['K', 'M'], "loop-order"), OtherNode("Graphics"))
    corr.add_edge(GetRootNode("A", ['K', 'M']), LoopNode("K"))
    corr.add_edge(LoopNode('K'), MetricsHeaderNode('M'))
    corr.add_edge(OtherNode('Graphics'), MetricsHeaderNode('K'))
    corr.add_edge(MetricsNode('Start'), MetricsHeaderNode('K'))
    corr.add_edge(MetricsHeaderNode('K'), LoopNode('K'))
    corr.add_edge(MetricsHeaderNode('M'), LoopNode('M'))
    corr.add_edge(LoopNode('M'), MetricsNode('Body'))
    corr.add_edge(MetricsNode('Body'), OtherNode('Body'))
    corr.add_edge(MetricsFooterNode('K'), MetricsNode('End'))
    corr.add_edge(EndLoopNode('M'), MetricsFooterNode('M'))
    corr.add_edge(EndLoopNode('K'), MetricsFooterNode('K'))
    corr.add_edge(MetricsFooterNode('M'), EndLoopNode('K'))
    corr.add_edge(MetricsFooterNode('K'), OtherNode('Footer'))
    corr.add_edge(EndLoopNode("K"), MetricsNode("End"))

    print_errs(graph, corr)

    assert nx.is_isomorphic(graph, corr)


def test_build_fiber_nodes_empty_graph():
    program = build_program_no_loops()
    flow_graph = FlowGraph(program, None, [])
    iter_graph = IterationGraph(program)

    with pytest.raises(ValueError) as excinfo:
        flow_graph._FlowGraph__build_fiber_nodes(iter_graph, {})

    assert str(excinfo.value) == "No loop node to connect"


def test_loop_hoisting():
    spec = """
        partitioning:
            Z:
                K: [uniform_occupancy(A.6), uniform_occupancy(A.3)]
                N: [uniform_occupancy(B.5)]
        loop-order:
            Z: [K2, M, N1, K1, N0, K0]
    """
    program = build_program_matmul(spec)
    corr = [15, 20, 25, 30, 31, 32]

    flow_graph = FlowGraph(program, None, [])
    pos = []
    for rank in program.get_loop_order().get_ranks():
        pos.append(flow_graph.get_sorted().index(LoopNode(rank)))

    # Note that this can technically happen, we just need to make sure that
    # the hoist option is doing something. If this test fails, try a different
    # schedule
    assert pos != corr

    flow_graph = FlowGraph(program, None, ["hoist"])
    pos = []
    for rank in program.get_loop_order().get_ranks():
        pos.append(flow_graph.get_sorted().index(LoopNode(rank)))

    assert pos == corr
