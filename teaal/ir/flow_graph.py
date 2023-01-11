"""
MIT License

Copyright (c) 2021 University of Illinois

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Representation of the control-dataflow graph of the program
"""

import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
from typing import cast, Dict, List, Optional

from teaal.ir.flow_nodes import *
from teaal.ir.iter_graph import IterationGraph
from teaal.ir.metrics import Metrics
from teaal.ir.node import Node
from teaal.ir.program import Program
from teaal.ir.tensor import Tensor


class FlowGraph:
    """
    The control-dataflow graph for the HiFiber program
    """

    def __init__(
            self,
            program: Program,
            metrics: Optional[Metrics],
            opts: List[str]) -> None:
        """
        Construct a new FlowGraph
        """
        self.program = program
        self.metrics = metrics

        self.__build()
        self.__prune()
        self.__sort()

        if "hoist" in opts:
            self.__hoist()

    def draw(self) -> None:  # pragma: no cover
        """
        Draw the graph
        """

        plt.figure(figsize=(8, 6))
        nx.draw(
            self.graph,
            with_labels=True,
            font_size=8,
            pos=nx.random_layout(
                self.graph,
                seed=2))
        plt.savefig("foo.png")

    def get_graph(self) -> nx.DiGraph:
        """
        Return the flow graph for this program
        """
        return self.graph

    def get_sorted(self) -> List[Node]:
        """
        Get the sorted graph
        """
        return self.sorted

    def __build(self) -> None:
        """
        Build the flow graph
        """
        self.graph = nx.DiGraph()
        self.iter_map: Dict[str, List[str]] = {}

        self.__build_loop_nest()
        self.__build_output()

        # Add Swizzle, GetRoot and FiberNodes for each tensor
        part = self.program.get_partitioning()
        for tensor in self.program.get_tensors():
            if tensor.get_is_output():
                continue

            root = tensor.root_name()

            # Add the partitioning
            init_ranks = tensor.get_ranks()
            for rank in init_ranks:
                self.graph.add_edge(TensorNode(root), RankNode(root, rank))

                # Add the static partitioning
                part_rank = part.partition_rank(rank)
                # TODO: allow for flattening
                if (part_rank,) in part.get_static_parts():
                    self.__build_static_part(tensor, rank)

                    in_rank = part.partition_names([rank], False)[0]
                    if in_rank != part.get_final_rank_id(tensor, in_rank):
                        self.__build_dyn_part(tensor, rank)

                # Add the dynamic partitioning
                # TODO: allow for flattening
                if (part_rank,) in part.get_dyn_parts():
                    self.__build_dyn_part(tensor, rank)

            # Get the root fiber
            self.__build_swizzle_root_fiber(tensor)

        # Add CollectingNodes
        for tensor in self.program.get_tensors():
            self.__build_collecting(tensor)

        # Connect all Fibers with the appropriate loop nodes
        iter_graph = IterationGraph(self.program)
        while iter_graph.peek()[0] is not None:
            self.__build_fiber_nodes(iter_graph)

        for tensor in self.program.get_tensors():
            # The last FiberNode is needed for the body
            self.graph.add_edge(
                FiberNode(tensor.fiber_name()),
                OtherNode("Body"))

        # Reset all tensors
        for tensor in self.program.get_tensors():
            is_output = tensor.get_is_output()
            tensor.reset()
            tensor.set_is_output(is_output)

    def __build_collecting(self, tensor: Tensor) -> None:
        """
        Build a CollectingNode should it be required
        """
        # None if there is no hardware
        if not self.metrics:
            return

        # None if the tensor is never stored in DRAM
        if not self.metrics.in_dram(tensor):
            return

        # None if the tensor is stationary
        if self.metrics.on_chip_stationary(tensor):
            return

        # Otherwise, add a CollectingNode
        root = tensor.root_name()
        rank = self.metrics.get_on_chip_rank(tensor)
        swizzle_node = SwizzleNode(root, tensor.get_ranks())
        collecting_node = CollectingNode(root, rank)

        self.graph.add_edge(swizzle_node, collecting_node)
        self.graph.add_edge(collecting_node, MetricsNode("Start"))

    def __build_dyn_part(self, tensor: Tensor, rank: str) -> None:
        """
        Build a dynamic partitioning
        """
        part = self.program.get_partitioning()
        root = tensor.root_name()

        int_ranks = part.get_intermediates(tensor, rank)
        part_rank = part.partition_rank(rank)
        # TODO: Allow flattening
        if part_rank and (part_rank,) in part.get_dyn_parts():
            int_ranks += [rank]

        for src in int_ranks:
            part_node = PartNode(root, (src,))
            dsts = part.partition_names([src], False)

            part_rank = part.partition_rank(src)
            # Very hard to test, since this means there is a problem with
            # part.get_intermediates()
            if part_rank is None:
                raise ValueError(
                    "Unknown intermediate: " +
                    src)  # pragma: no cover

            # TODO: allow flattening
            # TODO: use the correct destination rank
            leader = part.get_leader(src, dsts[0])

            # Add the edge from the source rank and fiber to the
            # PartNode
            self.graph.add_edge(RankNode(root, src), part_node)
            if root != leader:
                lead_name = leader.lower() + "_" + dsts[1].lower()
                self.graph.add_edge(
                    FiberNode(lead_name), part_node)

            # Add the destination RankNodes
            for dst in dsts:
                self.graph.add_edge(part_node, RankNode(root, dst))

    def __build_fiber_nodes(self, iter_graph: IterationGraph) -> None:
        """
        Build the FiberNodes between loops
        """
        # If this is a dynamically partitioned rank, add the relevant nodes
        part = self.program.get_partitioning()
        for tensor in self.program.get_tensors():
            rank = tensor.peek()
            if rank is None:
                continue

            rank = rank.upper()
            part_rank = part.partition_rank(rank)
            # TODO: allow for flattening
            if part_rank and (part_rank,) in part.get_dyn_parts():
                self.__connect_dyn_part(tensor, rank)

        rank, tensors = iter_graph.peek()

        if rank is None:
            raise ValueError("No loop node to connect")

        for tensor in tensors:
            # Connect the old fiber to the LoopNode
            fiber_node = FiberNode(tensor.fiber_name())
            self.graph.add_edge(fiber_node, LoopNode(rank))

        # Update the iter_map with the tensors iterated on at this rank
        self.iter_map[rank] = [tensor.root_name()
                               for tensor in tensors if not tensor.get_is_output()]

        # We need a EagerInputNode and an IntervalNode if at least one tensor
        # will be projected and it is a partitioned rank (so we don't know the
        # bounds)
        if any(tensor.peek() != rank.lower() for tensor in tensors) and \
                part.get_root_name(rank) != rank:
            self.__build_project_interval(rank, tensors)

        _, tensors = iter_graph.pop()

        # Connect the new fiber to the LoopNode
        for tensor in tensors:
            new_fnode = FiberNode(tensor.fiber_name())
            self.graph.add_edge(LoopNode(rank), new_fnode)

    def __build_loop_nest(self) -> None:
        """
        Build the loop nest
        """
        loop_order = self.program.get_loop_order().get_ranks()

        # Add the edges connecting the LoopNodes
        chain: List[Node] = [OtherNode("StartLoop")]
        for rank in loop_order:
            chain.append(LoopNode(rank))
            self.graph.add_edge(chain[-2], chain[-1])

        # Add the graphics generation, body, and footer
        self.graph.add_edge(OtherNode("Graphics"), OtherNode("StartLoop"))
        self.graph.add_edge(OtherNode("Output"), OtherNode("Graphics"))
        self.graph.add_edge(chain[-1], OtherNode("Body"))
        self.graph.add_edge(OtherNode("Body"), OtherNode("Footer"))

        # If we have Metrics, we need to add the MetricsNodes
        if self.metrics:
            self.graph.add_edge(OtherNode("StartLoop"), MetricsNode("Start"))
            self.graph.add_edge(MetricsNode("Start"), chain[1])

            self.graph.add_edge(OtherNode("Body"), MetricsNode("End"))
            self.graph.add_edge(MetricsNode("End"), OtherNode("Footer"))
            self.graph.add_edge(OtherNode("Footer"), MetricsNode("Dump"))

    def __build_output(self) -> None:
        """
        Build all of the output-specific edges
        """
        tensor = self.program.get_output()
        self.program.apply_all_partitioning(tensor)
        root = tensor.root_name()
        get_root_node = GetRootNode(root, tensor.get_ranks())

        # Construct the output for the relevant tensor to be available
        self.graph.add_edge(OtherNode("Output"), TensorNode(root))
        self.graph.add_edge(TensorNode(root), get_root_node)
        self.graph.add_edge(get_root_node, FiberNode(tensor.fiber_name()))

    def __build_project_interval(
            self,
            rank: str,
            tensors: List[Tensor]) -> None:
        """
        Build the EagerInputNode and IntervalNode if the projection is over a
        partitioned rank
        """
        part = self.program.get_partitioning()
        # TODO: think about flattening (actually I don't think it matters here)
        part_names = part.partition_names([part.get_root_name(rank)], True)
        rank0 = part_names[0]
        rank1 = part_names[1]

        # Connect the EagerInputNode
        eager_input_node = EagerInputNode(rank1, self.iter_map[rank1])
        for tname in self.iter_map[rank1]:
            fiber_name = tname.lower() + "_" + rank1.lower()
            self.graph.add_edge(FiberNode(fiber_name), eager_input_node)

        # Connect the IntervalNode
        self.graph.add_edge(LoopNode(rank1), IntervalNode(rank0))
        self.graph.add_edge(eager_input_node, IntervalNode(rank0))
        self.graph.add_edge(IntervalNode(rank0), LoopNode(rank0))

    def __build_static_part(self, tensor: Tensor, rank: str) -> None:
        """
        Build a static partitioning
        """
        part = self.program.get_partitioning()
        root = tensor.root_name()
        part_node = PartNode(root, (rank,))

        # Add the edge from the source rank to the partitioning
        self.graph.add_edge(RankNode(root, rank), part_node)

        # Add edges to for all resulting ranks
        # TODO: support flattening
        for res in part.partition_names([rank], False):
            self.graph.add_edge(part_node, RankNode(root, res))

        # We must do graphics after any static partitioning
        self.graph.add_edge(part_node, OtherNode("Graphics"))
        self.program.apply_partitioning(tensor, rank)

    def __build_swizzle_root_fiber(self, tensor: Tensor) -> None:
        """
        Build a swizzleRanks(), getRoot(), and the resulting FiberNode
        """
        self.program.get_loop_order().apply(tensor)
        root = tensor.root_name()

        tensor_node = TensorNode(root)
        swizzle_node = SwizzleNode(root, tensor.get_ranks().copy())
        get_root_node = GetRootNode(root, tensor.get_ranks().copy())
        fiber_node = FiberNode(tensor.fiber_name())

        self.graph.add_edge(tensor_node, swizzle_node)
        self.graph.add_edge(swizzle_node, get_root_node)
        self.graph.add_edge(get_root_node, fiber_node)

        # An SwizzleNode always requires the TensorNode and RankNodes of the
        # relevant ranks
        for rank in tensor.get_ranks():
            self.graph.add_edge(RankNode(root, rank), swizzle_node)

    def __connect_dyn_part(self, tensor: Tensor, rank: str) -> None:
        """
        Connect the dynamic partitioning node to the relevant fiber nodes
        """
        fiber_node = FiberNode(tensor.fiber_name())
        root = tensor.root_name()

        tensor.from_fiber()
        self.program.apply_partitioning(tensor, rank)
        self.program.get_loop_order().apply(tensor)

        part_node = PartNode(root, (rank,))
        ff_node = FromFiberNode(root, rank)

        self.graph.add_edge(fiber_node, ff_node)
        self.graph.add_edge(ff_node, part_node)

        self.__build_swizzle_root_fiber(tensor)

    def __prune(self) -> None:
        """
        Prune out all intermediate nodes
        """
        # Remove all FiberNodes
        nodes = [node for node in self.graph.nodes()
                 if isinstance(node, FiberNode) or
                 isinstance(node, RankNode) or
                 isinstance(node, TensorNode) or
                 (isinstance(node, OtherNode) and
                  node.get_type() == "StartLoop")]

        for node in nodes:
            # Connect all in and out edges
            for in_, _ in self.graph.in_edges(node):
                for _, out in self.graph.out_edges(node):
                    self.graph.add_edge(in_, out)

            # Remove the node
            self.graph.remove_node(node)

    def __sort(self) -> None:
        """
        Sort all nodes so that the generated code obeys all dependencies
        """
        # Get a topological sort
        self.sorted = list(nx.topological_sort(self.graph))

    def __hoist(self) -> None:
        """
        Hoist all nodes above loops they do not depend on
        """
        # Hoist all statements not dependent on the loop
        end = len(self.sorted)
        for rank in reversed(self.program.get_loop_order().get_ranks()):
            loop = self.sorted.index(LoopNode(rank))
            i = loop + 1
            descendants = nx.descendants(self.graph, LoopNode(rank))

            while i < end:
                # If this node does not depend on the loop node, then we can
                # hoist it
                if self.sorted[i] not in descendants:
                    node = self.sorted[i]
                    del self.sorted[i]
                    self.sorted.insert(loop, node)
                    loop += 1

                i += 1

            end = loop
