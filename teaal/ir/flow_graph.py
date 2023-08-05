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
from sympy import Symbol
from typing import cast, Dict, List, Optional, Tuple

from teaal.ir.component import *
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

        chain = self.__build_loop_nest()
        self.__build_output()

        # Add Swizzle, GetRoot and FiberNodes for each tensor
        part = self.program.get_partitioning()
        flatten_info: Dict[str, List[Tuple[str, ...]]] = {}
        for tensor in self.program.get_equation().get_tensors():
            if tensor.get_is_output():
                continue

            root = tensor.root_name()

            # Add the static partitioning
            init_ranks = tensor.get_ranks()
            for rank in init_ranks:
                self.graph.add_edge(TensorNode(root), RankNode(root, rank))

            self.__build_partition_swizzling(tensor, flatten_info)

            # Get the root fiber
            self.__build_swizzle_root_fiber(tensor, True)

        # Add metrics collection
        self.__build_collecting(chain)
        self.__build_hw_components(chain)

        iter_graph = IterationGraph(self.program)
        while iter_graph.peek_concord()[0] is not None:
            self.__build_fiber_nodes(iter_graph, flatten_info)

        for tensor in self.program.get_equation().get_tensors():
            # The last FiberNode is needed for the body
            self.graph.add_edge(
                FiberNode(tensor.fiber_name()),
                OtherNode("Body"))

        # Reset all tensors
        for tensor in self.program.get_equation().get_tensors():
            is_output = tensor.get_is_output()
            tensor.reset()
            tensor.set_is_output(is_output)

    def __build_collecting(self, chain: List[Node]) -> None:
        """
        Build a CollectingNode should it be required
        """
        if self.metrics is None:
            return

        loop_ranks = self.program.get_loop_order().get_ranks()
        if not loop_ranks:
            return

        collecting_nodes = []
        register_ranks_node: Optional[Node] = None
        for tensor in self.program.get_equation().get_tensors():
            tensor_name = tensor.root_name()
            part_ir = self.program.get_partitioning()
            part_ranks = part_ir.partition_ranks(
                tensor.get_init_ranks(), part_ir.get_all_parts(), True, True)
            final_tensor = Tensor(tensor_name, part_ranks)
            self.program.get_loop_order().apply(final_tensor)

            loop_node = LoopNode(loop_ranks[0])
            for rank, type_, consumable in self.metrics.get_collected_tensor_info(
                    tensor_name):
                if type_ == "fiber":
                    collecting_nodes.append(
                        CollectingNode(
                            tensor_name,
                            rank,
                            type_,
                            consumable,
                            True))

                    if tensor.get_is_output():
                        collecting_nodes.append(
                            CollectingNode(
                                tensor_name,
                                rank,
                                type_,
                                consumable,
                                False))

                elif type_ == "iter":
                    collecting_nodes.append(CollectingNode(
                        None, rank, type_, consumable, True))

                # Type is a rank name for eager iteration
                else:
                    collecting_nodes.append(
                        CollectingNode(
                            tensor_name,
                            rank,
                            type_,
                            consumable,
                            True))

                    trace_tree_node = TraceTreeNode(tensor_name, type_, True)
                    self.graph.add_edge(MetricsNode("Start"), trace_tree_node)

                    # Load right before use
                    self.graph.add_edge(
                        chain[chain.index(LoopNode(type_)) - 1], trace_tree_node)
                    self.graph.add_edge(
                        FiberNode(
                            tensor_name.lower() +
                            "_" +
                            type_.lower()),
                        trace_tree_node)

                    # If we are doing this, we need to also register the rank
                    # order explicitly
                    if register_ranks_node is None:
                        register_ranks_node = RegisterRanksNode(
                            self.program.get_loop_order().get_ranks())

                    self.graph.add_edge(
                        MetricsNode("Start"), register_ranks_node)

                    if tensor.get_is_output():
                        collecting_nodes.append(
                            CollectingNode(
                                tensor_name,
                                rank,
                                type_,
                                consumable,
                                False))

                        # Store at the last moment
                        i = final_tensor.get_ranks().index(type_)

                        last: Node
                        if i == 0:
                            last = OtherNode("Footer")
                        else:
                            last = EndLoopNode(final_tensor.get_ranks()[i - 1])
                        j = chain.index(last)

                        trace_tree_node = TraceTreeNode(
                            tensor_name, type_, False)
                        self.graph.add_edge(chain[j - 1], trace_tree_node)
                        self.graph.add_edge(trace_tree_node, chain[j])
                        self.graph.add_edge(
                            trace_tree_node, MetricsNode("End"))

        for collecting_node in collecting_nodes:
            self.graph.add_edge(MetricsNode("Start"), collecting_node)
            self.graph.add_edge(collecting_node, loop_node)
            if register_ranks_node is not None:
                self.graph.add_edge(collecting_node, register_ranks_node)

    def __build_dyn_part(
            self, tensor: Tensor, partitioning: Tuple[str, ...], flatten_info: Dict[str, List[Tuple[str, ...]]]) -> None:
        """
        Build a dynamic partitioning
        """
        part = self.program.get_partitioning()
        root = tensor.root_name()

        # Get the partitioning source ranks (including intermediates)
        src_ranks: List[Tuple[str, ...]]
        if len(partitioning) > 1:
            src_ranks = [partitioning]

            # Swizzle for flattening
            swizzle_node = SwizzleNode(
                root, list(partitioning), "partitioning")

            for rank in partitioning:
                self.graph.add_edge(RankNode(root, rank), swizzle_node)
            self.program.apply_partition_swizzling(tensor)

            # Add to flattening info
            flatten_info[root].append(partitioning)

        else:
            rank = partitioning[0]
            src_ranks = [cast(Tuple[str, ...], (rank,))] + [(inter,)
                                                            for inter in part.get_intermediates(rank)]

        # Connect them to the relevant destination ranks
        for srcs in src_ranks:
            part_node = PartNode(root, srcs)
            dsts = part.partition_names(srcs, False)

            # Add the edge from the source ranks to the PartNode
            for src in srcs:
                self.graph.add_edge(RankNode(root, src), part_node)

            # Connect the swizzle node
            if len(srcs) > 1:
                self.graph.add_edge(swizzle_node, part_node)

            # Leader is only relevant for a split
            else:
                leader = part.get_leader(src, dsts[-1])

                if root != leader:
                    lead_name = leader.lower() + "_" + dsts[1].lower()
                    self.graph.add_edge(
                        FiberNode(lead_name), part_node)

            # Add the destination RankNodes
            for dst in dsts:
                self.graph.add_edge(part_node, RankNode(root, dst))

    def __build_fiber_nodes(self, iter_graph: IterationGraph,
                            flatten_info: Dict[str, List[Tuple[str, ...]]]) -> None:
        """
        Build the FiberNodes between loops
        """
        # If this is a dynamically partitioned rank, add the relevant nodes
        part = self.program.get_partitioning()
        for tensor in self.program.get_equation().get_tensors():
            rank = tensor.peek()
            if rank is None:
                continue

            rank = rank.upper()
            # part_ranks = part.partition_rank((rank,))
            # if part_ranks and part_ranks in part.get_dyn_parts():
            if (rank,) in part.get_dyn_parts():
                self.__connect_dyn_part(tensor, rank, flatten_info)

        rank, tensors = iter_graph.peek_concord()

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
                self.program.get_partitioning().split_rank_name(rank)[1] == "0":
            self.__build_project_interval(rank)

        _, tensors = iter_graph.pop_concord()

        # Connect the new fiber to the LoopNode
        for tensor in tensors:
            new_fnode = FiberNode(tensor.fiber_name())
            self.graph.add_edge(LoopNode(rank), new_fnode)

        # Add the discordant accesses
        for ranks, tensor in iter_graph.peek_discord():
            get_payload_node = GetPayloadNode(tensor.root_name(), ranks)
            self.graph.add_edge(
                FiberNode(
                    tensor.fiber_name()),
                get_payload_node)

            for rank in ranks:
                loop_rank = part.get_final_rank_id(tensor, rank)
                self.graph.add_edge(LoopNode(loop_rank), get_payload_node)

        for ranks, tensor in iter_graph.pop_discord():
            get_payload_node = GetPayloadNode(tensor.root_name(), ranks)
            self.graph.add_edge(
                get_payload_node, FiberNode(
                    tensor.fiber_name()))

    def __build_hw_components(self, chain: List[Node]) -> None:
        """
        Build the creation of any necessary hardware components
        """
        if self.metrics is None:
            return

        einsum = self.program.get_equation().get_output().root_name()
        for component in self.metrics.get_hardware().get_components(einsum,
                                                                    IntersectorComponent):
            name = component.get_name()

            for binding in component.get_bindings()[einsum]:
                rank = binding["rank"]
                self.graph.add_edge(
                    CreateComponentNode(
                        name, rank), MetricsNode("Start"))

                consume_node = ConsumeTraceNode(name, rank)
                self.graph.add_edge(EndLoopNode(rank), consume_node)
                self.graph.add_edge(consume_node,
                                    chain[chain.index(EndLoopNode(rank)) + 1])
                self.graph.add_edge(consume_node, MetricsNode("End"))

    def __build_loop_nest(self) -> List[Node]:
        """
        Build the loop nest, returns the chain of nodes
        """
        loop_order = self.program.get_loop_order().get_ranks()

        # Add the edges connecting the LoopNodes
        chain: List[Node] = [OtherNode("StartLoop")]
        for rank in loop_order:
            chain.append(LoopNode(rank))
        chain.append(OtherNode("Body"))
        for rank in reversed(loop_order):
            chain.append(EndLoopNode(rank))
        chain.append(OtherNode("Footer"))

        # Note that the chain is guaranteed to have at least two nodes
        for i in range(len(chain) - 1):
            self.graph.add_edge(chain[i], chain[i + 1])

        # Add the graphics generation
        self.graph.add_edge(OtherNode("Graphics"), OtherNode("StartLoop"))
        self.graph.add_edge(OtherNode("Output"), OtherNode("Graphics"))

        # If we have Metrics, we need to add the MetricsNodes
        if self.metrics:
            self.graph.add_edge(OtherNode("StartLoop"), MetricsNode("Start"))
            self.graph.add_edge(MetricsNode("Start"), chain[1])

            self.graph.add_edge(chain[-2], MetricsNode("End"))
            self.graph.add_edge(MetricsNode("End"), OtherNode("Footer"))
            self.graph.add_edge(OtherNode("Footer"), MetricsNode("Dump"))

        return chain

    def __build_output(self) -> None:
        """
        Build all of the output-specific edges
        """
        tensor = self.program.get_equation().get_output()

        # Partition the output
        part = self.program.get_partitioning()
        self.program.apply_all_partitioning(tensor)
        self.program.get_loop_order().apply(tensor)

        # Get the root
        root = tensor.root_name()
        get_root_node = GetRootNode(root, tensor.get_ranks())

        # Construct the output for the relevant tensor to be available
        self.graph.add_edge(OtherNode("Output"), TensorNode(root))
        self.graph.add_edge(TensorNode(root), get_root_node)
        self.graph.add_edge(get_root_node, FiberNode(tensor.fiber_name()))

    def __build_partition_swizzling(
            self, tensor: Tensor, flatten_info: Dict[str, List[Tuple[str, ...]]]) -> None:
        """
        Build the partitioning for a particular tensor
        """
        part = self.program.get_partitioning()
        flatten_info[tensor.root_name()] = []

        # First, apply the static partitioning
        static_parts = part.get_valid_parts(
            tensor.get_ranks(), part.get_static_parts(), True)
        for partitioning in static_parts:
            self.__build_static_part(tensor, partitioning)

        # Now, apply the dynamic partitioning
        dyn_parts = part.get_valid_parts(
            tensor.get_ranks(), part.get_dyn_parts(), True)
        for partitioning in dyn_parts:
            self.__build_dyn_part(tensor, partitioning, flatten_info)

    def __build_project_interval(
            self,
            rank: str) -> None:
        """
        Build the EagerInputNode and IntervalNode if the projection is over a
        partitioned rank
        """
        part = self.program.get_partitioning()
        # Flattening does not matter because we cannot flatten index math ranks
        rank1 = rank[:-1] + "1"
        rank0 = rank

        # Connect the EagerInputNode
        eager_input_node = EagerInputNode(rank1, self.iter_map[rank1])
        for tname in self.iter_map[rank1]:

            # Get the tensor rank corresponding to this loop rank
            tensor = self.program.get_equation().get_tensor(tname)
            tranks = [Symbol(trank.lower())
                      for trank in tensor.get_init_ranks()]
            trans = self.program.get_coord_math().get_cond_expr(
                part.get_root_name(rank), lambda expr: any(
                    trank in expr.atoms(Symbol) for trank in tranks))
            matches = [
                trank for trank in tranks if trank in trans.atoms(Symbol)]
            assert len(matches) == 1
            trank_root = str(matches[0])

            # Add that fiber to the eager input
            fiber_name = tname.lower() + "_" + trank_root + "1"
            self.graph.add_edge(FiberNode(fiber_name), eager_input_node)

        # Connect the IntervalNode
        self.graph.add_edge(LoopNode(rank1), IntervalNode(rank0))
        self.graph.add_edge(eager_input_node, IntervalNode(rank0))
        self.graph.add_edge(IntervalNode(rank0), LoopNode(rank0))

    def __build_static_part(self, tensor: Tensor,
                            partitioning: Tuple[str, ...]) -> None:
        """
        Build a static partitioning
        """
        part = self.program.get_partitioning()
        root = tensor.root_name()
        part_node = PartNode(root, partitioning)

        # Put a swizzle node if this is flattening
        if len(partitioning) > 1:
            swizzle_node = SwizzleNode(
                root, list(partitioning), "partitioning")

            for rank in partitioning:
                self.graph.add_edge(RankNode(root, rank), swizzle_node)
            self.program.apply_partition_swizzling(tensor)

            self.graph.add_edge(swizzle_node, part_node)

            # Add an additional swizzle node to ensure that the tensor always
            # starts in the correct order before being merged by a hardware
            # merger
            if self.metrics:
                init_ranks = self.metrics.get_merger_init_ranks(
                    root, tensor.get_ranks())
                if init_ranks:
                    metrics_swizzle_node = SwizzleNode(
                        root, init_ranks, "metrics")

                    for rank in init_ranks:
                        self.graph.add_edge(
                            RankNode(root, rank), metrics_swizzle_node)

                    self.graph.add_edge(metrics_swizzle_node, swizzle_node)

        # Otherwise, add the edge from the source rank to the partitioning
        else:
            self.graph.add_edge(RankNode(root, partitioning[0]), part_node)

        # Add edges to for all resulting ranks
        for res in part.partition_names(partitioning, False):
            self.graph.add_edge(part_node, RankNode(root, res))

        # We must do graphics after any static partitioning
        self.graph.add_edge(part_node, OtherNode("Graphics"))
        self.program.apply_partitioning(tensor, partitioning)

    def __build_swizzle_root_fiber(self, tensor: Tensor, static: bool) -> None:
        """
        Build a swizzleRanks(), getRoot(), and the resulting FiberNode
        """
        self.program.get_loop_order().apply(tensor)
        root = tensor.root_name()

        tensor_node = TensorNode(root)
        swizzle_node = SwizzleNode(
            root, tensor.get_ranks().copy(), "loop-order")
        get_root_node = GetRootNode(root, tensor.get_ranks().copy())
        fiber_node = FiberNode(tensor.fiber_name())

        self.graph.add_edge(tensor_node, swizzle_node)
        self.graph.add_edge(swizzle_node, get_root_node)
        self.graph.add_edge(get_root_node, fiber_node)

        # An SwizzleNode always requires the TensorNode and RankNodes of the
        # relevant ranks
        for rank in tensor.get_ranks():
            self.graph.add_edge(RankNode(root, rank), swizzle_node)

        # If this is a static swizzle, do it before the graphics
        if static:
            self.graph.add_edge(swizzle_node, OtherNode("Graphics"))

        # Add an additional swizzle node to ensure that the tensor always
        # starts in the correct order before being merged by a hardware merger
        if self.metrics:
            init_ranks = self.metrics.get_merger_init_ranks(
                root, tensor.get_ranks())
            if init_ranks:
                metrics_swizzle_node = SwizzleNode(root, init_ranks, "metrics")

                for rank in init_ranks:
                    self.graph.add_edge(
                        RankNode(
                            root,
                            rank),
                        metrics_swizzle_node)

                self.graph.add_edge(metrics_swizzle_node, swizzle_node)

    def __connect_dyn_part(self, tensor: Tensor, rank: str,
                           flatten_info: Dict[str, List[Tuple[str, ...]]]) -> None:
        """
        Connect the dynamic partitioning node to the relevant fiber nodes
        """
        fiber_node = FiberNode(tensor.fiber_name())
        root = tensor.root_name()

        tensor.from_fiber()
        self.program.apply_partitioning(tensor, (rank,))

        # Apply all available flattening
        i = 0
        while i < len(flatten_info[root]):
            if not all(flat_rank in tensor.get_ranks()
                       for flat_rank in flatten_info[root][i]):
                i += 1
                continue

            flatten = flatten_info[root].pop(i)
            self.program.apply_partition_swizzling(tensor)
            self.program.apply_partitioning(tensor, flatten)

            # Start searching from the beginning
            i = 0

        self.program.get_loop_order().apply(tensor)

        part_node = PartNode(root, (rank,))
        ff_node = FromFiberNode(root, rank)

        self.graph.add_edge(fiber_node, ff_node)
        self.graph.add_edge(ff_node, part_node)

        self.__build_swizzle_root_fiber(tensor, False)

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
