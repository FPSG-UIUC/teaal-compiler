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
from typing import List

from es2hfa.ir.nodes import *
from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor


class FlowGraph:
    """
    The control-dataflow graph for the HFA program
    """

    def __init__(self, program: Program) -> None:
        """
        Construct a new FlowGraph
        """
        self.program = program

        self.__build()
        self.__prune()

    def __build(self) -> None:
        """
        Build the flow graph
        """
        self.graph = nx.DiGraph()

        # Add the LoopNodes
        loop_order = self.program.get_loop_order().get_final_loop_order()

        # Add the edges connecting the LoopNodes
        src = loop_order[0]
        for dst in loop_order[1:]:
            self.graph.add_edge(LoopNode(src), LoopNode(dst))
            src = dst

        # Add SRNodes and FiberNodes for each tensor
        part = self.program.get_partitioning()
        for tensor in self.program.get_tensors():
            root = tensor.root_name()

            # Add the partitioning
            init_ranks = tensor.get_ranks()
            for rank in init_ranks:

                # Add the static partitioning
                if rank in part.get_static_parts():
                    part_node = PartNode(root, (rank,))

                    # Add the edge from the source rank to the partitioning
                    self.graph.add_edge(RankNode(root, rank), part_node)

                    # Add edges to for all resulting ranks
                    for res in part.partition_names(rank, False):
                        self.graph.add_edge(part_node, RankNode(root, res))

                    self.program.apply_partitioning(tensor, rank)

                # Add the dynamic partitioning
                if rank in part.get_dyn_parts():
                    for src in [rank] + part.get_intermediates(rank):
                        part_node = PartNode(root, (src,))
                        dsts = part.partition_names(src, False)
                        leader = part.get_leader(part.get_dyn_parts()[src][0])

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

            # Insert the header SRNode
            self.program.get_loop_order().apply(tensor)
            sr_node = SRNode(root, tensor.get_ranks().copy())
            fiber_node = FiberNode(tensor.fiber_name())
            self.graph.add_edge(sr_node, fiber_node)

            # The header SRNode requires RankNodes of the relevant ranks
            for rank in tensor.get_ranks():
                self.graph.add_edge(RankNode(root, rank), sr_node)

            # Insert all other fiber nodes
            active = tensor
            opt_rank = active.peek()
            while opt_rank is not None:
                rank = opt_rank.upper()
                # If this is a dynamically partitioned rank, add the relevant
                # SRNode and connection to the FiberNode
                if rank in part.get_dyn_parts():
                    active = Tensor.from_tensor(active)
                    self.program.apply_partitioning(active, rank)
                    self.program.get_loop_order().apply(active)

                    sr_node = SRNode(root, active.get_ranks().copy())
                    part_node = PartNode(root, (rank,))
                    new_fnode = FiberNode(active.fiber_name())

                    self.graph.add_edge(fiber_node, part_node)
                    self.graph.add_edge(part_node, sr_node)
                    self.graph.add_edge(sr_node, new_fnode)

                    fiber_node = new_fnode

                rank = active.pop().upper()
                new_fnode = FiberNode(active.fiber_name())

                # Add the nodes corresponding to the inputs and outputs of this
                # loop
                self.graph.add_edge(fiber_node, LoopNode(rank))
                self.graph.add_edge(LoopNode(rank), new_fnode)

                fiber_node = new_fnode
                opt_rank = active.peek()

            tensor.reset()

    def __prune(self) -> None:
        """
        Prune out all intermediate nodes
        """
        # Remove all FiberNodes
        nodes = [node for node in self.graph.nodes()
                 if isinstance(node, FiberNode) or isinstance(node, RankNode)]

        for node in nodes:
            # Connect all in and out edges
            for in_, _ in self.graph.in_edges(node):
                for _, out in self.graph.out_edges(node):
                    self.graph.add_edge(in_, out)

            # Remove the node
            self.graph.remove_node(node)

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
                seed=10))
        plt.savefig("foo.png")

    def sort(self) -> List[Node]:
        """
        Sort all nodes so that the generated code obeys all dependencies
        """
        # Get the sort that places the loop orders the latest
        best_pos = {}
        for rank in self.program.get_loop_order().get_final_loop_order():
            loop_node = LoopNode(rank)
            decs = nx.descendants(self.graph, loop_node)
            best_pos[loop_node] = len(self.graph.nodes()) - len(decs) - 1

        for sort in nx.all_topological_sorts(self.graph):
            if all(sort.index(node) == i for node, i in best_pos.items()):
                return sort

        # Note: we should never reach here, there is no way to test
        raise ValueError("Something is wrong...")  # pragma: no cover
