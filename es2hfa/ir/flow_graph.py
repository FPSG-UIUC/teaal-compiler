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

from es2hfa.ir.nodes import *
from es2hfa.ir.program import Program


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
        self.__clean()

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

        # Add the RankNodes for the pre-partitioned ranks
        for tensor in self.program.get_tensors():
            for rank in tensor.get_ranks():
                # self.graph.add_node(RankNode(tensor.root_name(), rank))
                pass

        # Add SRNodes and FiberNodes for each tensor
        outer_loop = LoopNode(loop_order[0])
        for tensor in self.program.get_tensors():
            self.program.apply_curr_loop_order(tensor)

            sr_node = SRNode(tensor.root_name(), set(tensor.get_ranks()))
            fiber_node = FiberNode(tensor.fiber_name())

            # Insert the header
            self.graph.add_edge(sr_node, fiber_node)

            # Insert all other fiber nodes
            while tensor.peek() is not None:
                rank = tensor.pop().upper()
                new_fnode = FiberNode(tensor.fiber_name())

                self.graph.add_edge(fiber_node, LoopNode(rank))
                self.graph.add_edge(LoopNode(rank), new_fnode)

                fiber_node = new_fnode

            tensor.reset()

        # plt.figure(figsize=(8, 6))
        # nx.draw(self.graph, with_labels=True, font_size=8, pos=nx.random_layout(self.graph, seed=11))
        # plt.savefig("foo.png")

    def __clean(self) -> None:
        """
        Clean out all intermediate nodes
        """

        # Remove all FiberNodes
        nodes = [
            node for node in self.graph.nodes() if isinstance(
                node, FiberNode)]

        for node in nodes:
            # Connect all in and out edges
            for in_, _ in self.graph.in_edges(node):
                for _, out in self.graph.out_edges(node):
                    self.graph.add_edge(in_, out)

            # Remove the node
            self.graph.remove_node(node)

    def sort(self) -> None:
        """
        Sort all nodes so that the generated code obeys all dependencies
        """
        # Get the sort that places the loop orders the latest
        best_sort = None
        loop_ind_sum = -float('inf')

        for sort in nx.all_topological_sorts(self.graph):
            sum_ = 0
            for rank in self.program.get_loop_order().get_final_loop_order():
                sum_ += sort.index(LoopNode(rank))

            if sum_ > loop_ind_sum:
                best_sort = sort

        return sort
