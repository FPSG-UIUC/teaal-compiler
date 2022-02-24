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

Translate an Einsum to the corresponding HFA code
"""

from typing import cast, List, Optional

from es2hfa.hfa import *
from es2hfa.ir.flow_graph import FlowGraph
from es2hfa.ir.iter_graph import IterationGraph
from es2hfa.ir.nodes import *
from es2hfa.ir.program import Program
from es2hfa.parse.einsum import Einsum
from es2hfa.parse.mapping import Mapping
from es2hfa.trans.graphics import Graphics
from es2hfa.trans.equation import Equation
from es2hfa.trans.footer import Footer
from es2hfa.trans.header import Header
from es2hfa.trans.loop_nest import LoopNest
from es2hfa.trans.partitioner import Partitioner
from es2hfa.trans.utils import TransUtils


class HFA:
    """
    Translate a given Einsum into the corresponding HFA code
    """

    def __init__(self, einsum: Einsum, mapping: Mapping) -> None:
        """
        Perform the Einsum to HFA translation
        """
        self.einsum = einsum
        self.mapping = mapping

        program = Program(einsum, mapping)
        trans_utils = TransUtils()

        code = SBlock([])
        for i in range(len(einsum.get_expressions())):
            # Add Einsum to program
            program.add_einsum(i)

            # Create a graphics object
            graphics = Graphics(program)

            # Create a partitioner
            partitioner = Partitioner(program, trans_utils)

            # Build the header
            header = Header(program, partitioner)
            code.add(header.make_global_header(graphics))

            # Build the loop nests
            graph = IterationGraph(program)
            eqn = Equation(program)
            code.add(LoopNest.make_loop_nest(eqn, graph, graphics, header))

            # Build the footer
            code.add(Footer.make_footer(program, graphics, partitioner))

            program.reset()

        self.hfa = cast(Statement, code)

        self.program = Program(self.einsum, self.mapping)
        self.trans_utils = TransUtils()

    def translate(self, i: int) -> Statement:
        """
        Generate a single loop nest
        """
        # Generate for the given einsum
        self.program.add_einsum(i)

        # Create the flow graph and get the relevant nodes
        flow_graph = FlowGraph(self.program)
        self.fgraph = flow_graph
        nodes = flow_graph.get_sorted()

        # Create all relevant translator objects
        self.graphics = Graphics(self.program)
        self.partitioner = Partitioner(self.program, self.trans_utils)
        self.header = Header(self.program, self.partitioner)
        self.graph = IterationGraph(self.program)
        self.eqn = Equation(self.program)

        stmt = self.__trans_nodes(nodes, 0)[1]

        self.program.reset()
        return stmt

    def __trans_nodes(self,
                      nodes: List[Node],
                      depth: int) -> Tuple[int,
                                           Statement]:
        """
        Recursive function to generate the actual HFA program
        """
        code = SBlock([])

        i = 0
        while i < len(nodes):
            node = nodes[i]
            if isinstance(node, FromFiberNode):
                tensor = self.program.get_tensor(node.get_tensor())
                code.add(Header.make_tensor_from_fiber(tensor))

            elif isinstance(node, LoopNode):
                # Generate the for loop
                rank, tensors = self.graph.peek()
                expr = self.eqn.make_iter_expr(cast(str, rank), tensors)
                _, tensors = self.graph.pop()
                payload = self.eqn.make_payload(cast(str, rank), tensors)

                # Recurse for the for loop body
                j, body = self.__trans_nodes(nodes[(i + 1):], depth + 1)
                code.add(cast(Statement, SFor(payload, expr, body)))
                i += j

            elif isinstance(node, OtherNode):
                if node.get_type() == "Body":
                    code.add(self.eqn.make_update())
                    code.add(self.graphics.make_body())

                elif node.get_type() == "Footer":
                    if depth == 0:
                        code.add(
                            Footer.make_footer(
                                self.program,
                                self.graphics,
                                self.partitioner))

                    else:
                        # Pop back up a level and retry this node
                        return i, cast(Statement, code)

                elif node.get_type() == "Graphics":
                    code.add(self.graphics.make_header())

                elif node.get_type() == "Output":
                    code.add(self.header.make_output())

                else:
                    raise ValueError(
                        "Unknown node: " +
                        repr(node))  # pragma: no cover

            elif isinstance(node, PartNode):
                rank = node.get_ranks()[0]
                self.program.start_partitioning(rank)

                tensor = self.program.get_tensor(node.get_tensor())
                code.add(self.partitioner.partition(tensor, {rank}))

            elif isinstance(node, SRNode):
                tensor = self.program.get_tensor(node.get_tensor())
                code.add(self.header.make_swizzle_root(tensor))

            else:
                raise ValueError(
                    "Unknown node: " +
                    repr(node))  # pragma: no cover

            i += 1

        return i, cast(Statement, code)

    def __str__(self) -> str:
        """
        Return the string representation of this HFA program
        """

        return self.hfa.gen(0)
