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
from es2hfa.ir.hardware import Hardware
from es2hfa.ir.iter_graph import IterationGraph
from es2hfa.ir.metrics import Metrics
from es2hfa.ir.nodes import *
from es2hfa.ir.program import Program
from es2hfa.parse import *
from es2hfa.trans.collector import Collector
from es2hfa.trans.graphics import Graphics
from es2hfa.trans.equation import Equation
from es2hfa.trans.footer import Footer
from es2hfa.trans.header import Header
from es2hfa.trans.partitioner import Partitioner
from es2hfa.trans.utils import TransUtils


class HFA:
    """
    Translate a given Einsum into the corresponding HFA code
    """

    def __init__(
            self,
            einsum: Einsum,
            mapping: Mapping,
            arch: Optional[Architecture] = None,
            bindings: Optional[Bindings] = None,
            format_: Optional[Format] = None) -> None:
        """
        Perform the Einsum to HFA translation
        """
        self.program = Program(einsum, mapping)

        self.hardware: Optional[Hardware] = None
        self.format = format_
        if arch and bindings and arch.get_spec():
            self.hardware = Hardware(arch, bindings)

        self.trans_utils = TransUtils()

        self.hfa = SBlock([])
        for i in range(len(einsum.get_expressions())):
            self.hfa.add(self.__translate(i))

    def __translate(self, i: int) -> Statement:
        """
        Generate a single loop nest
        """
        # Generate for the given einsum
        self.program.add_einsum(i)

        # Build metrics if there is hardware
        self.metrics: Optional[Metrics] = None
        if self.hardware and self.format:
            self.metrics = Metrics(self.program, self.hardware, self.format)

        # Create the flow graph and get the relevant nodes
        flow_graph = FlowGraph(self.program, self.metrics, ["hoist"])
        nodes = flow_graph.get_sorted()

        # Create all relevant translator objects
        self.graphics = Graphics(self.program)
        self.partitioner = Partitioner(self.program, self.trans_utils)
        self.header = Header(self.program, self.partitioner)
        self.graph = IterationGraph(self.program)
        self.eqn = Equation(self.program)

        if self.metrics:
            self.collector = Collector(self.program, self.metrics)

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
                code.add(SFor(payload, expr, body))
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
                        return i, code

                elif node.get_type() == "Graphics":
                    code.add(self.graphics.make_header())

                elif node.get_type() == "Output":
                    code.add(self.header.make_output())

                else:
                    raise ValueError(
                        "Unknown node: " +
                        repr(node))  # pragma: no cover

            elif isinstance(node, PartNode):
                tensor = self.program.get_tensor(node.get_tensor())
                rank = node.get_ranks()[0]

                tensor.from_fiber()
                code.add(self.partitioner.partition(tensor, rank))

            elif isinstance(node, SwizzleNode):
                tensor = self.program.get_tensor(node.get_tensor())
                code.add(self.header.make_swizzle(tensor))

            elif isinstance(node, GetRootNode):
                tensor = self.program.get_tensor(node.get_tensor())
                code.add(Header.make_get_root(tensor))

            elif isinstance(node, EagerInputNode):
                code.add(
                    self.eqn.make_eager_inputs(
                        node.get_rank(),
                        node.get_tensors()))

            elif isinstance(node, IntervalNode):
                code.add(self.eqn.make_interval(node.get_rank()))

            elif isinstance(node, MetricsNode):
                if node.get_type() == "Dump":
                    code.add(self.collector.dump())

                elif node.get_type() == "End":
                    if depth == 0:
                        code.add(self.collector.end())
                    else:
                        # Pop back up a level and retry this node
                        return i, code

                elif node.get_type() == "Start":
                    code.add(self.collector.start())

                else:
                    raise ValueError(
                        "Unknown node: " +
                        repr(node))  # pragma: no cover

            elif isinstance(node, CollectingNode):
                code.add(
                    self.collector.set_collecting(
                        node.get_tensor(),
                        node.get_rank()))

            else:
                raise ValueError(
                    "Unknown node: " +
                    repr(node))  # pragma: no cover

            i += 1

        return i, code

    def __str__(self) -> str:
        """
        Return the string representation of this HFA program
        """

        return self.hfa.gen(0)
