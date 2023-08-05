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

Translate an Einsum to the corresponding HiFiber code
"""

from typing import cast, List, Optional

from teaal.hifiber import *
from teaal.ir.flow_graph import FlowGraph
from teaal.ir.flow_nodes import *
from teaal.ir.hardware import Hardware
from teaal.ir.iter_graph import IterationGraph
from teaal.ir.metrics import Metrics
from teaal.ir.node import Node
from teaal.ir.program import Program
from teaal.parse import *
from teaal.trans.collector import Collector
from teaal.trans.graphics import Graphics
from teaal.trans.equation import Equation
from teaal.trans.footer import Footer
from teaal.trans.header import Header
from teaal.trans.partitioner import Partitioner
from teaal.trans.utils import TransUtils


class HiFiber:
    """
    Translate a given Einsum into the corresponding HiFiber code
    """

    def __init__(
            self,
            einsum: Einsum,
            mapping: Mapping,
            arch: Optional[Architecture] = None,
            bindings: Optional[Bindings] = None,
            format_: Optional[Format] = None) -> None:
        """
        Perform the Einsum to HiFiber translation
        """
        self.program = Program(einsum, mapping)

        self.hardware: Optional[Hardware] = None
        self.format = format_
        if arch and bindings and arch.get_spec():
            self.hardware = Hardware(arch, bindings, self.program)

        self.trans_utils = TransUtils()

        self.hifiber = SBlock([])
        for i in range(len(einsum.get_expressions())):
            self.hifiber.add(self.__translate(i))

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
        self.graphics = Graphics(self.program, self.metrics)
        self.partitioner = Partitioner(self.program, self.trans_utils)
        self.header = Header(self.program, self.metrics, self.partitioner)
        self.graph = IterationGraph(self.program)
        self.eqn = Equation(self.program, self.metrics)

        if self.metrics:
            self.collector = Collector(self.program, self.metrics)

        stmt = self.__trans_nodes(nodes)[1]

        self.program.reset()
        return stmt

    def __trans_nodes(self, nodes: List[Node]) -> Tuple[int, Statement]:
        """
        Recursive function to generate the actual HiFiber program
        """
        code = SBlock([])

        i = 0
        while i < len(nodes):
            node = nodes[i]

            if isinstance(node, CollectingNode):
                code.add(
                    self.collector.set_collecting(
                        node.get_tensor(),
                        node.get_rank(),
                        node.get_type(),
                        node.get_consumable(),
                        node.get_is_read_trace()))

            elif isinstance(node, ConsumeTraceNode):
                code.add(
                    self.collector.consume_traces(
                        node.get_component(),
                        node.get_rank()))

            elif isinstance(node, CreateComponentNode):
                code.add(
                    self.collector.create_component(
                        node.get_component(),
                        node.get_rank()))

            elif isinstance(node, EagerInputNode):
                code.add(
                    self.eqn.make_eager_inputs(
                        node.get_rank(),
                        node.get_tensors()))

            elif isinstance(node, EndLoopNode):
                return i + 1, code

            elif isinstance(node, FromFiberNode):
                tensor = self.program.get_equation().get_tensor(node.get_tensor())
                code.add(Header.make_tensor_from_fiber(tensor))

            elif isinstance(node, GetPayloadNode):
                tensor = self.program.get_equation().get_tensor(node.get_tensor())
                code.add(Header.make_get_payload(tensor, node.get_ranks()))

            elif isinstance(node, GetRootNode):
                tensor = self.program.get_equation().get_tensor(node.get_tensor())
                code.add(Header.make_get_root(tensor))

            elif isinstance(node, IntervalNode):
                code.add(self.eqn.make_interval(node.get_rank()))

            elif isinstance(node, LoopNode):
                # Generate the for loop
                rank, tensors = self.graph.peek_concord()
                expr = self.eqn.make_iter_expr(cast(str, rank), tensors)
                _, tensors = self.graph.pop_concord()
                payload = self.eqn.make_payload(cast(str, rank), tensors)

                # Recurse for the for loop body
                j, body = self.__trans_nodes(nodes[(i + 1):])
                code.add(SFor(payload, expr, body))
                i += j

            elif isinstance(node, MetricsNode):
                if node.get_type() == "Dump":
                    code.add(self.collector.dump())

                elif node.get_type() == "End":
                    code.add(self.collector.end())

                elif node.get_type() == "Start":
                    code.add(self.collector.start())

                else:
                    raise ValueError(
                        "Unknown node: " +
                        repr(node))  # pragma: no cover

            elif isinstance(node, MetricsHeaderNode):
                code.add(self.collector.make_loop_header(node.get_rank()))

            elif isinstance(node, OtherNode):
                if node.get_type() == "Body":
                    code.add(self.eqn.make_update())
                    code.add(self.graphics.make_body())
                    if self.metrics:
                        code.add(self.collector.make_body())

                elif node.get_type() == "Footer":
                    code.add(
                        Footer.make_footer(
                            self.program,
                            self.graphics,
                            self.partitioner))

                elif node.get_type() == "Graphics":
                    code.add(self.graphics.make_header())

                elif node.get_type() == "Output":
                    code.add(self.header.make_output())

                else:
                    raise ValueError(
                        "Unknown node: " +
                        repr(node))  # pragma: no cover

            elif isinstance(node, PartNode):
                tensor = self.program.get_equation().get_tensor(node.get_tensor())
                ranks = node.get_ranks()

                tensor.from_fiber()
                code.add(self.partitioner.partition(tensor, ranks))

            elif isinstance(node, RegisterRanksNode):
                code.add(self.collector.register_ranks(node.get_ranks()))

            elif isinstance(node, SwizzleNode):
                tensor = self.program.get_equation().get_tensor(node.get_tensor())
                code.add(
                    self.header.make_swizzle(
                        tensor,
                        node.get_ranks(),
                        node.get_type()))

            elif isinstance(node, TraceTreeNode):
                code.add(
                    self.collector.trace_tree(
                        node.get_tensor(),
                        node.get_rank(),
                        node.get_is_read_trace()))
            else:
                raise ValueError(
                    "Unknown node: " +
                    repr(node))  # pragma: no cover

            i += 1

        return i, code

    def __str__(self) -> str:
        """
        Return the string representation of this HiFiber program
        """

        return self.hifiber.gen(0)
