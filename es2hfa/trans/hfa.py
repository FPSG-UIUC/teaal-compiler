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

from es2hfa.hfa.base import Statement
from es2hfa.hfa.stmt import SBlock, SFor
from es2hfa.ir.iter_graph import IterationGraph
from es2hfa.ir.program import Program
from es2hfa.parse.einsum import Einsum
from es2hfa.parse.mapping import Mapping
from es2hfa.trans.canvas import Canvas
from es2hfa.trans.equation import Equation
from es2hfa.trans.footer import Footer
from es2hfa.trans.header import Header
from es2hfa.trans.utils import TransUtils


class HFA:
    """
    Translate a given Einsum into the corresponding HFA code
    """

    def __init__(self, einsum: Einsum, mapping: Mapping) -> None:
        """
        Perform the Einsum to HFA translation
        """
        program = Program(einsum, mapping)
        trans_utils = TransUtils()

        code = SBlock([])
        for i in range(len(einsum.get_expressions())):
            # Add Einsum to program
            program.add_einsum(i)

            # Create a canvas
            canvas = Canvas(program)

            # Build the header
            code.add(Header.make_header(program, canvas, trans_utils))

            # Build the loop nests
            graph = IterationGraph(program)
            eqn = Equation(program)
            code.add(HFA.__build_loop_nest(graph, eqn, canvas))

            # Build the footer
            code.add(Footer.make_footer(program, canvas, trans_utils))

            program.reset()

        self.hfa = cast(Statement, code)

    @staticmethod
    def __build_loop_nest(
            graph: IterationGraph,
            eqn: Equation,
            canvas: Canvas) -> Statement:
        """
        Recursively build the loop nest
        """
        ind, tensors = graph.peek()

        # If we are at the bottom of the loop nest, build the update
        if not ind:
            bottom = SBlock([eqn.make_update()])

            # Add the canvas information if possible
            if canvas.displayable():
                bottom.add(canvas.add_activity())

            return cast(Statement, bottom)

        # Otherwise, get the information for the for loop
        expr = eqn.make_iter_expr(ind, tensors)
        _, tensors = graph.pop()
        payload = eqn.make_payload(ind, tensors)

        # Recurse for the for loop body
        body = HFA.__build_loop_nest(graph, eqn, canvas)

        return cast(Statement, SFor(payload, expr, body))

    def __str__(self) -> str:
        """
        Return the string representation of this HFA program
        """

        return self.hfa.gen(0)
