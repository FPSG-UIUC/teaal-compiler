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

Translate the loop nest of an Einsum to the corresponding HFA code
"""

from typing import cast

from es2hfa.hfa import *
from es2hfa.ir.iter_graph import IterationGraph
from es2hfa.trans.equation import Equation
from es2hfa.trans.graphics import Graphics


class LoopNest:
    """
    Translate the loop nest of an Einsum to the corresponding HFA code
    """

    @staticmethod
    def make_loop_nest(
            graph: IterationGraph,
            eqn: Equation,
            graphics: Graphics) -> Statement:
        """
        Recursively build the loop nest
        """
        ind, tensors = graph.peek()

        # If we are at the bottom of the loop nest, build the update
        if not ind:
            bottom = SBlock([eqn.make_update()])

            # Add the graphics information if possible
            bottom.add(graphics.make_body())

            return cast(Statement, bottom)

        # Otherwise, get the information for the for loop
        expr = eqn.make_iter_expr(ind, tensors)
        _, tensors = graph.pop()
        payload = eqn.make_payload(ind, tensors)

        # Recurse for the for loop body
        body = LoopNest.make_loop_nest(graph, eqn, graphics)

        return cast(Statement, SFor(payload, expr, body))