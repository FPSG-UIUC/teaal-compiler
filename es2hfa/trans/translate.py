"""
Translate an Einsum to the corresponding HFA code
"""

from typing import cast, List, Optional

from es2hfa.hfa.base import Statement
from es2hfa.hfa.stmt import SBlock, SFor
from es2hfa.ir.iter_graph import IterationGraph
from es2hfa.ir.mapping import Mapping
from es2hfa.parse.input import Input
from es2hfa.trans.canvas import Canvas
from es2hfa.trans.equation import Equation
from es2hfa.trans.footer import Footer
from es2hfa.trans.header import Header
from es2hfa.trans.utils import Utils


class Translator:
    """
    Translate a given Einsum into the corresponding HFA code
    """

    @staticmethod
    def translate(input_: Input) -> Statement:
        """
        Perform the Einsum to HFA translation
        """
        mapping = Mapping(input_)
        utils = Utils()

        program = SBlock([])
        for i in range(len(input_.get_expressions())):
            # Add Einsum to mapping
            mapping.add_einsum(i)

            # Create a canvas
            canvas = Canvas(mapping)

            # Build the header
            program.add(Header.make_header(mapping, canvas, utils))

            # Build the loop nests
            graph = IterationGraph(mapping)
            eqn = Equation(mapping)
            program.add(Translator.__build_loop_nest(graph, eqn, canvas))

            # Build the footer
            program.add(Footer.make_footer(mapping, canvas, utils))

            mapping.reset()

        return cast(Statement, program)

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
        expr = eqn.make_iter_expr(tensors)
        _, tensors = graph.pop()
        payload = eqn.make_payload(ind, tensors)

        # Recurse for the for loop body
        body = Translator.__build_loop_nest(graph, eqn, canvas)

        return cast(Statement, SFor(payload, expr, body))
