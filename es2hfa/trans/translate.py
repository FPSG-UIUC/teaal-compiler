"""
Translate an Einsum to the corresponding HFA code
"""

from typing import cast, List, Optional

from es2hfa.hfa.base import Statement
from es2hfa.hfa.stmt import SBlock, SFor
from es2hfa.ir.iter_graph import IterationGraph
from es2hfa.ir.mapping import Mapping
from es2hfa.parse.input import Input
from es2hfa.trans.equation import Equation
from es2hfa.trans.header import Header


class Translator:
    """
    Translate a given Einsum into the corresponding HFA code
    """

    @staticmethod
    def translate(input_: Input) -> Statement:
        """
        Perform the Einsum to HFA translation
        """
        mapping = Mapping(input_.get_declaration(), input_.get_rank_orders())

        program = SBlock([])
        for einsum in input_.get_expressions():
            mapping.add_einsum(einsum, input_.get_loop_orders())
            program.add(Header.make_header(mapping))

            graph = IterationGraph(mapping)
            eqn = Equation(einsum)
            program.add(Translator.__build_loop_nest(graph, eqn))

            mapping.reset()

        return cast(Statement, program)

    @staticmethod
    def __build_loop_nest(graph: IterationGraph, eqn: Equation) -> Statement:
        """
        Recursively build the loop nest
        """
        ind, tensors = graph.peek()

        # If we are at the bottom of the loop nest, build the update
        if not ind:
            return eqn.make_update()

        # Otherwise, get the information for the for loop
        expr = eqn.make_iter_expr(tensors)
        _, tensors = graph.pop()
        payload = eqn.make_payload(tensors)

        # Recurse for the for loop body
        body = Translator.__build_loop_nest(graph, eqn)

        return cast(Statement, SFor(ind, payload, expr, body))
