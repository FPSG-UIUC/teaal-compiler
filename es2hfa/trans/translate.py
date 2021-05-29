"""
Translate an Einsum to the corresponding HFA code
"""

from typing import cast, List, Optional

from lark.tree import Tree

from es2hfa.hfa.base import Statement
from es2hfa.hfa.stmt import SFor
from es2hfa.ir.iter_graph import IterationGraph
from es2hfa.trans.equation import Equation


class Translator:
    """
    Translate a given Einsum into the corresponding HFA code
    """

    @staticmethod
    def translate(einsum: Tree, loop_order: Optional[List[str]]) -> Statement:
        """
        Perform the Einsum to HFA translation
        """
        graph = IterationGraph(einsum, loop_order)
        eqn = Equation(einsum)

        return Translator.__build_loop_nest(graph, eqn)

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
