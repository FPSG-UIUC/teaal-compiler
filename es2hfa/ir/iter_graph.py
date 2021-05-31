"""
Represetation of an iteration graph
"""
from typing import cast, List, Optional, Tuple

from lark.tree import Tree

from es2hfa.ir.mapping import Mapping
from es2hfa.ir.tensor import Tensor


class IterationGraph:
    """
    A graph storing the tensor IR used to build the loop nests
    """

    def __init__(self, mapping: Mapping) -> None:
        """
        Construct a new IterationGraph
        """
        # Add a None to the end of the loop_order for the bottom of the loop
        # nest
        self.loop_order = cast(List[Optional[str]],
                               mapping.get_loop_order().copy()) + [None]

        # Place the tensors in the appropriate locations in the iteration graph
        self.graph: List[List[Tensor]] = []
        for ind in self.loop_order:
            self.graph.append([])
            for tensor in mapping.get_tensors():
                if tensor.peek() == ind:
                    self.graph[-1].append(tensor)

    def peek(self) -> Tuple[Optional[str], List[Tensor]]:
        """
        Peek at the next iteration
        """
        ind = self.loop_order[0]
        if ind:
            ind = ind[0].lower() + ind[1:]

        return ind, self.graph[0]

    def pop(self) -> Tuple[Optional[str], List[Tensor]]:
        """
        Pop the next iteration off the graph
        """
        # Pop off the next iteration
        ind = self.loop_order.pop(0)
        if ind:
            ind = ind[0].lower() + ind[1:]

        tensors = self.graph.pop(0)

        # Update each of the tensors and re-insert them into the graph at the
        # appropriate location
        for tensor in tensors:
            tensor.pop()
            self.graph[self.loop_order.index(tensor.peek())].append(tensor)

        return ind, tensors
