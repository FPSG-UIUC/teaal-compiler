"""
Represetation of an iteration graph
"""
from typing import cast, List, Optional, Tuple

from lark.tree import Tree

from es2hfa.ir.tensor import Tensor


class IterationGraph:
    """
    A graph storing the tensor IR used to build the loop nests

    TODO: currently uses the einsum as opposed to the declaration to build the
    tensors
    TODO: Currently calculates default loop order. If this is calculated earlier
    make this parameter no longer optional
    """

    def __init__(self, einsum: Tree, loop_order: Optional[List[str]]) -> None:
        """
        Construct a new IterationGraph
        """
        # Make sure we are starting with the full einsum
        if einsum.data != "einsum":
            raise ValueError("Input parse tree must be an einsum")

        # First, check if the default loop order should be computed
        default_loop_order = not loop_order
        self.loop_order = []
        if loop_order is not None:
            self.loop_order = cast(List[Optional[str]], loop_order)

        # Build the list of tensors, starting with the output tensor
        tensors = []
        tensor_tree = next(einsum.find_data("output"))
        tensors.append(Tensor(tensor_tree))

        # We need to use the output tensor to extract the default loop order
        if default_loop_order:
            self.loop_order = list(
                next(tensor_tree.find_data("tinds")).scan_values(
                    lambda _: True))

        # Add in the rest of the tensors
        for tensor_tree in einsum.find_data("tensor"):
            tensors.append(Tensor(tensor_tree))

        # Now, build the rest of the self.loop_order if necessary
        if default_loop_order:
            for sum_ in einsum.find_data("sum"):
                self.loop_order += [ind[0].lower() + ind[1:] for ind in
                                    next(sum_.find_data("sinds")).scan_values(
                    lambda _: True)]

        # Add a None to the end of the loop_order for the bottom of the loop
        # nest
        self.loop_order += [None]

        # Swizzle each of the tensors to use the designated loop order
        for tensor in tensors:
            tensor.swizzle(self.loop_order)

        # Place the tensors in the appropriate locations in the iteration graph
        self.graph: List[List[Tensor]] = []
        for ind in self.loop_order:
            self.graph.append([])
            for tensor in tensors:
                if tensor.peek() == ind:
                    self.graph[-1].append(tensor)

    def peek(self) -> Tuple[Optional[str], List[Tensor]]:
        """
        Peek at the next iteration
        """
        return self.loop_order[0], self.graph[0]

    def pop(self) -> Tuple[Optional[str], List[Tensor]]:
        """
        Pop the next iteration off the graph
        """
        # Pop off the next iteration
        ind = self.loop_order.pop(0)
        tensors = self.graph.pop(0)

        # Update each of the tensors and re-insert them into the graph at the
        # appropriate location
        for tensor in tensors:
            tensor.pop()
            self.graph[self.loop_order.index(tensor.peek())].append(tensor)

        return ind, tensors
