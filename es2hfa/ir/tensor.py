"""
Representation of a tensor as it moves through the iteration graph
"""
from typing import List, Optional

from lark.tree import Tree


class Tensor:
    """
    Intermediate representation for a tensor

    TODO: currently assumes lowercase indices from the einsum as opposed to
    the uppercase indices from the declaration
    """

    def __init__(self, tree: Tree) -> None:
        """
        Construct a new Tensor from its parse tree, and note if this is an
        output tensor
        """
        if tree.data == "output":
            self.is_output = True
        elif tree.data == "tensor":
            self.is_output = False
        else:
            raise ValueError("Input parse tree must be a tensor")

        # Extract the name and indices
        values = list(tree.scan_values(lambda _: True))
        self.name = values[0]
        self.inds = values[1:]

        # Set the index pointer
        self.ind_ptr = 0

    def root_name(self) -> str:
        """
        Return the name of the tensor as defined in the Einsum
        """
        return self.name

    def fiber_name(self) -> str:
        """
        Return the current fiber name for this tensor
        """
        stub = self.name[0].lower() + self.name[1:] + "_"
        if self.ind_ptr < len(self.inds):
            return stub + self.inds[self.ind_ptr]
        elif self.is_output:
            return stub + "ref"
        else:
            return stub + "val"

    def peek(self) -> Optional[str]:
        """
        Peek at the top index, returns None if there are no more indices
        """
        if self.ind_ptr < len(self.inds):
            return self.inds[self.ind_ptr]
        return None

    def pop(self) -> str:
        """
        Pop off the top index
        """
        ind = self.inds[self.ind_ptr]
        self.ind_ptr += 1
        return ind

    def swizzle(self, loop_order: List[Optional[str]]) -> None:
        """
        Re-order the indices of this tensor to match the given loop order
        """
        self.inds.sort(key=lambda i: loop_order.index(i))

    def __eq__(self, other: object) -> bool:
        """
        The == operator for Tensors
        """
        if isinstance(other, type(self)):
            return self.name == other.name and \
                self.inds == other.inds and \
                self.is_output == other.is_output
        return False
