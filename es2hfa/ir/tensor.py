"""
Representation of a tensor as it moves through the iteration graph
"""
from typing import List, Optional

from lark.tree import Tree


class Tensor:
    """
    Intermediate representation for a tensor
    """

    def __init__(self, tree: Tree) -> None:
        """
        Construct a new Tensor from its parse tree
        """
        # TODO: remove allowing output at this point
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
        self.init_inds = self.inds.copy()

        # Set the index pointer
        self.ind_ptr = 0

    def fiber_name(self) -> str:
        """
        Return the current fiber name for this tensor
        """
        stub = self.name[0].lower() + self.name[1:] + "_"
        if self.ind_ptr < len(self.inds):
            return stub + self.__get_ind()
        elif self.is_output:
            return stub + "ref"
        else:
            return stub + "val"

    def get_inds(self) -> List[str]:
        """
        Return a list of indices for this tensor
        """
        return self.inds

    def peek(self) -> Optional[str]:
        """
        Peek at the top index, returns None if there are no more indices
        """
        if self.ind_ptr < len(self.inds):
            return self.__get_ind()
        return None

    def pop(self) -> str:
        """
        Pop off the top index
        """
        ind = self.__get_ind()
        self.ind_ptr += 1
        return ind

    def reset(self) -> None:
        """
        Reset the tensor to its initial state
        """
        self.ind_ptr = 0
        self.inds = self.init_inds
        self.is_output = False

    def root_name(self) -> str:
        """
        Return the name of the tensor as defined in the Einsum
        """
        return self.name

    def set_is_output(self, is_output: bool) -> None:
        """
        Specify if this is the output tensor
        """
        self.is_output = is_output

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

    def __get_ind(self) -> str:
        """
        Get the name of the current index of the tensor
        """
        ind = self.inds[self.ind_ptr]
        return ind[0].lower() + ind[1:]
