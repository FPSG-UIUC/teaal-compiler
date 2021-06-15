"""
Representation of a tensor as it moves through the iteration graph
"""
from typing import Dict, List, Optional

from lark.tree import Tree


class Tensor:
    """
    Intermediate representation for a tensor
    """

    def __init__(self, tree: Tree) -> None:
        """
        Construct a new Tensor from its parse tree
        """
        if tree.data != "tensor":
            raise ValueError("Input parse tree must be a tensor")

        # Extract the name and indices
        values = list(tree.scan_values(lambda _: True))
        self.name = values[0]
        self.inds = values[1:]
        self.init_inds = self.inds.copy()

        # Set the index pointer and output status
        self.ind_ptr = 0
        self.is_output = False

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

    def partition(self, partitioning: Dict[str, List[Tree]]) -> None:
        """
        Partition this tensor across all relevant dimensions
        """
        for ind, parts in partitioning.items():
            if ind in self.inds:
                # Remove the old index
                i = self.inds.index(ind)
                self.inds.pop(i)

                # Insert the new indices
                new_inds = [ind + str(j) for j in range(len(parts) + 1)]
                for new_ind in new_inds:
                    self.inds.insert(i, new_ind)

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
        ind = self.__get_ind()
        self.ind_ptr += 1
        return ind

    def reset(self) -> None:
        """
        Reset the tensor to its initial state
        """
        self.ind_ptr = 0
        self.inds = self.init_inds.copy()
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

    def tensor_name(self) -> str:
        """
        Get the current name of the tensor
        """
        return self.name + "_" + "".join(self.inds)

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
