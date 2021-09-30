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

Representation of a tensor as it moves through the iteration graph
"""

from lark.tree import Tree
from typing import Dict, List, Optional


class Tensor:
    """
    Intermediate representation for a tensor
    """

    def __init__(self, name: str, inds: List[str]) -> None:
        """
        Construct a new tensor from a name and list of indices
        """
        # Check for no repeated indices
        if len(inds) > len(set(inds)):
            bad_tensor = name + ": [" + ", ".join(inds) + "]"
            raise ValueError("All indices must be unique; given " + bad_tensor)

        self.name = name
        self.inds = inds.copy()
        self.init_inds = self.inds.copy()

        # Set the index pointer and output status
        self.ind_ptr = 0
        self.is_output = False

    @classmethod
    def from_tensor(cls, parent: "Tensor") -> "Tensor":
        """
        Construct a new Tensor from the current fiber
        """
        child = cls(parent.root_name(), parent.get_inds()[parent.ind_ptr:])
        child.set_is_output(parent.get_is_output())

        return child

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

    def get_access(self) -> List[str]:
        """
        Return a (lowercase) list of indices for this tensor
        """
        return [ind[0].lower() + ind[1:] for ind in self.inds]

    def get_inds(self) -> List[str]:
        """
        Return a (capitalized) list of indices for this tensor
        """
        return self.inds

    def get_is_output(self) -> bool:
        """
        Returns true if this tensor is an output tensor
        """
        return self.is_output

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
        self.inds.sort(key=loop_order.index)

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
