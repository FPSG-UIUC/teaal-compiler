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

    def __init__(self, name: str, ranks: List[str]) -> None:
        """
        Construct a new tensor from a name and list of ranks
        """
        # Check for no repeated ranks
        if len(ranks) > len(set(ranks)):
            bad_tensor = name + ": [" + ", ".join(ranks) + "]"
            raise ValueError("All ranks must be unique; given " + bad_tensor)

        self.name = name
        self.ranks = ranks.copy()
        self.init_ranks = self.ranks.copy()

        # Set the ranks pointer and output status
        self.rank_ptr = 0
        self.is_output = False

    @classmethod
    def from_tensor(cls, parent: "Tensor") -> "Tensor":
        """
        Construct a new Tensor from the current fiber
        """
        child = cls(parent.root_name(), parent.get_ranks()[parent.rank_ptr:])
        child.set_is_output(parent.get_is_output())

        return child

    def fiber_name(self) -> str:
        """
        Return the current fiber name for this tensor
        """
        stub = self.name[0].lower() + self.name[1:] + "_"
        if self.rank_ptr < len(self.ranks):
            return stub + self.__get_rank()
        elif self.is_output:
            return stub + "ref"
        else:
            return stub + "val"

    def get_access(self) -> List[str]:
        """
        Return a (lowercase) list of ranks for this tensor
        """
        return [rank[0].lower() + rank[1:] for rank in self.ranks]

    def get_ranks(self) -> List[str]:
        """
        Return a (capitalized) list of ranks for this tensor
        """
        return self.ranks

    def get_is_output(self) -> bool:
        """
        Returns true if this tensor is an output tensor
        """
        return self.is_output

    def partition(self, partitioning: Dict[str, List[Tree]]) -> None:
        """
        Partition this tensor across all relevant ranks
        """
        for rank, parts in partitioning.items():
            if rank in self.ranks:
                # Remove the old rank
                i = self.ranks.index(rank)
                self.ranks.pop(i)

                # Insert the new ranks
                new_ranks = [rank + str(j) for j in range(len(parts) + 1)]
                for new_rank in new_ranks:
                    self.ranks.insert(i, new_rank)

    def peek(self) -> Optional[str]:
        """
        Peek at the top rank, returns None if there are no more ranks
        """
        if self.rank_ptr < len(self.ranks):
            return self.ranks[self.rank_ptr]
        return None

    def pop(self) -> str:
        """
        Pop off the top rank
        """
        rank = self.__get_rank()
        self.rank_ptr += 1
        return rank

    def reset(self) -> None:
        """
        Reset the tensor to its initial state
        """
        self.rank_ptr = 0
        self.ranks = self.init_ranks.copy()
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
        Re-order the ranks of this tensor to match the given loop order
        """
        self.ranks.sort(key=loop_order.index)

    def tensor_name(self) -> str:
        """
        Get the current name of the tensor
        """
        return self.name + "_" + "".join(self.ranks)

    def __eq__(self, other: object) -> bool:
        """
        The == operator for Tensors
        """
        if isinstance(other, type(self)):
            return self.name == other.name and \
                self.ranks == other.ranks and \
                self.is_output == other.is_output
        return False

    def __get_rank(self) -> str:
        """
        Get the name of the current rank of the tensor
        """
        rank = self.ranks[self.rank_ptr]
        return rank[0].lower() + rank[1:]
