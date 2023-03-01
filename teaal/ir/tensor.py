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

from collections import Counter

from lark.tree import Tree
from typing import Any, Dict, Iterable, List, Optional


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

        # Set the pointers and output status
        self.iter_ptr = 0
        self.rank_ptr = 0
        self.is_output = False
        self.is_flat = False

    def fiber_name(self) -> str:
        """
        Return the current fiber name for this tensor
        """
        stub = self.name.lower() + "_"
        if self.iter_ptr < len(self.ranks):
            return stub + self.__get_rank()
        elif self.is_output:
            return stub + "ref"
        else:
            return stub + "val"

    def from_fiber(self) -> None:
        """
        Construct a new Tensor from the current fiber
        """
        if self.is_flat:
            self.is_flat = self.rank_ptr == self.iter_ptr

        self.rank_ptr = self.iter_ptr

    def get_access(self) -> List[str]:
        """
        Return a (lowercase) list of ranks for this tensor
        """
        return [rank.lower() for rank in self.ranks[self.rank_ptr:]]

    def get_init_ranks(self) -> List[str]:
        """
        Get the inital set of ranks declared for this tensor (with no
        partitioning or swizzling)
        """
        return self.init_ranks

    def get_is_output(self) -> bool:
        """
        Returns true if this tensor is an output tensor
        """
        return self.is_output

    def get_prefix(self, rank: str) -> List[str]:
        """
        Get a list of ranks up to the current rank

        Note: "root" returns the empty list
        """
        if rank == "root":
            return []

        i = self.ranks.index(rank)
        return self.ranks[self.rank_ptr:(i + 1)]

    def get_ranks(self) -> List[str]:
        """
        Return a (capitalized) list of ranks for this tensor
        """
        return self.ranks[self.rank_ptr:]

    def peek(self) -> Optional[str]:
        """
        Peek at the top rank, returns None if there are no more ranks
        """
        if self.iter_ptr < len(self.ranks):
            return self.__get_rank()
        return None

    def peek_rest(self) -> List[str]:
        """
        Return the list of ranks that have not yet been iterated over for this
        tensor
        """
        return self.ranks[self.iter_ptr:]

    def pop(self) -> str:
        """
        Pop off the top rank
        """
        rank = self.__get_rank()
        self.iter_ptr += 1
        return rank

    def reset(self) -> None:
        """
        Reset the tensor to its initial state
        """
        self.iter_ptr = 0
        self.rank_ptr = 0
        self.ranks = self.init_ranks.copy()
        self.is_output = False
        self.is_flat = False

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

    def swizzle(self, rank_order: List[str]) -> None:
        """
        Re-order the ranks of this tensor to match the given rank order
        """
        old_active = self.ranks[self.rank_ptr:]

        # Ensure that the new rank order is just a permutation of the old rank
        # order
        if Counter(old_active) != Counter(rank_order):
            raise ValueError(
                str(rank_order) +
                " is not a permutation of old rank order " +
                str(old_active))

        self.ranks[self.rank_ptr:] = rank_order

        if self.is_flat:
            self.is_flat = old_active == rank_order

    def tensor_name(self) -> str:
        """
        Get the current name of the tensor
        """
        tname = self.name + "_" + "".join(self.ranks[self.rank_ptr:])
        if self.is_flat and not self.is_output:
            tname += "_flat"
        return tname

    def update_ranks(self, ranks: List[str]) -> None:
        """
        Update the ranks with a new list of ranks
        Note: usually requried for partitioning
        """
        self.is_flat = len(self.ranks) - self.rank_ptr > len(ranks)
        self.ranks = self.ranks[:self.rank_ptr] + ranks

    def __eq__(self, other: object) -> bool:
        """
        The == operator for Tensors
        """
        if isinstance(other, type(self)):
            return self.__key() == other.__key()
        return False

    def __get_rank(self) -> str:
        """
        Get the name of the current rank of the tensor
        """
        return self.ranks[self.iter_ptr].lower()

    def __key(self) -> Iterable[Any]:
        """
        Return an iterable of attributes
        """
        return self.name, self.ranks, self.is_output, self.iter_ptr, self.rank_ptr

    def __repr__(self) -> str:
        """
        Get a string representation of this object
        """
        strs = [key if isinstance(key, str) else repr(key)
                for key in self.__key()]
        return "(" + type(self).__name__ + ", " + ", ".join(strs) + ")"
