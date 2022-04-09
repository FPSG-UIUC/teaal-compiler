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

Intermediate representation of the loop order information
"""

from lark.tree import Tree
from sympy import Symbol
from typing import Any, cast, Iterable, List, Optional

from es2hfa.ir.index_math import IndexMath
from es2hfa.ir.partitioning import Partitioning
from es2hfa.ir.tensor import Tensor


class LoopOrder:
    """
    An abstract representation of the loop order information
    """

    def __init__(self, equation: Tree, output: Tensor) -> None:
        """
        Create a new LoopOrder object
        """
        # Save the equation and output
        self.equation = equation
        self.output = output

        # Make placeholders for the loop order, index math, and partitioning
        self.ranks: Optional[List[str]] = None
        self.index_math: Optional[IndexMath] = None
        self.partitioning: Optional[Partitioning] = None

    def add(self,
            loop_order: Optional[List[str]],
            index_math: IndexMath,
            partitioning: Partitioning) -> None:
        """
        Add the loop order information, selecting the default loop order if
        one was not provided
        """
        self.index_math = index_math
        self.partitioning = partitioning

        # First build the final loop order
        if loop_order is None:
            self.ranks = self.__default_loop_order()
        else:
            self.ranks = loop_order

    def apply(self, tensor: Tensor) -> None:
        """
        Swizzle the tensor according to the ranks available
        """
        # Make sure that the loop order has been configured
        if self.ranks is None or self.partitioning is None:
            raise ValueError(
                "Unconfigured loop order. Make sure to first call add()")

        # Get the names of the final rank ids for the tensor
        final_ids = []
        for rank in tensor.get_ranks():
            final_ids.append(self.partitioning.get_final_rank_id(rank))

        # Order the current rank ids based on their final posititon
        order = self.ranks.copy()
        for i in range(len(self.ranks)):
            for j, rank in enumerate(tensor.get_ranks()):

                if self.is_ready(final_ids[j], i):
                    order[i] = rank
                    break

        tensor.swizzle(cast(List[Optional[str]], order))

    def get_ranks(self) -> List[str]:
        """
        Get the final loop order
        """
        # Make sure that the final loop order has been set
        if self.ranks is None:
            raise ValueError(
                "Unconfigured loop order. Make sure to first call add()")

        return self.ranks

    def is_ready(self, rank: str, pos: int) -> bool:
        """
        Returns true if all variables needed to compute a payload in the given
        rank should be iterated on at the given loop

        Assumes uppercase rank name
        """
        if self.ranks is None or self.index_math is None or self.partitioning is None:
            raise ValueError(
                "Unconfigured loop order. Make sure to first call add()")

        # If this rank is partitioned and this is not the inner-most rank, no
        # projecting should occur
        if not self.__innermost_rank(rank):
            return self.ranks[pos] == rank

        # Otherwise, check if we have all index variables available
        avail = [self.partitioning.get_root_name(lrank).lower(
        ) for lrank in self.ranks[:(pos + 1)] if self.__innermost_rank(lrank)]

        root = self.partitioning.get_root_name(rank).lower()
        math = self.index_math.get_trans(root)

        ready = all(str(ind) in avail for ind in math.atoms(Symbol))
        curr = Symbol(self.partitioning.get_root_name(
            self.ranks[pos]).lower()) in math.atoms(Symbol)

        return ready and curr

    def __default_loop_order(self) -> List[str]:
        """
        Compute the default loop order
        """
        if self.partitioning is None:
            raise ValueError("Must configure partitioning before loop order")

        loop_order = self.__default_loop_order_unpartitioned()

        for rank, parts in self.partitioning.get_all_parts().items():
            # Skip intermediate ranks
            if rank not in loop_order:
                continue

            # Remove the old rank
            i = loop_order.index(rank)
            loop_order.pop(i)

            # Insert the new ranks
            new_ranks = self.partitioning.partition_names(rank, True)
            for new_rank in new_ranks:
                loop_order.insert(i, new_rank)

        return loop_order

    def __default_loop_order_unpartitioned(self) -> List[str]:
        """
        Get the names of the ranks before partitioning
        """
        ranks = self.output.get_ranks().copy()

        for sum_ in self.equation.find_data("sum"):
            ranks += list(next(sum_.find_data("sranks")
                               ).scan_values(lambda _: True))

        return ranks

    def __eq__(self, other: object) -> bool:
        """
        The == operator for LoopOrders
        """
        if isinstance(other, type(self)):
            for field1, field2 in zip(self.__key(), other.__key()):
                if field1 != field2:
                    return False
            return True
        return False

    def __key(self) -> Iterable[Any]:
        """
        Get the fields of the LoopOrder
        """
        return self.equation, self.output, self.ranks, self.partitioning

    def __innermost_rank(self, rank: str) -> bool:
        """
        Returns true if the the given rank is the inner-most rank
        """
        if self.partitioning is None:
            raise ValueError(
                "Unconfigured loop order. Make sure to first call add()")

        suffix = rank[len(self.partitioning.get_root_name(rank)):]
        return len(suffix) == 0 or suffix == "0"
