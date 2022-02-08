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
from typing import List, Optional

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

        # Make placeholders for the loop orders and partitioning
        self.curr_loop_order: Optional[List[str]] = None
        self.final_loop_order: Optional[List[str]] = None
        self.partitioning: Optional[Partitioning] = None

    def add_loop_order(self,
                       loop_order: Optional[List[str]],
                       partitioning: Partitioning) -> None:
        """
        Add the loop order information, selecting the default loop order if
        one was not provided
        """
        self.partitioning = partitioning

        # First build the final loop order
        if loop_order is None:
            self.final_loop_order = self.__default_loop_order()
        else:
            self.final_loop_order = loop_order

        # Update the current loop order
        self.update_loop_order()

    def get_curr_loop_order(self) -> List[str]:
        """
        Get the current loop order
        """
        # Make sure that the final loop order has been set
        if self.curr_loop_order is None:
            raise ValueError(
                "Unconfigured loop order. Make sure to first call add_loop_order()")

        return self.curr_loop_order

    def get_final_loop_order(self) -> List[str]:
        """
        Get the final loop order
        """
        # Make sure that the final loop order has been set
        if self.final_loop_order is None:
            raise ValueError(
                "Unconfigured loop order. Make sure to first call add_loop_order()")

        return self.final_loop_order

    def get_unpartitioned_ranks(self) -> List[str]:
        """
        Get the names of the ranks before partitioning
        """
        ranks = self.output.get_ranks().copy()

        for sum_ in self.equation.find_data("sum"):
            ranks += list(next(sum_.find_data("sranks")
                               ).scan_values(lambda _: True))

        return ranks

    def update_loop_order(self) -> None:
        """
        Update the loop order with the latest partitioning information
        """
        # Make sure that the final loop order has been set
        if self.final_loop_order is None or self.partitioning is None:
            raise ValueError(
                "Unconfigured loop order. Make sure to first call add_loop_order()")

        # Compute the current loop order
        self.curr_loop_order = []
        for rank in self.final_loop_order:
            opt_rank = self.partitioning.get_curr_rank_id(rank)
            if opt_rank:
                self.curr_loop_order.append(opt_rank)

    def __default_loop_order(self) -> List[str]:
        """
        Compute the default loop order
        """
        if self.partitioning is None:
            raise ValueError("Must configure partitioning before loop order")

        loop_order = self.get_unpartitioned_ranks()

        for rank, parts in self.partitioning.get_all_parts().items():
            # Remove the old rank
            i = loop_order.index(rank)
            loop_order.pop(i)

            # Insert the new ranks
            new_ranks = [rank + str(j) for j in range(len(parts) + 1)]
            for new_rank in new_ranks:
                loop_order.insert(i, new_rank)

        return loop_order
