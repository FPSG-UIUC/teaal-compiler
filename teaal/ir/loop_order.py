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

from itertools import chain

from lark.tree import Tree
from sympy import Basic, Symbol
from typing import Any, cast, Iterable, List, Optional, Set, Tuple

from teaal.ir.coord_math import CoordMath
from teaal.ir.equation import Equation
from teaal.ir.partitioning import Partitioning
from teaal.ir.tensor import Tensor
from teaal.parse.utils import ParseUtils


class LoopOrder:
    """
    An abstract representation of the loop order information
    """

    def __init__(self, equation: Equation) -> None:
        """
        Create a new LoopOrder object
        """
        # Save the equation and output
        self.equation = equation

        # Make placeholders for the loop order, coord math, and partitioning
        self.ranks: Optional[List[str]] = None
        self.coord_math: Optional[CoordMath] = None
        self.partitioning: Optional[Partitioning] = None

    def add(self,
            loop_order: Optional[List[str]],
            coord_math: CoordMath,
            partitioning: Partitioning) -> None:
        """
        Add the loop order information, selecting the default loop order if
        one was not provided
        """
        self.coord_math = coord_math
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
            final_ids.append(self.partitioning.get_final_rank_id(tensor, rank))

        # Order the current rank ids based on their final posititon
        expanded: List[List[str]] = [[] for _ in range(len(self.ranks))]
        for i, rank in enumerate(tensor.get_ranks()):
            for j in range(len(self.ranks)):
                if self.is_ready(final_ids[i], j):
                    expanded[j].append(rank)
                    break

        order = list(chain.from_iterable(expanded))
        tensor.swizzle(order)

    def get_available_roots(self) -> Set[str]:
        """
        Get the root names for ranks actually available for index math
        """
        if self.partitioning is None or self.ranks is None:
            raise ValueError(
                "Unconfigured loop order. Make sure to first call add()")

        def trans_names(r):
            ranks = self.partitioning.get_available(r)
            return {self.partitioning.get_root_name(
                rank) for rank in ranks if not self.partitioning.is_flattened(rank)}

        names = [trans_names(rank) for rank in self.ranks]
        if names:
            return set.union(*names)

        return set()

    def get_iter_ranks(self, rank: str) -> Tuple[str, ...]:
        """
        Get the ranks that this rank should expand into when iterating over
        this rank; used to unpack flattened ranks
        """
        if self.partitioning is None:
            raise ValueError(
                "Unconfigured loop order. Make sure to first call add()")

        # If it is not flattened, nothing to unpack
        if not self.partitioning.is_flattened(rank):
            return (rank,)

        # If it is not the innermost rank, nothing to unpack
        if not self.__innermost_rank(rank):
            return (rank,)

        # Otherwise, unpack the flattened rank
        return self.partitioning.unpack(rank)

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
        if self.ranks is None or self.coord_math is None or self.partitioning is None:
            raise ValueError(
                "Unconfigured loop order. Make sure to first call add()")

        # If this rank is partitioned and this is not the inner-most rank, we
        # don't need the other indices in the index math
        if not self.__innermost_rank(rank):
            return self.partitioning.partition_rank(
                (self.ranks[pos],)) == self.partitioning.partition_rank(
                (rank,))

        # Otherwise, check if we have all index variables available
        avail = [self.partitioning.get_root_name(lrank).lower()
                 for lrank in self.ranks[:(pos + 1)]
                 if self.__innermost_rank(lrank)]

        # Translate if the rank is not flattened
        root = self.partitioning.get_root_name(rank).lower()
        if self.partitioning.is_flattened(rank):
            math = cast(Basic, Symbol(root))
        else:
            math = self.coord_math.get_trans(root)

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

        unpartitioned_loop_order = self.equation.get_einsum_ranks()
        loop_order = self.partitioning.partition_ranks(
            unpartitioned_loop_order, self.partitioning.get_all_parts(), True, True)
        return loop_order

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
        return self.equation, self.ranks, self.partitioning

    def __innermost_rank(self, rank: str) -> bool:
        """
        Returns true if the the given rank is the inner-most rank
        """
        if self.partitioning is None:
            raise ValueError(
                "Unconfigured loop order. Make sure to first call add()")

        _, suffix = self.partitioning.split_rank_name(rank)
        return len(suffix) == 0 or suffix == "0"
