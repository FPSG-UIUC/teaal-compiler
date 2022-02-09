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

Intermediate representation of the partitioning information
"""

from lark.tree import Tree
from typing import Dict, Iterable, List, Optional

from es2hfa.parse.utils import ParseUtils


class Partitioning:
    """
    An abstract representation of the partitioning information
    """

    def __init__(self,
                 partitioning: Dict[str,
                                    List[Tree]],
                 ranks: List[str]) -> None:
        """
        Create a new representation of the partitioning information
        """
        # Filter the partitioning information into the ranks that can
        # be partitioned statically vs dynamically
        self.dyn_parts = {}
        self.static_parts = {}

        for rank, parts in partitioning.items():

            # Continue if this rank is not actually partitioned
            if not parts:
                continue

            # Make sure that the rank is either partitioned completely
            # statically or completely dynamically
            static = Partitioning.__is_static(parts[0])
            for part in parts[1:]:
                if Partitioning.__is_static(part) != static:
                    raise ValueError(
                        "Rank " + rank + " cannot be partitioned both statically and dynamically")

            # Add the partitioning specification to the appropriate dictionary
            if static:
                self.static_parts[rank] = parts
            else:
                self.dyn_parts[rank] = parts

        self.all_parts = {**self.static_parts, **self.dyn_parts}

        # Build a dictionary from final rank name to an optional rank name
        # where the value is determined by:
        # - If the rank is unpartitioned, the value is the same as the key
        # - If the rank has already been partitioned, value == key
        # - If the rank will be partitioned
        #    - If this is the largest rank name in this original rank, the
        #      the value is the initial rank name
        #    - If this is not the largest rank name in this original rank,
        #      the value is None
        self.curr_rank_id: Dict[str, Optional[str]] = {}
        for rank in ranks:
            if rank in self.all_parts.keys():
                for i in range(len(self.all_parts[rank])):
                    self.curr_rank_id[rank + str(i)] = None
                self.curr_rank_id[rank +
                                  str(len(self.all_parts[rank]))] = rank
            else:
                self.curr_rank_id[rank] = rank

    def get_all_parts(self) -> Dict[str, List[Tree]]:
        """
        Get the partitioning information for all partitioned ranks
        """
        return self.all_parts

    def get_curr_rank_id(self, rank: str) -> Optional[str]:
        """
        Get the name of this rank in the current loop order
        """
        return self.curr_rank_id[rank]

    def get_dyn_rank(self, rank: str) -> str:
        """
        Convert from a (potentially) static rank name to the corresponding
        dynamic rank name

        Used for the spacetime stamp of dynamically partitioned tensors
        """
        if rank.upper() in self.dyn_parts.keys():
            return rank + "0"
        return rank

    def get_dyn_parts(self) -> Dict[str, List[Tree]]:
        """
        Get the partitioning information for all dynamically partitioned
        ranks
        """
        return self.dyn_parts

    def get_leader(self, part: Tree) -> str:
        """
        Return the leader tensor for this partitioning
        """
        if part.data == "uniform_occupancy":
            return ParseUtils.find_str(part, "leader")

        raise ValueError("Style " + part.data + " has no leader")

    def get_static_parts(self) -> Dict[str, List[Tree]]:
        """
        Get the partitioning information for all statically partitioned
        ranks
        """
        return self.static_parts

    def get_tensor_spec(self, tensor_ranks: Iterable[str],
                        part_ranks: Iterable[str]) -> Dict[str, List[Tree]]:
        """
        Get the partitioning for a specific tensor's ranks
        """
        partitioning = {}
        for rank, part in self.all_parts.items():
            if rank in part_ranks and rank in tensor_ranks:
                partitioning[rank] = part
        return partitioning

    def partition_rank(self, rank: str) -> None:
        """
        Update the partitioning information to include the fact that the given
        rank has been partitioned
        """
        for i in range(len(self.all_parts[rank]) + 1):
            self.curr_rank_id[rank + str(i)] = rank + str(i)

    def __eq__(self, other):
        """
        The == operator for Partitionings
        """

        if isinstance(other, type(self)):
            return self.curr_rank_id == other.curr_rank_id and \
                self.dyn_parts == other.dyn_parts and \
                self.static_parts == other.static_parts

        return False

    @staticmethod
    def __is_static(part: Tree):
        """
        Return true if this style of partitioning can be performed statically
        """
        return part.data == "uniform_shape" or part.data == "nway_shape"
