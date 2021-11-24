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
from typing import Dict, List, Optional, Set

from es2hfa.ir.tensor import Tensor
from es2hfa.parse.utils import ParseUtils


class Partitioning:
    """
    An abstract representation of the partitioning information
    """

    def __init__(self,
                 partitioning: Dict[str,
                                    List[Tree]],
                 inds: List[str]) -> None:
        """
        Create a new representation of the partitioning information
        """
        # Filter the partitioning information into the dimensions that can
        # be partitioned statically vs dynamically
        self.dyn_parts = {}
        self.static_parts = {}

        for ind, parts in partitioning.items():

            # Continue if this dimension is not actually partitioned
            if not parts:
                continue

            # Make sure that the dimension is either partitioned completely
            # statically or completely dynamically
            static = Partitioning.__is_static(parts[0])
            for part in parts[1:]:
                if Partitioning.__is_static(part) != static:
                    raise ValueError(
                        "Dimension " +
                        ind +
                        " cannot be partitioned both statically and dynamically")

            # Add the partitioning specification to the appropriate dictionary
            if static:
                self.static_parts[ind] = parts
            else:
                self.dyn_parts[ind] = parts

        self.all_parts = {**self.static_parts, **self.dyn_parts}

        # Build a dictionary from final index name to an optional index name
        # where the value is determined by:
        # - If the dimension is unpartitioned, the value is the same as the key
        # - If the dimension has already been partitioned, value == key
        # - If the dimension will be partitioned
        #    - If this is the largest index name in this original dimension, the
        #      the value is the initial dimension name
        #    - If this is not the largest index name in this original dimension,
        #      the value is None
        self.curr_ind_name: Dict[str, Optional[str]] = {}
        for ind in inds:
            if ind in self.all_parts.keys():
                for i in range(len(self.all_parts[ind])):
                    self.curr_ind_name[ind + str(i)] = None
                self.curr_ind_name[ind + str(len(self.all_parts[ind]))] = ind
            else:
                self.curr_ind_name[ind] = ind

    def get_all_parts(self) -> Dict[str, List[Tree]]:
        """
        Get the partitioning information for all partitioned dimensions
        """
        return self.all_parts

    def get_curr_ind_name(self, ind: str) -> Optional[str]:
        """
        Get the index name of this index in the current loop order
        """
        return self.curr_ind_name[ind]

    def get_dyn_ind(self, ind: str) -> str:
        """
        Convert from a (potentially) static index name to the corresponding
        dynamic index name
        """
        if ind[0].upper() + ind[1:] in self.dyn_parts.keys():
            return ind + "0"
        return ind

    def get_dyn_parts(self) -> Dict[str, List[Tree]]:
        """
        Get the partitioning information for all dynamically partitioned
        dimensions
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
        dimensions
        """
        return self.static_parts

    def get_tensor_spec(self, tensor: Tensor,
                        inds: Set[str]) -> Dict[str, List[Tree]]:
        """
        Get the partitioning for a specific tensor
        """
        partitioning = {}
        for ind, part in self.all_parts.items():
            if ind in inds and ind in tensor.get_inds():
                partitioning[ind] = part
        return partitioning

    def partition_dim(self, ind: str) -> None:
        """
        Update the partitioning information to include the fact that the given
        dimension has been partitioned
        """
        for i in range(len(self.all_parts[ind]) + 1):
            self.curr_ind_name[ind + str(i)] = ind + str(i)

    def __eq__(self, other):
        """
        The == operator for Partitionings
        """

        if isinstance(other, type(self)):
            return self.curr_ind_name == other.curr_ind_name and \
                self.dyn_parts == other.dyn_parts and \
                self.static_parts == other.static_parts

        return False

    @staticmethod
    def __is_static(part: Tree):
        """
        Return true if this style of partitioning can be performed statically
        """
        return part.data == "uniform_shape" or part.data == "nway_shape"
