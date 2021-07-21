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
from typing import Dict, List

from es2hfa.ir.tensor import Tensor


class Partitioning:
    """
    An abstract representation of the partitioning information
    """

    def __init__(self, partitioning: Dict[str, List[Tree]]) -> None:
        """
        Create a new representation of the partitioning information
        """
        self.static_parts = {}
        # Filter the partitioning information into the dimensions that can
        # be partitioned statically vs dynamically
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

    def get_static_parts(self) -> Dict[str, List[Tree]]:
        """
        Get the partitioning information for all statically partitioned
        dimensions
        """
        return self.static_parts

    @staticmethod
    def __is_static(part: Tree):
        """
        Return true if this style of partitioning can be performed statically
        """
        return part.data == "uniform_shape" or part.data == "nway_shape"
