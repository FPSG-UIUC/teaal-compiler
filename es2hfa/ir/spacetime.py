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

Intermediate representation of the display information
"""

from typing import List, Optional

from es2hfa.ir.partitioning import Partitioning
from es2hfa.parse.utils import ParseUtils


class SpaceTime:
    """
    An abstract representation of the display(/canvas/graphics) information
    """

    def __init__(self,
                 yaml: dict,
                 loop_order: List[str],
                 partitioning: Partitioning,
                 out_name: str) -> None:
        """
        Build the spacetime object
        """
        self.styles = {}

        # Check the type of the space argument
        space = yaml["space"]
        if not isinstance(space, list):
            raise TypeError(
                "SpaceTime space argument must be a list, given " +
                str(space) +
                " on output " +
                out_name)

        # Otherwise, collect the display style information and build the list
        # of indices
        self.space = []
        for tree in space:
            ind = ParseUtils.next_str(tree)
            self.space.append(ind)
            self.styles[ind] = tree.data

        # Check the type of the time argument
        time = yaml["time"]
        if not isinstance(time, list):
            raise TypeError(
                "SpaceTime time argument must be a list, given " +
                str(time) +
                " on output " +
                out_name)

        # Otherwise, collect the display style information and build the list
        # of indices
        self.time = []
        for tree in time:
            ind = ParseUtils.next_str(tree)
            self.time.append(ind)
            self.styles[ind] = tree.data

        # Find the offset index name associated with the partitioned indices
        self.offsets = {}
        for ind, parts in partitioning.get_all_parts().items():
            for i in range(len(parts)):
                self.offsets[ind + str(i)] = ind + str(i + 1)

        # Store slip if it is specified
        self.slip = False
        if "opt" in yaml.keys():
            if yaml["opt"] == "slip":
                self.slip = True
            elif yaml["opt"] is not None:
                raise ValueError(
                    "Unknown spacetime optimization " +
                    yaml["opt"] +
                    " on output " +
                    out_name)

    def emit_pos(self, ind: str) -> bool:
        """
        Return true if the ind_pos variable needs to be emitted
        """
        if self.get_style(ind) == "coord":
            return False

        return not self.get_slip() or ind in self.space

    def get_offset(self, ind: str) -> Optional[str]:
        """
        Get the offset index name associated with a given index

        Used to convert from absolute coordinates to relative coordinates
        """
        if ind in self.offsets:
            return self.offsets[ind]
        return None

    def get_slip(self) -> bool:
        """
        Returns true if slip should be implemented
        """
        return self.slip

    def get_space(self) -> List[str]:
        """
        Get the space argument of the display
        """
        return self.space

    def get_style(self, ind: str) -> str:
        """
        Get the style of display for the given index
        """
        return self.styles[ind]

    def get_time(self) -> List[str]:
        """
        Get the time argument of the display
        """
        return self.time

    def __eq__(self, other: object) -> bool:
        """
        The == operator for SpaceTimes
        """
        if isinstance(other, type(self)):
            return self.space == other.space and \
                self.styles == other.styles and \
                self.time == other.time
        return False
