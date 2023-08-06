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

from teaal.ir.partitioning import Partitioning
from teaal.parse.utils import ParseUtils


class SpaceTime:
    """
    An abstract representation of the display(/canvas/graphics) information
    """

    def __init__(self,
                 yaml: dict,
                 partitioning: Partitioning,
                 out_name: str) -> None:
        """
        Build the spacetime object
        """
        self.partitioning = partitioning
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
        # of ranks
        self.space = []
        for tree in space:
            rank = ParseUtils.next_str(tree)
            self.space.append(rank)
            self.styles[rank] = tree.data

        # Check the type of the time argument
        time = yaml["time"]
        if not isinstance(time, list):
            raise TypeError(
                "SpaceTime time argument must be a list, given " +
                str(time) +
                " on output " +
                out_name)

        # Otherwise, collect the display style information and build the list
        # of ranks
        self.time = []
        for tree in time:
            rank = ParseUtils.next_str(tree)
            self.time.append(rank)
            self.styles[rank] = tree.data

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

    def emit_pos(self, rank: str) -> bool:
        """
        Return true if the rank_pos variable needs to be emitted
        """
        if self.get_style(rank) == "coord":
            return False

        return not self.get_slip() or rank in self.space

    def get_offset(self, rank: str) -> Optional[str]:
        """
        Get the offset rank id associated with a given rank

        Used to convert from absolute coordinates to relative coordinates
        """
        return self.partitioning.get_offset(rank)

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

    def get_style(self, rank: str) -> str:
        """
        Get the style of display for the given rank
        """
        final = self.partitioning.get_final_rank_id([rank], rank)
        return self.styles[final]

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
