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

Intermediate representation of display information
"""

from collections import Counter

from lark.tree import Tree
from typing import Dict, List, Optional, Union


class Display:
    """
    An abstract representation of the display(/canvas/graphics) information
    """

    def __init__(self,
                 yaml: Dict[str, Union[str, List[str]]],
                 loop_order: List[str],
                 partitioning: Dict[str, List[Tree]],
                 out_name: str) -> None:
        """
        Build the display object
        """
        # Check the type of the space argument
        space = yaml["space"]
        if not isinstance(space, list):
            raise TypeError(
                "Display space argument must be a list, given " +
                str(space) +
                " on output " +
                out_name)
        else:
            self.space = space
            self.space.sort(key=loop_order.index)

        # Check the type of the time argument
        time = yaml["time"]
        if not isinstance(time, list):
            raise TypeError(
                "Display time argument must be a list, given " +
                str(time) +
                " on output " +
                out_name)
        else:
            self.time = time
            self.time.sort(key=loop_order.index)

        # Now make sure that all indices are scheduled
        if Counter(loop_order) != Counter(self.space + self.time):
            raise ValueError(
                "Incorrect schedule for display on output " +
                out_name)

        # Check the type of the style argument
        style = yaml["style"]
        if not isinstance(style, str):
            raise TypeError(
                "Display style argument must be a string, given " +
                str(style) +
                " on output " +
                out_name)
        else:
            self.style = style

        # Make sure that we are given a correct PE distribution style
        if self.style != "shape" and self.style != "occupancy":
            raise ValueError(
                "Unknown display style " +
                self.style +
                " on output " +
                out_name)

        # Find the base index name associated with the partitioned indices
        self.bases = {}
        for ind in partitioning:
            for i in range(len(partitioning[ind])):
                self.bases[ind + str(i)] = ind + str(i + 1)

    def get_base(self, ind: str) -> Optional[str]:
        """
        Get the base index name associated with a given index

        Used to convert from absolute coordinates to relative coordinates
        """
        if ind in self.bases:
            return self.bases[ind]
        return None

    def get_space(self) -> List[str]:
        """
        Get the space argument of the display
        """
        return self.space

    def get_style(self) -> str:
        """
        Get the style argument of the display
        """
        return self.style

    def get_time(self) -> List[str]:
        """
        Get the time argument of the display
        """
        return self.time

    def __eq__(self, other: object) -> bool:
        """
        The == operator for Displays
        """
        if isinstance(other, type(self)):
            return self.space == other.space and \
                self.style == other.style and \
                self.time == other.time
        return False
