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

Representation an hardware component
"""

from typing import Any, Iterable, Union


class Component:
    """
    Representation an hardware component
    """

    def __init__(self, name: str, attrs: dict) -> None:
        """
        Construct a component
        """
        self.name = name
        self.attrs = attrs

    def get_name(self) -> str:
        """
        Get the component name
        """
        return self.name

    def __eq__(self, other: object) -> bool:
        """
        The == operator for components

        """
        if isinstance(other, type(self)):
            return self.__key() == other.__key()
        return False

    def __key(self) -> Iterable[Any]:
        """
        A tuple of all fields of a component
        """
        return (self.name, self.attrs)

    def __repr__(self) -> str:
        """
        A string representation of the component
        """
        strs = [key if isinstance(key, str) else repr(key)
                for key in self.__key()]
        return "(" + type(self).__name__ + ", " + ", ".join(strs) + ")"


class CacheComponent(Component):
    """
    A Component for a Cache
    """

    def get_depth(self) -> int:
        """
        Get the cache depth
        """
        return self.attrs["depth"]

    def get_width(self) -> int:
        """
        Get the cache width
        """
        return self.attrs["width"]


class ComputeComponent(Component):
    """
    A Component for compute
    """
    pass


class DRAMComponent(Component):
    """
    A Component for DRAM
    """

    def get_bandwidth(self) -> int:
        """
        Get the bandwidth
        """
        return self.attrs["bandwidth"]

    def get_datawidth(self) -> int:
        """
        Get the datawidth
        """
        return self.attrs["datawidth"]


class LeaderFollowerComponent(Component):
    """
    A Component for leader-follower intersection
    """
    pass


class MergerComponent(Component):
    """
    A Component for a merger
    """

    def get_next_latency(self) -> Union[int, str]:
        """
        Get the latency of accessing the next element
        """
        return self.attrs["next_latency"]

    def get_radix(self) -> float:
        """
        Get the radix
        """
        if self.attrs["radix"] == "inf":
            return float("inf")

        return self.attrs["radix"]


class SkipAheadComponent(Component):
    """
    A Component for skip-ahead intersection
    """
    pass


class SRAMComponent(Component):
    """
    A Component for SRAM
    """
    pass
