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

from typing import Any, Dict, Iterable, List, Optional, Union


class Component:
    """
    Representation an hardware component
    """

    def __init__(self, name: str, attrs: dict, bindings: List[dict]) -> None:
        """
        Construct a component
        """
        self.name = name
        self.attrs = attrs
        self.bindings: dict = {}

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
        return (self.name, self.attrs, self.bindings)

    def __repr__(self) -> str:
        """
        A string representation of the component
        """
        strs = [key if isinstance(key, str) else repr(key)
                for key in self.__key()]
        return "(" + type(self).__name__ + ", " + ", ".join(strs) + ")"


class ComputeComponent(Component):
    """
    A Component for compute (acting also as a superclass for all compute
    operations)
    """

    def __init__(self, name: str, attrs: dict, bindings: List[dict]) -> None:
        """
        Construct a compute component
        """
        super().__init__(name, attrs, bindings)
        self.bindings = {}

        for binding in bindings:
            einsum = binding["einsum"]
            if einsum not in self.bindings.keys():
                self.bindings[einsum] = []

            # Append the dictionary containing the other properties
            info = binding.copy()
            del info["einsum"]
            self.bindings[einsum].append(info)

    def get_bindings(self, einsum: str) -> List[dict]:
        """
        Get the operations that are bound for this einsum
        """
        if einsum not in self.bindings.keys():
            return []

        return self.bindings[einsum]


class MemoryComponent(Component):
    """
    Superclass for all memory components
    """

    def __init__(self, name: str, attrs: dict, bindings: List[dict]) -> None:
        """
        Construct a memory component
        """
        super().__init__(name, attrs, bindings)
        self.bindings = {}

        for binding in bindings:
            self.bindings[binding["tensor"]] = binding["rank"]

    def get_binding(self, tensor: str) -> Optional[str]:
        """
        Given a tensor, give the rank bound to this memory
        """
        if tensor not in self.bindings.keys():
            return None

        return self.bindings[tensor]


class CacheComponent(MemoryComponent):
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


class DRAMComponent(MemoryComponent):
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


class LeaderFollowerComponent(ComputeComponent):
    """
    A Component for leader-follower intersection
    """
    pass


class MergerComponent(Component):
    """
    A Component for a merger
    """

    def __init__(self, name: str, attrs: dict, bindings: List[dict]) -> None:
        """
        Construct a compute component
        """
        super().__init__(name, attrs, bindings)
        self.bindings = {}

        for binding in bindings:
            tensor = binding["tensor"]
            if tensor not in self.bindings.keys():
                self.bindings[tensor] = []

            # Append the dictionary containing the other properties
            info = binding.copy()
            del info["tensor"]
            self.bindings[tensor].append(info)

    def get_bindings(self, tensor: str) -> List[str]:
        """
        Get the operations that are bound for this tensor
        """
        if tensor not in self.bindings.keys():
            return []

        return self.bindings[tensor]

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


class SkipAheadComponent(ComputeComponent):
    """
    A Component for skip-ahead intersection
    """
    pass


class SRAMComponent(MemoryComponent):
    """
    A Component for SRAM
    """
    pass
