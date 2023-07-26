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

from typing import Any, Dict, Iterable, List, Optional, Set, Union


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
        self.bindings: Any = {}

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

    def _check_bool_attr(self, attrs: dict, key: str) -> Optional[bool]:
        """
        Check that a boolean attribute is correctly specified
        """
        if key not in attrs.keys():
            return None

        if not isinstance(attrs[key], bool):
            class_ = type(self).__name__[:-9]
            raise ValueError("Bad " + key + " " + str(attrs[key]) + " for " + class_ + " " + self.name)

        return attrs[key]

    def _check_int_attr(self, attrs: dict, key: str) -> Optional[int]:
        """
        Check that an integer attribute is correctly specified
        """
        if key not in attrs.keys():
            return None

        if not isinstance(attrs[key], int):
            class_ = type(self).__name__[:-9]
            raise ValueError("Bad " + key + " " + str(attrs[key]) + " for " + class_ + " " + self.name)

        return attrs[key]

    def _check_str_attr(self, attrs: dict, key: str, options: Set[str]) -> Optional[str]:
        """
        Check that a string attribute is correctly specified
        """
        if key not in attrs.keys():
            return None

        class_ = type(self).__name__[:-9]
        if not isinstance(attrs[key], str):
            raise ValueError("Bad " + key + " " + str(attrs[key]) + " for " + class_ + " " + self.name)

        if attrs[key] not in options:
            raise ValueError(attrs[key] + " is not a valid value for attribute " + key + " of class " + class_ + ". Choose one of " + str(options))

        return attrs[key]



class FunctionalComponent(Component):
    """
    Superclass for all functional unit components (compute, intersection, mergers, etc.)
    """

    def __init__(self, name: str, attrs: dict, bindings: List[dict]) -> None:
        """
        Construct a functional component
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

        self.bandwidth = self._check_int_attr(attrs, "bandwidth")

        for binding in bindings:
            self.bindings[binding["tensor"]] = binding["rank"]

    def get_binding(self, tensor: str) -> Optional[str]:
        """
        Given a tensor, give the rank bound to this memory
        """
        if tensor not in self.bindings.keys():
            return None

        return self.bindings[tensor]

    def get_bandwidth(self) -> int:
        """
        Get the bandwidth
        """
        if self.bandwidth is None:
            raise ValueError("Bandwidth unspecified for component " + self.name)

        return self.bandwidth

class BufferComponent(MemoryComponent):
    """
    A Component for a buffer
    """
    def __init__(self, name: str, attrs: dict, bindings: List[dict]) -> None:
        """
        Construct a buffet component
        """
        super().__init__(name, attrs, bindings)

        self.depth = self._check_int_attr(attrs, "depth")
        self.width = self._check_int_attr(attrs, "width")

    def get_depth(self) -> int:
        """
        Get the buffer depth
        """
        if self.depth is None:
            raise ValueError("Depth unspecified for component " + self.name)

        return self.depth

    def get_width(self) -> int:
        """
        Get the buffer width
        """
        if self.width is None:
            raise ValueError("Width unspecified for component " + self.name)

        return self.width

class BuffetComponent(BufferComponent):
    """
    A Component for a Buffet

    TODO: Do we even need a separate class for this?
    """
    pass

class CacheComponent(BufferComponent):
    """
    A Component for a Cache

    TODO: Do we even need a separate class for this?
    """
    pass

class ComputeComponent(FunctionalComponent):
    """
    A Component for a compute functional unit
    """

    def __init__(self, name: str, attrs: dict, bindings: List[dict]) -> None:
        """
        Construct a compute component
        """
        super().__init__(name, attrs, bindings)

        type_ = self._check_str_attr(attrs, "type", {"mul", "add"})
        if type_ is None:
            raise ValueError("Type unspecified for component " + self.name)
        self.type = type_

    def get_type(self) -> str:
        """
        Get the type of compute component
        """
        return self.type

class DRAMComponent(MemoryComponent):
    """
    A Component for DRAM
    """
    pass

class LeaderFollowerComponent(FunctionalComponent):
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

        inputs = self._check_int_attr(attrs, "inputs")
        if inputs is None:
            raise ValueError("Inputs unspecified for component " + self.name)
        self.inputs = inputs

        comparator_radix = self._check_int_attr(attrs, "comparator_radix")
        if comparator_radix is None:
            raise ValueError("Comparator radix unspecified for component " + self.name)
        self.comparator_radix = comparator_radix

        outputs = self._check_int_attr(attrs, "outputs")
        if outputs is None:
            self.outputs = 1
        else:
            self.outputs = outputs

        order = self._check_str_attr(attrs, "order", {"fifo", "opt"})
        if order is None:
            self.order = "fifo"
        else:
            self.order = order

        reduce_ = self._check_bool_attr(attrs, "reduce")
        if reduce_:
            raise NotImplementedError("Concurrent merge and reduction not supported")
        self.reduce = False

        self.bindings = []
        for binding in bindings:
            init = binding["init_ranks"]
            d = binding["swap_depth"]
            final = init[:d] + [init[d + 1]] + [init[d]] + init[(d + 2):]

            info = binding.copy()
            info["final_ranks"] = final

            self.bindings.append(info)

    def get_bindings(self) -> List[dict]:
        """
        Get the operations that are bound to this merger
        """
        return self.bindings

    def get_comparator_radix(self) -> int:
        """
        Get the comparator_radix
        """
        return self.comparator_radix

    def get_inputs(self) -> int:
        """
        Get the number of inputs
        """
        return self.inputs

    def get_order(self) -> str:
        """
        Get the order
        """
        return self.order

    def get_outputs(self) -> int:
        """
        Get the number of outputs
        """
        return self.outputs

    def get_reduce(self) -> bool:
        """
        Get whether or not the merger performs concurrent reduction
        """
        return self.reduce

class SkipAheadComponent(FunctionalComponent):
    """
    A Component for skip-ahead intersection
    """
    pass
