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

from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

S = TypeVar("S")


class Component:
    """
    Representation an hardware component
    """

    def __init__(self, name: str, attrs: dict,
                 bindings: Dict[str, List[dict]]) -> None:
        """
        Construct a component
        """
        self.name = name
        # TODO: Remove this (not used)
        self.attrs = attrs
        self.bindings = bindings

    def get_name(self) -> str:
        """
        Get the component name
        """
        return self.name

    def get_bindings(self) -> Dict[str, List[dict]]:
        """
        Get the operations that are bound to this component
        """
        return self.bindings

    def __eq__(self, other: object) -> bool:
        """
        The == operator for components

        """
        if isinstance(other, type(self)):
            return self.__key() == other.__key()
        return False

    def __hash__(self) -> int:
        """
        Hash the component
        """
        return hash(repr(self))

    def __key(self) -> Tuple[Any, ...]:
        """
        A tuple of all fields of a component
        """
        return (self.name, self.bindings)

    def __repr__(self) -> str:
        """
        A string representation of the component
        """
        strs = [key if isinstance(key, str) else repr(key)
                for key in self.__key()]
        return "(" + type(self).__name__ + ", " + ", ".join(strs) + ")"

    def _check_attr(
            self,
            attrs: dict,
            key: str,
            type_: Type[S]) -> Optional[S]:
        """
        Check that the attribute is correctly specified
        """
        if key not in attrs.keys():
            return None

        if not isinstance(attrs[key], type_):
            class_ = type(self).__name__[:-9]
            raise ValueError("Bad " +
                             key +
                             " " +
                             str(attrs[key]) +
                             " for " +
                             class_ +
                             " " +
                             self.name)

        return attrs[key]

    def _check_float_attr(self, attrs: dict, key: str) -> Optional[float]:
        """
        Check that the attribute is correctly specified
        """
        if key not in attrs.keys():
            return None

        if attrs[key] == "inf":
            return float("inf")

        if not isinstance(
                attrs[key],
                float) and not isinstance(
                attrs[key],
                int):
            class_ = type(self).__name__[:-9]
            raise ValueError("Bad " +
                             key +
                             " " +
                             str(attrs[key]) +
                             " for " +
                             class_ +
                             " " +
                             self.name)

        return attrs[key]

    def _check_str_attr(
            self,
            attrs: dict,
            key: str,
            options: Set[str]) -> Optional[str]:
        """
        Check that a string attribute is correctly specified
        """
        if key not in attrs.keys():
            return None

        class_ = type(self).__name__[:-9]
        if not isinstance(attrs[key], str):
            raise ValueError("Bad " +
                             key +
                             " " +
                             str(attrs[key]) +
                             " for " +
                             class_ +
                             " " +
                             self.name)

        if attrs[key] not in options:
            raise ValueError(
                attrs[key] +
                " is not a valid value for attribute " +
                key +
                " of class " +
                class_ +
                ". Choose one of " +
                str(options))

        return attrs[key]


class FunctionalComponent(Component):
    """
    Superclass for all functional unit components (compute, intersection, mergers, etc.)
    """

    def __init__(self, name: str, attrs: dict,
                 bindings: Dict[str, List[dict]]) -> None:
        """
        Construct a functional component
        """
        super().__init__(name, attrs, bindings)


class MemoryComponent(Component):
    """
    Superclass for all memory components
    """

    def __init__(self, name: str, attrs: dict,
                 bindings: Dict[str, List[dict]]) -> None:
        """
        Construct a memory component
        """
        super().__init__(name, attrs, bindings)

        self.bandwidth = self._check_attr(attrs, "bandwidth", int)

        self.tensor_bindings: Dict[str, Dict[str, List[dict]]] = {}
        for einsum in self.bindings.keys():
            self.tensor_bindings[einsum] = {}
            for binding in self.bindings[einsum]:
                if "tensor" not in binding:
                    raise ValueError(
                        "Tensor not specified for Einsum " +
                        einsum +
                        " in binding to " +
                        self.name)

                tensor = binding["tensor"]
                if "rank" not in binding:
                    raise ValueError(
                        "Rank not specified for tensor " +
                        tensor +
                        " in Einsum " +
                        einsum +
                        " in binding to " +
                        self.name)

                if "type" not in binding:
                    raise ValueError(
                        "Type not specified for tensor " +
                        tensor +
                        " in Einsum " +
                        einsum +
                        " in binding to " +
                        self.name)

                types = {"coord", "payload", "elem"}
                if binding["type"] not in types:
                    raise ValueError("Type " +
                                     str(binding["type"]) +
                                     " for " +
                                     self.name +
                                     " on tensor " +
                                     tensor +
                                     " in Einsum " +
                                     einsum +
                                     " not one of " +
                                     str(types))

                if "format" not in binding:
                    raise ValueError(
                        "Format not specified for tensor " +
                        tensor +
                        " in Einsum " +
                        einsum +
                        " in binding to " +
                        self.name)

                if binding["tensor"] not in self.tensor_bindings[einsum]:
                    self.tensor_bindings[einsum][binding["tensor"]] = []
                self.tensor_bindings[einsum][binding["tensor"]].append(binding)

    def get_bandwidth(self) -> int:
        """
        Get the bandwidth
        """
        if self.bandwidth is None:
            raise ValueError(
                "Bandwidth unspecified for component " +
                self.name)

        return self.bandwidth

    def get_binding(self, einsum: str, tensor: str, rank: str,
                    type_: str, format_: str) -> Optional[Dict[str, Any]]:
        """
        Given a tensor, get a list of bindings to that rank
        """
        if einsum not in self.tensor_bindings:
            return None

        if tensor not in self.tensor_bindings[einsum]:
            return None

        final_binding: Optional[Dict[str, Any]] = None
        for binding in self.tensor_bindings[einsum][tensor]:
            if binding["rank"] == rank and binding["type"] == type_ and binding["format"] == format_:

                if final_binding is None:
                    final_binding = binding

                else:
                    raise ValueError("Multiple bindings for " + str(
                        [("einsum", einsum), ("tensor", tensor), ("rank", rank), ("type", type_), ("format", format_)]))

        return final_binding

    def _Component__key(self) -> Tuple[Any, ...]:
        """
        A tuple of all fields
        """
        return (self.name, self.bindings, self.bandwidth)


class BufferComponent(MemoryComponent):
    """
    A Component for a buffer
    """

    def __init__(self, name: str, attrs: dict,
                 bindings: Dict[str, List[dict]]) -> None:
        """
        Construct a buffer component
        """
        super().__init__(name, attrs, bindings)

        self.depth = self._check_float_attr(attrs, "depth")
        self.width = self._check_attr(attrs, "width", int)

    def get_depth(self) -> float:
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

    def _Component__key(self) -> Tuple[Any, ...]:
        """
        A tuple of all fields
        """
        return (
            self.name,
            self.bindings,
            self.bandwidth,
            self.depth,
            self.width)


class BuffetComponent(BufferComponent):
    """
    A Component for a Buffet
    """

    def __init__(self, name: str, attrs: dict,
                 bindings: Dict[str, List[dict]]) -> None:
        """
        Construct a buffet component
        """
        super().__init__(name, attrs, bindings)
        for einsum in self.tensor_bindings:
            for tensor, tensor_bindings in self.tensor_bindings[einsum].items(
            ):
                for binding in tensor_bindings:
                    if "evict-on" not in binding:
                        raise ValueError(
                            "Evict-on not specified for tensor " +
                            tensor +
                            " in Einsum " +
                            einsum +
                            " in binding to " +
                            self.name)

                    if "style" not in binding:
                        binding["style"] = "lazy"

                    styles = {"lazy", "eager"}
                    if binding["style"] not in styles:
                        raise ValueError("Style " +
                                         str(binding["style"]) +
                                         " for " +
                                         self.name +
                                         " on tensor " +
                                         tensor +
                                         " in Einsum " +
                                         einsum +
                                         " not one of " +
                                         str(styles))


class CacheComponent(BufferComponent):
    """
    A Component for a Cache
    """
    pass


class ComputeComponent(FunctionalComponent):
    """
    A Component for a compute functional unit
    """

    def __init__(self, name: str, attrs: dict,
                 bindings: Dict[str, List[dict]]) -> None:
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

    def _Component__key(self) -> Tuple[Any, ...]:
        """
        A tuple of all fields
        """
        return (self.name, self.bindings, self.type)


class DRAMComponent(MemoryComponent):
    """
    A Component for DRAM
    """
    pass


class IntersectorComponent(FunctionalComponent):
    """
    A Component superclass for all intersectors
    """

    def __init__(self, name: str, attrs: dict,
                 bindings: Dict[str, List[dict]]) -> None:
        """
        Construct an intersector component
        """
        super().__init__(name, attrs, bindings)

        for einsum, einsum_bindings in bindings.items():
            for binding in einsum_bindings:
                if "rank" not in binding:
                    raise ValueError(
                        "Rank unspecified in Einsum " +
                        einsum +
                        " in binding to " +
                        self.name)


class LeaderFollowerComponent(IntersectorComponent):
    """
    A Component for leader-follower intersection
    """

    def __init__(self, name: str, attrs: dict,
                 bindings: Dict[str, List[dict]]) -> None:
        """
        Construct a leader-follower intersector component
        """
        super().__init__(name, attrs, bindings)

        for einsum, einsum_bindings in bindings.items():
            for binding in einsum_bindings:
                if "leader" not in binding:
                    raise ValueError(
                        "Leader unspecified in Einsum " +
                        einsum +
                        " in binding to " +
                        self.name)


class MergerComponent(Component):
    """
    A Component for a merger
    """

    def __init__(self, name: str, attrs: dict,
                 bindings: Dict[str, List[dict]]) -> None:
        """
        Construct a merger component
        """
        super().__init__(name, attrs, bindings)

        # TODO: change back to int
        inputs = self._check_float_attr(attrs, "inputs")
        if inputs is None:
            raise ValueError("Inputs unspecified for component " + self.name)
        self.inputs = inputs

        # TODO: change back to int
        comparator_radix = self._check_float_attr(attrs, "comparator_radix")
        if comparator_radix is None:
            raise ValueError(
                "Comparator radix unspecified for component " +
                self.name)
        self.comparator_radix = comparator_radix

        outputs = self._check_attr(attrs, "outputs", int)
        if outputs is None:
            self.outputs = 1
        else:
            self.outputs = outputs

        order = self._check_str_attr(attrs, "order", {"fifo", "opt"})
        if order is None:
            self.order = "fifo"
        else:
            self.order = order

        reduce_ = self._check_attr(attrs, "reduce", bool)
        if reduce_:
            raise NotImplementedError(
                "Concurrent merge and reduction not supported")
        self.reduce = False

        self.tensor_bindings: Dict[str, Dict[str, List[dict]]] = {}
        for einsum, einsum_bindings in self.bindings.items():
            self.tensor_bindings[einsum] = {}
            for binding in einsum_bindings:
                if "tensor" not in binding:
                    raise ValueError(
                        "Tensor not specified for Einsum " +
                        einsum +
                        " in binding to " +
                        self.name)

                tensor = binding["tensor"]
                if tensor not in self.tensor_bindings[einsum]:
                    self.tensor_bindings[einsum][tensor] = []

                if "init-ranks" not in binding:
                    raise ValueError(
                        "Initial ranks not specified for tensor " +
                        tensor +
                        " in Einsum " +
                        einsum +
                        " in binding to " +
                        self.name)

                if "final-ranks" not in binding:
                    raise ValueError(
                        "Final ranks not specified for tensor " +
                        tensor +
                        " in Einsum " +
                        einsum +
                        " in binding to " +
                        self.name)

                self.tensor_bindings[einsum][tensor].append(binding)

    def get_comparator_radix(self) -> float:
        """
        Get the comparator_radix
        """
        return self.comparator_radix

    def get_init_ranks(self, einsum: str, tensor: str,
                       final_ranks: List[str]) -> Optional[List[str]]:
        """
        Get the initial ranks for the given merge
        """
        if einsum not in self.tensor_bindings:
            return None

        if tensor not in self.tensor_bindings[einsum]:
            return None

        init_ranks: Optional[List[str]] = None
        for binding in self.tensor_bindings[einsum][tensor]:
            if binding["final-ranks"] == final_ranks:
                if init_ranks is not None:
                    raise ValueError("Merge binding from both " +
                                     str(init_ranks) +
                                     " and " +
                                     str(binding["init-ranks"]) +
                                     " to " +
                                     str(final_ranks))

                init_ranks = binding["init-ranks"]

        return init_ranks

    def get_inputs(self) -> float:
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

    def _Component__key(self) -> Tuple[Any, ...]:
        """
        A tuple of all fields
        """
        return (
            self.name,
            self.bindings,
            self.inputs,
            self.comparator_radix,
            self.outputs,
            self.order,
            self.reduce)


class SkipAheadComponent(IntersectorComponent):
    """
    A Component for skip-ahead intersection
    """
    pass


class TwoFingerComponent(IntersectorComponent):
    """
    A Component for two-finger intersection
    """
    pass
