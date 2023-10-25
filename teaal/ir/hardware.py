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

Representation of the hardware of an accelerator
"""

from typing import Dict, Set, Type, TypeVar

from teaal.ir.component import *
from teaal.ir.level import Level
from teaal.ir.program import Program

from teaal.parse import *

T = TypeVar("T")


class Hardware:
    """
    Representation of the hardware of an accelerator
    """

    def __init__(
            self,
            arch: Architecture,
            bindings: Bindings,
            program: Program) -> None:
        """
        Construct the hardware

        TODO: The program is only used to get the Einsum name; standardize
        so all use program or all take it as an argument
        """
        self.bindings = bindings
        self.program = program

        self.components: Dict[str, Component] = {}

        # Get the configuration for each Einsum
        self.configs = {}
        for einsum in self.program.get_all_einsums():
            self.configs[einsum] = self.bindings.get_config(einsum)

        spec = arch.get_spec()
        if spec is None:
            raise ValueError("Empty architecture specification")

        # Build the architecture tree for each configuration
        self.tree = {}
        for config in spec["architecture"]:
            subtree = spec["architecture"][config]
            if len(subtree) != 1:
                raise ValueError(
                    "Configuration " +
                    config +
                    " must have a single root level")

            self.tree[config] = self.__build_level(subtree[0])

    def get_component(self, name: str) -> Component:
        """
        Get component by its name
        """
        return self.components[name]

    def get_components(self, einsum: str, class_: Type[T]) -> List[T]:
        """
        Get a list of components relevant to this einsum
        """
        components: List[T] = []
        for name in self.bindings.get_bindings()[einsum]:
            component = self.components[name]
            if isinstance(component, class_):
                components.append(component)
        return components

    def get_config(self, einsum: str) -> str:
        """
        Get the name of the hardware configuration for this Einsum
        """
        return self.configs[einsum]

    def get_frequency(self, einsum: str) -> int:
        """
        The clock_frequency (in Hz) should be specified as an attribute at the
        top level
        """
        top_level = self.tree[self.configs[einsum]]
        freq = top_level.get_attr("clock_frequency")

        if freq is None:
            raise ValueError("Unspecified clock frequency for config " + self.configs[einsum])

        if isinstance(freq, str):
            raise ValueError("Bad clock frequency for config " + self.configs[einsum])

        return freq

    def get_prefix(self, einsum: str) -> str:
        """
        Get the prefix for collected metrics for the given Einsum
        """
        return self.bindings.get_prefix(einsum)

    def get_traffic_path(
            self,
            tensor: str,
            rank: str,
            type_: str,
            format_: str) -> List[Tuple[MemoryComponent, str]]:
        """
        Get a list of components  this tensor will be loaded into and either
        a lazy style or the source rank of the eager load
        """
        einsum = self.program.get_equation().get_output().root_name()

        components: List[Tuple[MemoryComponent, str]] = []

        levels = [(self.tree[self.configs[einsum]], 0)]
        depths_covered = set()
        while levels:
            level, depth = levels.pop()

            for component in level.get_local():
                if not isinstance(component, MemoryComponent):
                    continue

                binding = component.get_binding(
                    einsum, tensor, rank, type_, format_)
                if binding:
                    if isinstance(
                            component,
                            BuffetComponent) and binding["style"] == "eager":
                        components.append((component, binding["root"]))
                    else:
                        components.append((component, "lazy"))

                    if depth in depths_covered:
                        raise ValueError(
                            "Multiple traffic paths for tensor " +
                            tensor +
                            " in Einsum " +
                            einsum)
                    depths_covered.add(depth)

            levels.extend((tree, depth + 1) for tree in level.get_subtrees())

        return components

    def get_tree(self) -> Level:
        """
        Get the architecture tree
        """
        einsum = self.program.get_equation().get_output().root_name()
        return self.tree[self.configs[einsum]]

    def __build_component(self, local: dict, num_instances: int) -> Component:
        """
        Build a component
        """
        class_: Type[Component]
        class_name = local["class"].lower()
        if class_name == "buffet":
            class_ = BuffetComponent

        elif class_name == "cache":
            class_ = CacheComponent

        elif class_name == "compute":
            class_ = ComputeComponent

        elif class_name == "dram":
            class_ = DRAMComponent

        elif class_name == "intersector":
            type_ = local["attributes"]["type"].lower()
            if type_ == "leader-follower":
                class_ = LeaderFollowerComponent

            elif type_ == "skip-ahead":
                class_ = SkipAheadComponent

            elif type_ == "two-finger":
                class_ = TwoFingerComponent

            else:
                raise ValueError("Unknown intersection type: " + type_)

        elif class_name == "merger":
            class_ = MergerComponent

        elif class_name == "sequencer":
            class_ = SequencerComponent

        else:
            raise ValueError("Unknown class: " + local["class"])

        name = local["name"]
        binding = self.bindings.get_component(name)

        component = class_(name, num_instances, local["attributes"], binding)
        self.components[component.get_name()] = component

        return component

    def __build_level(self, tree: dict) -> Level:
        """
        Build the levels of the architecture tree
        """
        attrs = tree["attributes"]
        local = [self.__build_component(comp, tree["num"])
                 for comp in tree["local"]]
        subtrees = [self.__build_level(subtree)
                    for subtree in tree["subtree"]]

        return Level(tree["name"], tree["num"], attrs, local, subtrees)
