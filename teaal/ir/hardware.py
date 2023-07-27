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

from typing import Dict, Type

from teaal.ir.component import *
from teaal.ir.level import Level
from teaal.parse import *


class Hardware:
    """
    Representation of the hardware of an accelerator
    """

    def __init__(self, arch: Architecture, bindings: Bindings) -> None:
        """
        Construct the hardware
        """
        self.components: Dict[str, Component] = {}

        spec = arch.get_spec()
        if spec is None:
            raise ValueError("Empty architecture specification")

        subtree = spec["architecture"]["subtree"]
        if len(subtree) != 1:
            raise ValueError("Architecture must have a single root level")

        self.tree = self.__build_level(subtree[0], bindings)

    def get_component(self, name: str) -> Component:
        """
        Get component by its name
        """
        return self.components[name]

    def get_compute_path(self, einsum: str) -> List[Level]:
        """
        Get a list of levels with dataflow corresponding to this einsum
        """
        return self.__compute_helper(einsum, self.tree)

    def get_functional_components(
            self, einsum: str) -> List[FunctionalComponent]:
        """
        Get a list of compute components relevant to this einsum
        """
        path = self.get_compute_path(einsum)

        components = []
        for level in path:
            for component in level.get_local():
                if isinstance(component, FunctionalComponent) and \
                        component.get_bindings()[einsum]:
                    components.append(component)

        return components

    def get_merger_components(self) -> List[MergerComponent]:
        """
        Get all merger components
        """
        mergers = []
        for component in self.components.values():
            if isinstance(component, MergerComponent):
                mergers.append(component)

        return mergers

    def get_traffic_path(
            self,
            einsum: str,
            tensor: str) -> List[MemoryComponent]:
        """
        Get a list of paths this tensor will be loaded into
        """
        paths = self.__traffic_helper(einsum, tensor, self.tree)

        # Merge all paths together
        final: List[MemoryComponent] = []
        compute_path = self.get_compute_path(einsum)

        for path in paths:
            sub_path = Hardware.__sub_path(path, compute_path)

            if len(final) < len(sub_path) and final == sub_path[:len(final)]:
                final = sub_path

            elif len(sub_path) < len(final) and sub_path == final[:len(sub_path)]:
                pass

            elif sub_path == final:
                pass

            else:
                raise ValueError(
                    "Multiple bindings for einsum " +
                    einsum +
                    " and tensor " +
                    tensor)

        return final

    def get_tree(self) -> Level:
        """
        Get the architecture tree
        """
        return self.tree

    @staticmethod
    def __sub_path(
            mem_path: List[MemoryComponent],
            compute_path: List[Level]) -> List[MemoryComponent]:
        """
        Return the prefix of the mem_path captured by this compute_path
        """
        i = 0
        for level in compute_path:
            if mem_path[i] in level.get_local():
                i += 1

            if i == len(mem_path):
                break

        return mem_path[:i]

    def __build_component(self, local: dict, bindings: Bindings) -> Component:
        """
        Build a component
        """
        class_: Type[Component]
        if local["class"].lower() == "buffet":
            class_ = BuffetComponent

        elif local["class"].lower() == "cache":
            class_ = CacheComponent

        elif local["class"].lower() == "compute":
            class_ = FunctionalComponent

        elif local["class"].lower() == "dram":
            class_ = DRAMComponent

        elif local["class"].lower() == "leaderfollower":
            class_ = LeaderFollowerComponent

        elif local["class"].lower() == "merger":
            class_ = MergerComponent

        elif local["class"].lower() == "skipahead":
            class_ = SkipAheadComponent

        else:
            raise ValueError("Unknown class: " + local["class"])

        name = local["name"]
        binding = bindings.get(name)

        component = class_(name, local["attributes"], binding)
        self.components[component.get_name()] = component

        return component

    def __build_level(self, tree: dict, bindings: Bindings) -> Level:
        """
        Build the levels of the architecture tree
        """
        attrs = tree["attributes"]
        local = [self.__build_component(comp, bindings)
                 for comp in tree["local"]]
        subtrees = [self.__build_level(subtree, bindings)
                    for subtree in tree["subtree"]]

        return Level(tree["name"], tree["num"], attrs, local, subtrees)

    def __compute_helper(self, einsum: str, level: Level) -> List[Level]:
        """
        Recursive implementation to find the dataflow to compute for a given
        einsum
        """
        # Recurse down the tree
        paths = []
        for subtree in level.get_subtrees():
            sub_path = self.__compute_helper(einsum, subtree)
            if sub_path:
                paths.append(sub_path)

        if len(paths) > 1:
            raise ValueError("Only one compute path allowed per einsum")

        if paths:
            return [level] + paths[0]

        # Check if a local component performs compute for this einsum
        root = False
        for comp in level.get_local():
            if isinstance(comp, FunctionalComponent) and \
                    comp.get_bindings()[einsum]:
                return [level]

        return []

    def __traffic_helper(self, einsum: str, tensor: str,
                         level: Level) -> List[List[MemoryComponent]]:
        """
        Recursive implementation to find the memory traffic pattern of a tensor
        from a given subtree
        """
        # Recurse down the tree
        paths = []
        for subtree in level.get_subtrees():
            paths.extend(self.__traffic_helper(einsum, tensor, subtree))

        # Check if the memory components at this level store the tensor
        mem_components = []
        for comp in level.get_local():
            if isinstance(
                    comp,
                    MemoryComponent) and comp.get_binding(
                    einsum,
                    tensor):
                mem_components.append(comp)

        # Return a list of paths
        if not paths:
            return [[mem] for mem in mem_components]

        if not mem_components:
            return paths

        final = []
        for mem in mem_components:
            for path in paths:
                final.append([mem] + path)

        return final
