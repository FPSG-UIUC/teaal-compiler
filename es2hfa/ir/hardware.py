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

from es2hfa.ir.component import *
from es2hfa.ir.level import Level


class Hardware:
    """
    Representation of the hardware of an accelerator
    """

    def __init__(self, arch: dict, bindings: dict) -> None:
        """
        Construct the hardware
        """
        self.components: Dict[str, Component] = {}

        subtree = arch["architecture"]["subtree"]
        if len(subtree) != 1:
            raise ValueError("Architecture must have a single root level")

        self.tree = self.__build_level(subtree[0], bindings)

    def get_component(self, name: str) -> Component:
        """
        Get component by its name
        """
        return self.components[name]

    def get_tree(self) -> Level:
        """
        Get the architecture tree
        """
        return self.tree

    def __build_component(self, local: dict, bindings: dict) -> Component:
        """
        Build a component
        """
        class_: Type[Component]
        if local["class"].lower() == "cache":
            class_ = CacheComponent

        elif local["class"].lower() == "compute":
            class_ = ComputeComponent

        elif local["class"].lower() == "dram":
            class_ = DRAMComponent

        elif local["class"].lower() == "leaderfollower":
            class_ = LeaderFollowerComponent

        elif local["class"].lower() == "merger":
            class_ = MergerComponent

        elif local["class"].lower() == "skipahead":
            class_ = SkipAheadComponent

        elif local["class"].lower() == "sram":
            class_ = SRAMComponent

        else:
            raise ValueError("Unknown class: " + local["class"])

        name = local["name"]
        if name in bindings.keys():
            binding = bindings[name]
        else:
            binding = []

        component = class_(name, local["attributes"], binding)
        self.components[component.get_name()] = component

        return component

    def __build_level(self, arch: dict, bindings: dict) -> Level:
        """
        Build the levels of the architecture tree
        """
        attrs = arch["attributes"]
        local = [self.__build_component(comp, bindings)
                 for comp in arch["local"]]
        subtrees = [self.__build_level(tree, bindings)
                    for tree in arch["subtree"]]

        return Level(arch["name"], arch["num"], attrs, local, subtrees)
