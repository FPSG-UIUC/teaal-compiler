"""
MIT License

Copyright (c) 2023 University of Illinois

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

Representation of the fusion schedule of this accelerator
"""

from typing import List, Optional, Set

from teaal.ir.component import *
from teaal.ir.hardware import Hardware
from teaal.ir.program import Program


class Fusion:
    """
    Representation of the fusion schedule of the accelerator
    """

    def __init__(self, hardware: Hardware) -> None:
        """
        Construct a new fusion object
        """
        self.hardware = hardware

        self.blocks: List[List[str]] = []
        self.curr_block: List[str] = []
        self.fused_ranks: List[str] = []

        self.curr_config: Optional[str] = None
        self.components_used: Set[str] = set()

        self.component_dict: Dict[str, List[str]] = {}

    def add_einsum(self, program: Program) -> None:
        """
        Add the information corresponding to this Einsum
        """
        einsum = program.get_equation().get_output().root_name()
        loop_ranks = program.get_loop_order().get_ranks()

        spacetime = program.get_spacetime()
        if not spacetime:
            raise ValueError("Undefined spacetime for Einsum " + einsum)

        space_ranks = spacetime.get_space()

        # Get the temporal ranks in all loop orders before the first spatial
        # rank
        fused_ranks: List[str]
        if space_ranks:
            fused_ranks = loop_ranks[:loop_ranks.index(space_ranks[0])]
        else:
            fused_ranks = loop_ranks

        # Get the components used for this Einsum
        components_used = set()
        for component in self.hardware.get_components(
                einsum, FunctionalComponent):
            if component.get_bindings()[einsum]:
                components_used.add(component.get_name())

        # Get the config
        config = self.hardware.get_config(einsum)

        # Check if the fusion conditions are met
        if config == self.curr_config and fused_ranks == self.fused_ranks and not self.components_used.intersection(
                components_used):
            self.curr_block.append(einsum)
            self.components_used = self.components_used.union(components_used)

        # Otherwise, start a new block
        else:
            self.blocks.append([einsum])
            self.curr_block = self.blocks[-1]
            self.fused_ranks = fused_ranks
            self.curr_config = config

        # Prepare to record the components contributing to the exectuion time
        self.component_dict[einsum] = []

    def add_component(self, einsum: str, component: str) -> None:
        """
        Add a component whose time is being tracked
        """
        self.component_dict[einsum].append(component)

    def get_blocks(self) -> List[List[str]]:
        """
        Get the Einsums organized by their fusion blocks
        """
        return self.blocks

    def get_components(self, einsum: str) -> List[str]:
        """
        Get the names of the components used for this Einsum
        """
        return self.component_dict[einsum]
