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

Representation of the metrics that need to be collected for this accelerator
"""

from typing import Tuple

from es2hfa.ir.component import *
from es2hfa.ir.hardware import Hardware
from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor


class Metrics:
    """
    Representation of the metrics that need to be collected for this
    accelerator
    """

    def __init__(self, program: Program, hardware: Hardware) -> None:
        """
        Construct a new metrics object
        """
        self.program = program
        self.hardware = hardware

        # Get the final form of all tensors
        for tensor in self.program.get_tensors():
            self.program.apply_all_partitioning(tensor)
            self.program.get_loop_order().apply(tensor)

        # Collect the memory traffic information
        self.__build_dram_tensors()
        self.__build_dram_access_rank()
        self.__build_stationary()

        # Reset all tensors
        for tensor in self.program.get_tensors():
            tensor.reset()

    def get_dram_access_rank(self, tensor: Tensor) -> str:
        """
        Returns the rank of the given tensor that is used for memory traffic
        """
        if not self.in_dram(tensor):
            raise ValueError(
                "Tensor " +
                tensor.root_name() +
                " not stored in DRAM")

        return self.dram_access_rank[tensor.root_name()][1]

    def in_dram(self, tensor: Tensor) -> bool:
        """
        Returns True if the tensor is stored in DRAM
        """
        return tensor.root_name() in self.dram_tensors

    def on_chip_stationary(self, tensor: Tensor) -> bool:
        """
        Returns True if this tensor is stationary (i.e., its DRAM traffic
        can be computed by calculating its footprint)
        """
        return tensor.root_name() in self.stationary

    def __build_dram_access_rank(self) -> None:
        """
        Build a mapping from tensors to the rank buffered on chip
        """
        self.dram_access_rank = {}
        einsum = self.program.get_output().root_name()

        # For each tensor
        for tensor in self.program.get_tensors():
            # We don't care about tensors not in DRAM
            if not self.in_dram(tensor):
                continue

            path = self.hardware.get_traffic_path(einsum, tensor.root_name())

            # Get the bindings
            mem_binding = path[0].get_binding(tensor.root_name())
            on_chip_binding = path[1].get_binding(tensor.root_name())

            # Indicates an error with Hardware.get_traffic_path()
            if not mem_binding or not on_chip_binding:
                raise ValueError("Something is wrong...")  # pragma: no cover

            self.dram_access_rank[tensor.root_name()] = (
                mem_binding, on_chip_binding)

    def __build_dram_tensors(self) -> None:
        """
        Build the set of tensors stored in DRAM
        """
        self.dram_tensors = set()
        einsum = self.program.get_output().root_name()

        # For each tensor
        for tensor in self.program.get_tensors():
            path = self.hardware.get_traffic_path(einsum, tensor.root_name())

            if not path or not isinstance(path[0], DRAMComponent):
                continue

            if len(path) < 2:
                raise ValueError(
                    "Tensor " +
                    tensor.root_name() +
                    " never buffered on chip")

            self.dram_tensors.add(tensor.root_name())

    def __build_stationary(self) -> None:
        """
        Build a set of DRAM -> on chip stationary tensors
        """
        self.stationary = set()
        einsum = self.program.get_output().root_name()

        for name, (mem_rank, on_chip_rank) in self.dram_access_rank.items():
            tensor = self.program.get_tensor(name)

            if mem_rank != "root":
                raise NotImplementedError

            if on_chip_rank == "root":
                prefix = []
            else:
                i = tensor.get_ranks().index(on_chip_rank)
                prefix = tensor.get_ranks()[:(i + 1)]

            # The tensor is stationary if its prefix is also a prefix to the
            # loop order
            if prefix == self.program.get_loop_order().get_ranks()[
                    :len(prefix)]:
                self.stationary.add(name)
