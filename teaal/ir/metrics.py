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

from teaal.ir.component import *
from teaal.ir.hardware import Hardware
from teaal.ir.program import Program
from teaal.ir.tensor import Tensor
from teaal.parse.format import Format


class Metrics:
    """
    Representation of the metrics that need to be collected for this
    accelerator
    """

    def __init__(
            self,
            program: Program,
            hardware: Hardware,
            format_: Format) -> None:
        """
        Construct a new metrics object
        """
        self.program = program
        self.hardware = hardware
        self.format = format_

        # Check that we can collect metrics for this accelerator
        self.__check_configuration()

        # Get the final form of all tensors
        for tensor in self.program.get_equation().get_tensors():
            self.program.apply_all_partitioning(tensor)
            self.program.get_loop_order().apply(tensor)

        # Collect the memory traffic information
        self.__build_dram_tensors()
        self.__build_off_chip_traffic_info()
        self.__build_stationary()

        # Reset all tensors
        for tensor in self.program.get_equation().get_tensors():
            is_output = tensor.get_is_output()
            tensor.reset()
            tensor.set_is_output(is_output)

        # Collect other information
        self.__build_mergers()

    def get_compute_components(self) -> List[ComputeComponent]:
        """
        Get all relevant compute components for this Einsum
        """
        einsum = self.program.get_equation().get_output().root_name()
        return self.hardware.get_compute_components(einsum)

    def get_format(self, tensor: Tensor) -> dict:
        """
        Get the format specification for the given tensor
        """
        return self.format.get_spec(tensor.root_name())

    def get_merger_components(self) -> List[Tuple[MergerComponent, dict]]:
        """
        Get all relevant merger components and the relevant tensor being merged
        """
        return self.mergers

    def get_on_chip_buffer(self, tensor: Tensor) -> MemoryComponent:
        """
        Gets the on-chip buffer for a particular tensor
        """
        if not self.in_dram(tensor):
            raise ValueError(
                "Tensor " +
                tensor.root_name() +
                " not stored in DRAM")

        return self.on_chip_buffer[tensor.root_name()]

    def get_on_chip_rank(self, tensor: Tensor) -> str:
        """
        Returns the rank of the given tensor that is used for memory traffic
        """
        if not self.in_dram(tensor):
            raise ValueError(
                "Tensor " +
                tensor.root_name() +
                " not stored in DRAM")

        return self.on_chip_rank[tensor.root_name()][1]

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

    def __build_dram_tensors(self) -> None:
        """
        Build the set of tensors stored in DRAM
        """
        self.dram_tensors = set()
        einsum = self.program.get_equation().get_output().root_name()

        # For each tensor
        for tensor in self.program.get_equation().get_tensors():
            path = self.hardware.get_traffic_path(einsum, tensor.root_name())

            if not path or not isinstance(path[0], DRAMComponent):
                continue

            if len(path) < 2:
                raise ValueError(
                    "Tensor " +
                    tensor.root_name() +
                    " never buffered on chip")

            self.dram_tensors.add(tensor.root_name())

    def __build_mergers(self) -> None:
        """
        Build a list of mergers that will be relevant
        """
        all_mergers = self.hardware.get_merger_components()
        easy_access = {}

        for merger in all_mergers:
            for binding in merger.get_bindings():
                # Create a map from
                # (tensor name, init ranks, final ranks) to the component
                name = binding["tensor"]
                init = tuple(binding["init_ranks"])
                final = tuple(binding["final_ranks"])

                easy_access[(name, init, final)] = (merger, binding)

        self.mergers = []
        part = self.program.get_partitioning()

        def check_tensor(tensor):
            """
            Check if the tensor matches a merge operation, if so, add it
            """
            name = tensor.root_name()
            init = tuple(tensor.get_ranks())
            self.program.get_loop_order().apply(tensor)
            final = tuple(tensor.get_ranks())

            if (name, init, final) in easy_access.keys():
                self.mergers.append(easy_access[(name, init, final)])

        for tensor in self.program.get_equation().get_tensors():
            # If it is the output, we swizzle on the way out
            is_output = tensor.get_is_output()
            if is_output:
                name = tensor.root_name()

                # With the output, we first swizzle back, and then flatten
                self.program.apply_all_partitioning(tensor)
                self.program.get_loop_order().apply(tensor)

                init = tuple(tensor.get_ranks())
                tensor.reset()
                self.program.apply_all_partitioning(tensor)
                final = tuple(tensor.get_ranks())

                if (name, init, final) in easy_access.keys():
                    self.mergers.append(easy_access[(name, init, final)])

            else:
                name = tensor.root_name()

                # First apply all static partitioning
                for ranks in part.get_static_parts():
                    # TODO: allow flattening
                    if len(ranks) > 1:
                        raise ValueError("Cannot deal with this yet")
                    rank = ranks[0]
                    if rank in tensor.get_ranks():
                        # TODO Support flattening
                        self.program.apply_partitioning(tensor, (rank,))

                check_tensor(tensor)

                # Now check any dynamic swizzling after partitioning
                # opt_rank = tensor.peek()
                # while opt_rank is not None:
                #     if opt_rank.upper() in part.get_dyn_parts().keys():
                #         tensor.from_fiber()
                #         self.program.apply_partitioning(
                #             tensor, (opt_rank.upper(),))

                #         check_tensor(tensor)

                #     tensor.pop()
                #     opt_rank = tensor.peek()

            tensor.reset()
            tensor.set_is_output(is_output)

    def __build_off_chip_traffic_info(self) -> None:
        """
        Build a mapping from tensors to the rank buffered on chip
        """
        self.on_chip_rank = {}
        self.on_chip_buffer = {}
        einsum = self.program.get_equation().get_output().root_name()

        # For each tensor
        for tensor in self.program.get_equation().get_tensors():
            # We don't care about tensors not in DRAM
            if not self.in_dram(tensor):
                continue

            name = tensor.root_name()
            path = self.hardware.get_traffic_path(einsum, name)

            # Get the bindings
            mem_binding = path[0].get_binding(name)
            on_chip_binding = path[1].get_binding(name)

            # Indicates an error with Hardware.get_traffic_path()
            if not mem_binding or not on_chip_binding:
                raise ValueError("Something is wrong...")  # pragma: no cover

            # Build a dictionary of tensors to the
            # (rank in DRAM, rank in last on-chip buffer)
            self.on_chip_rank[name] = (mem_binding, on_chip_binding)

            # Save the component where the tensor is buffered on-chip
            self.on_chip_buffer[name] = path[1]

    def __build_stationary(self) -> None:
        """
        Build a set of DRAM -> on chip stationary tensors
        """
        self.stationary = set()
        einsum = self.program.get_equation().get_output().root_name()

        for name, (mem_rank, on_chip_rank) in self.on_chip_rank.items():
            tensor = self.program.get_equation().get_tensor(name)

            if mem_rank != "root":
                raise NotImplementedError

            prefix = tensor.get_prefix(on_chip_rank)

            # The tensor is stationary if its prefix is also a prefix to the
            # loop order
            if prefix == self.program.get_loop_order().get_ranks()[
                    :len(prefix)]:
                self.stationary.add(name)

    def __check_configuration(self) -> None:
        """
        There are many mappings that we cannot model right now. Make sure this
        is a legal configuration
        """
        # Check that there is no dynamic partitioning
        if self.program.get_partitioning().get_dyn_parts() != set():
            raise NotImplementedError

        # Check that there are at most three tensors (no danger of multiple
        # intersections per rank)
        if len(self.program.get_equation().get_tensors()) > 3:
            raise NotImplementedError
