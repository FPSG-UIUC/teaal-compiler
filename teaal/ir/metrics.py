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

    def get_collected_tensor_info(self, tensor: str) -> Set[Tuple[str, str]]:
        """
        Get a specification for which ranks in the loop order need to be
        collected in the form {(rank, type)}, where type is one of
            - "fiber" - corresponding to iteration over that fiber
            - "iter" - corresponding to the iteration of the loop nest
        """
        spec = self.format.get_spec(tensor)

        # Identify the formats that can correspond to the iteration of this
        # loop nest
        formats = []
        loop_order = self.program.get_loop_order()
        for format_ in spec:
            format_ranks = spec[format_]["rank-order"]

            temp_tensor = Tensor(tensor, format_ranks)
            loop_order.apply(temp_tensor)

            if temp_tensor.get_ranks() == format_ranks:
                formats.append(format_)

        # Build the set of specs to collect
        einsum = self.program.get_equation().get_output().root_name()
        loop_format: Optional[str] = None
        info = set()

        # TODO: What about eager binding?
        for format_ in formats:
            for rank in spec[format_]:
                if rank == "rank-order":
                    continue

                coord_path = self.hardware.get_traffic_path(
                    einsum, tensor, rank, "coord", format_)
                payload_path = self.hardware.get_traffic_path(
                    einsum, tensor, rank, "payload", format_)
                elem_path = self.hardware.get_traffic_path(
                    einsum, tensor, rank, "elem", format_)
                used = coord_path or payload_path or elem_path

                if used and loop_format is not None and format_ != loop_format:
                    raise ValueError("Multiple potential formats " +
                                     str([loop_format, format_]) +
                                     " for tensor " +
                                     tensor +
                                     " in Einsum " +
                                     einsum)
                loop_format = format_

                if used:
                    info.add((rank, "fiber"))

                if payload_path:
                    info.add((rank, "iter"))

        return info

    def get_merger_init_ranks(self, tensor: str,
                              final_ranks: List[str]) -> Optional[List[str]]:
        """
        Get the initial ranks for merges that must be tracked by the hardware
        """
        einsum = self.program.get_equation().get_output().root_name()
        mergers = self.hardware.get_components(einsum, MergerComponent)
        init_ranks: Optional[List[str]] = None
        for merger in mergers:
            opt_init_ranks = merger.get_init_ranks(einsum, tensor, final_ranks)

            if opt_init_ranks is None:
                continue

            if init_ranks is not None:
                raise ValueError(
                    "Multiple bindings for merge of tensor " +
                    tensor +
                    " to final rank order " +
                    str(final_ranks))

            init_ranks = opt_init_ranks

        return init_ranks

    # TODO: Delete
    #     # Check that we can collect metrics for this accelerator
    #     self.__check_configuration()

    #     # Get the final form of all tensors
    #     for tensor in self.program.get_equation().get_tensors():
    #         self.program.apply_all_partitioning(tensor)
    #         self.program.get_loop_order().apply(tensor)

    #     # Reset all tensors
    #     for tensor in self.program.get_equation().get_tensors():
    #         is_output = tensor.get_is_output()
    #         tensor.reset()
    #         tensor.set_is_output(is_output)

    #     # Collect other information
    #     self.__build_mergers()

    # def get_functional_components(self) -> List[FunctionalComponent]:
    #     """
    #     Get all relevant compute components for this Einsum
    #     """
    #     einsum = self.program.get_equation().get_output().root_name()
    #     return self.hardware.get_functional_components(einsum)

    # def get_format(self, tensor: Tensor) -> dict:
    #     """
    #     Get the format specification for the given tensor
    #     """
    #     return self.format.get_spec(tensor.root_name())

    # def get_merger_components(self) -> List[Tuple[MergerComponent, dict]]:
    #     """
    #     Get all relevant merger components and the relevant tensor being merged
    #     """
    #     return self.mergers

    # def __build_mergers(self) -> None:
    #     """
    #     Build a list of mergers that will be relevant
    #     """
    #     all_mergers = self.hardware.get_merger_components()
    #     easy_access = {}

    #     einsum = self.program.get_equation().get_output().root_name()
    #     for merger in all_mergers:
    #         for binding in merger.get_bindings()[einsum]:
    #             # Create a map from
    #             # (tensor name, init ranks, final ranks) to the component
    #             name = binding["tensor"]
    #             init = tuple(binding["init_ranks"])
    #             final = tuple(binding["final_ranks"])

    #             easy_access[(name, init, final)] = (merger, binding)

    #     self.mergers = []
    #     part = self.program.get_partitioning()

    #     def check_tensor(tensor):
    #         """
    #         Check if the tensor matches a merge operation, if so, add it
    #         """
    #         name = tensor.root_name()
    #         init = tuple(tensor.get_ranks())
    #         self.program.get_loop_order().apply(tensor)
    #         final = tuple(tensor.get_ranks())

    #         if (name, init, final) in easy_access.keys():
    #             self.mergers.append(easy_access[(name, init, final)])

    #     for tensor in self.program.get_equation().get_tensors():
    #         # If it is the output, we swizzle on the way out
    #         is_output = tensor.get_is_output()
    #         if is_output:
    #             name = tensor.root_name()

    #             # With the output, we first swizzle back, and then flatten
    #             self.program.apply_all_partitioning(tensor)
    #             self.program.get_loop_order().apply(tensor)

    #             init = tuple(tensor.get_ranks())
    #             tensor.reset()
    #             self.program.apply_all_partitioning(tensor)
    #             final = tuple(tensor.get_ranks())

    #             if (name, init, final) in easy_access.keys():
    #                 self.mergers.append(easy_access[(name, init, final)])

    #         else:
    #             name = tensor.root_name()

    #             # First apply all static partitioning
    #             for ranks in part.get_static_parts():
    #                 # TODO: allow flattening
    #                 if len(ranks) > 1:
    #                     raise ValueError("Cannot deal with this yet")
    #                 rank = ranks[0]
    #                 if rank in tensor.get_ranks():
    #                     # TODO Support flattening
    #                     self.program.apply_partitioning(tensor, (rank,))

    #             check_tensor(tensor)

    #             # TODO: What is this? Do we care about it?
    #             # Now check any dynamic swizzling after partitioning
    #             # opt_rank = tensor.peek()
    #             # while opt_rank is not None:
    #             #     if opt_rank.upper() in part.get_dyn_parts().keys():
    #             #         tensor.from_fiber()
    #             #         self.program.apply_partitioning(
    #             #             tensor, (opt_rank.upper(),))

    #             #         check_tensor(tensor)

    #             #     tensor.pop()
    #             #     opt_rank = tensor.peek()

    #         tensor.reset()
    #         tensor.set_is_output(is_output)

    # def __check_configuration(self) -> None:
    #     """
    #     There are many mappings that we cannot model right now. Make sure this
    #     is a legal configuration
    #     """
    #     # Check that there is no dynamic partitioning
    #     if self.program.get_partitioning().get_dyn_parts() != set():
    #         raise NotImplementedError

    #     # Check that there are at most three tensors (no danger of multiple
    #     # intersections per rank)
    #     if len(self.program.get_equation().get_tensors()) > 3:
    #         raise NotImplementedError
