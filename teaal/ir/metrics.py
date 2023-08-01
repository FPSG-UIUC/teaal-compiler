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

from typing import Dict, List, Optional, Tuple, Union

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

        self.__build_coiter_ranks()
        self.__build_fiber_traces()
        self.__build_used_traffic_paths()

    def get_buffered_data(self) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
        """
        Get, for each memory, the tensor, rank, and type (coord, payload, elem)
        buffered for all buffers that we actually need to track traffic

        Returns Dict[component, Dict[tensor, List[(rank, type)]]]
        """
        buffered_data: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
        types = ["coord", "payload", "elem"]
        for tensor, (_, paths_per_rank) in self.used_traffic_paths.items():
            for rank, paths in paths_per_rank.items():
                for type_, path in zip(types, paths):
                    # If this data is not loaded from somewhere to somewhere,
                    # there is nothing to track
                    if len(path) == 0:
                        continue

                    for buffer_ in path:
                        buffer_name = buffer_.get_name()
                        if buffer_name not in buffered_data:
                            buffered_data[buffer_name] = {}

                        if tensor not in buffered_data[buffer_name]:
                            buffered_data[buffer_name][tensor] = []

                        buffered_data[buffer_name][tensor].append(
                            (rank, type_))

        return buffered_data

    def get_collected_tensor_info(
            self, tensor: str) -> Set[Tuple[str, str, bool]]:
        """
        Get a specification for which ranks in the loop order need to be
        collected in the form {(rank, type, consumable)}, where type is one of
            - "fiber" - corresponding to iteration over that fiber
            - "iter" - corresponding to the iteration of the loop nest
        """
        # TODO: What about eager binding?
        # Collect traces for data traffic
        info = set()
        einsum = self.program.get_equation().get_output().root_name()
        if tensor in self.used_traffic_paths:
            for rank, paths in self.used_traffic_paths[tensor][1].items():
                if any(len(path) > 1 for path in paths):
                    info.add((rank, "fiber", False))

                # We only want to load the payload if we actually make it into
                # the loop
                if paths[1] and len(paths[1]) > 1:
                    info.add((rank, "iter", False))

        # Collect traces for intersection
        if not tensor == einsum:
            tensor_ir = self.program.get_equation().get_tensor(tensor)
            part_ir = self.program.get_partitioning()
            final_ranks = part_ir.partition_ranks(
                tensor_ir.get_init_ranks(), part_ir.get_all_parts(), True, True)

            for intersector in self.hardware.get_components(
                    einsum, IntersectorComponent):
                for binding in intersector.get_bindings()[einsum]:
                    if isinstance(intersector, LeaderFollowerComponent) and \
                            binding["leader"] != tensor:
                        continue

                    if binding["rank"] not in final_ranks:
                        continue

                    info.add((binding["rank"], "fiber", True))

        return info

    def get_fiber_trace(
            self,
            tensor: str,
            rank: str,
            is_read_trace: bool) -> str:
        """
        Get the name of the fiber trace for this fiber
        """
        return self.fiber_traces[rank][tensor][is_read_trace]

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

    def __build_coiter_ranks(self) -> None:
        """
        Map the ranks to the coiterators that coiterate over them
        """
        self.coiterators: Dict[str, Component] = {}
        einsum = self.program.get_equation().get_output().root_name()
        for intersector in self.hardware.get_components(
                einsum, IntersectorComponent):
            for binding in intersector.get_bindings()[einsum]:
                rank = binding["rank"]
                # Not clear how to map co-iterators onto components
                if rank in self.coiterators:
                    raise NotImplementedError

                self.coiterators[rank] = intersector

    def __build_fiber_traces(self) -> None:
        """
        Build the fiber traces

        self.fiber_traces: Dict[rank, Dict[tensor, Dict[is_read_trace, trace]]]
        """
        part_ir = self.program.get_partitioning()

        tensors = self.program.get_equation().get_tensors()
        final_ranks = []
        for tensor in tensors:
            final_ranks.append(set(part_ir.partition_ranks(
                tensor.get_init_ranks(), part_ir.get_all_parts(), True, True)))

        coiters: Dict[str, List[Tensor]] = {}
        for rank in self.program.get_loop_order().get_ranks():
            coiters[rank] = []
            for tensor, final in zip(tensors, final_ranks):
                if rank in final:
                    coiters[rank].append(tensor)

        self.fiber_traces: Dict[str, Dict[str, Dict[bool, str]]] = {}
        for rank in self.program.get_loop_order().get_ranks():
            self.fiber_traces[rank] = {}
            output, inputs = self.program.get_equation().get_iter(
                coiters[rank])

            parent = "iter"
            next_label = 0
            if output and not inputs:
                # If there is only an output, there is no separate read and
                # write trace
                self.fiber_traces[rank][output.root_name()] = {
                    True: parent, False: parent}
                continue

            if output:
                self.fiber_traces[rank][output.root_name()] = {
                    True: "populate_read_0", False: "populate_write_0"}

                parent = "populate_1"

                next_label = 2

            union_label: Optional[int] = None
            if len(inputs) > 1:
                union_label = next_label
                next_label += 2

            for i, term in enumerate(inputs):
                if len(term) == 1:
                    if i + 1 < len(inputs):
                        self.fiber_traces[rank][term[0].root_name()] = {
                            True: "union_" + str(union_label)}
                    # i + 1 == len(inputs)
                    else:
                        self.fiber_traces[rank][term[0].root_name()] = {
                            True: parent}

                # Otherwise we have multiple tensors intersected together
                else:
                    for j, tensor in enumerate(term[:-1]):
                        # Not clear which intersection should performed
                        # with this component
                        if rank in self.coiterators and len(inputs) > 1:
                            raise NotImplementedError

                        self.fiber_traces[rank][tensor.root_name()] = {
                            True: "intersect_" + str(next_label)}

                        if rank in self.coiterators and isinstance(
                                self.coiterators[rank], LeaderFollowerComponent) and j + 2 < len(term):
                            # TODO: The inputs need to be reorganized so that
                            # the leader is first

                            next_label += 1
                        else:
                            next_label += 2

                    self.fiber_traces[rank][term[-1].root_name()
                                            ] = {True: "intersect_" + str(next_label - 1)}

                if union_label is not None:
                    parent = "union_" + str(union_label + 1)
                    union_label = next_label
                    next_label += 2

    def __build_used_traffic_paths(self) -> None:
        """
        Build a dictionary of used loop formats:
        Dict[tensor, Tuple[format, Dict[rank, Tuple[coord_path, payload_path, elem_path]]]]
        """
        self.used_traffic_paths: Dict[str,
                                      Tuple[str,
                                            Dict[str,
                                                 Tuple[List[MemoryComponent],
                                                       List[MemoryComponent],
                                                       List[MemoryComponent]]]]] = {}
        for tensor_ir in self.program.get_equation().get_tensors():
            tensor = tensor_ir.root_name()
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
                    used = any(
                        len(path) > 1 for path in [
                            coord_path,
                            payload_path,
                            elem_path] if path is not None)

                    if not used:
                        continue

                    if used and tensor in self.used_traffic_paths and self.used_traffic_paths[
                            tensor][0] != format_:
                        raise ValueError("Multiple potential formats " +
                                         str({self.used_traffic_paths[tensor][0], format_}) +
                                         " for tensor " +
                                         tensor +
                                         " in Einsum " +
                                         einsum)

                    if tensor not in self.used_traffic_paths:
                        self.used_traffic_paths[tensor] = (format_, {})

                    self.used_traffic_paths[tensor][1][rank] = (
                        coord_path, payload_path, elem_path)
