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

        self.__build_format_options()
        self.__build_eager_evicts()
        self.__expand_eager()

        self.__build_coiter_ranks()
        self.__build_fiber_traces()
        self.__build_traffic_paths()

    def get_coiter(self, rank: str) -> Optional[Component]:
        """
        Get the coiterator used for this rank
        """
        if rank not in self.coiterators:
            return None

        return self.coiterators[rank]

    def get_coiter_traces(self, coiter: str, rank: str) -> List[str]:
        """
        Get the trace names used for this coiterator
        """
        return self.coiter_traces[coiter][rank]

    def get_collected_tensor_info(
            self, tensor: str) -> Set[Tuple[str, str, bool]]:
        """
        Get a specification for which ranks in the loop order need to be
        collected in the form {(rank, type, consumable)}, where type is one of
            - "fiber" - corresponding to iteration over that fiber
            - "iter" - corresponding to the iteration of the loop nest
            - rank - the rank that the eager iteration starts at
        """
        # Collect traces for data traffic
        info = set()
        einsum = self.program.get_equation().get_output().root_name()
        if tensor in self.traffic_paths:
            for rank, paths in self.traffic_paths[tensor][1].items():
                for i, path in enumerate(paths):
                    for component, style in path:
                        if isinstance(component, DRAMComponent):
                            continue

                        if style == "lazy":
                            info.add((rank, "fiber", False))
                            fiber_trace = self.get_fiber_trace(
                                tensor, rank, True)
                            if i == 1 and fiber_trace != "iter" and fiber_trace[:11] != "get_payload":
                                info.add((rank, "iter", False))

                        else:
                            info.add((rank, style, False))

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

    def get_eager_evict_on(self, tensor: str, rank: str) -> List[str]:
        """
        Get the ranks eager load should be evicted on in loop order

        TODO: Test that they are in loop order
        """
        ranks = []
        for loop_rank, evicts in self.eager_evicts.items():
            if (tensor, rank) in evicts:
                ranks.append(loop_rank)
        return ranks

    def get_eager_evicts(self, rank: str) -> List[Tuple[str, str]]:
        """
        Get the subtrees that were eager loaded and should be evicted on this
        rank
        """
        if rank not in self.eager_evicts:
            return []

        return self.eager_evicts[rank]

    def get_eager_write(self) -> bool:
        """
        Returns True if the kernel perfoms an eager write
        """
        return self.eager_write

    def get_fiber_trace(
            self,
            tensor: str,
            rank: str,
            is_read_trace: bool) -> str:
        """
        Get the name of the fiber trace for this fiber
        """
        # If the rank is not in the set of fiber_traces (not in the loop
        # order), it must be being iterated with a get payload
        if rank not in self.fiber_traces:
            return "get_payload_" + tensor
        return self.fiber_traces[rank][tensor][is_read_trace]

    def get_format(self) -> Format:
        """
        Get the parsed format yaml
        """
        return self.format

    def get_hardware(self) -> Hardware:
        """
        Get the hardware IR
        """
        return self.hardware

    def get_loop_formats(self) -> Dict[str, str]:
        """
        Get the tensors that have assigned formats during the loop nest as
        well as the corresponding format
        """
        loop_formats = {}
        for tensor, (format_, _) in self.traffic_paths.items():
            loop_formats[tensor] = format_
        return loop_formats

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

    def get_source_memory(
            self,
            component: str,
            tensor: str,
            rank: str,
            type_: str) -> Optional[MemoryComponent]:
        """
        Get the source for this data
        """
        t = ["coord", "payload", "elem"].index(type_)

        if tensor not in self.traffic_paths:
            return None

        path = self.traffic_paths[tensor][1][rank][t]
        component_ir = self.hardware.get_component(component)
        if not isinstance(component_ir, MemoryComponent):
            raise ValueError(
                "Destination component " +
                component +
                " not a memory")

        inds = [i for i, (comp, _) in enumerate(path) if comp == component_ir]
        if not inds:
            return None

        if inds[0] == 0:
            return None

        return path[inds[0] - 1][0]

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
                # Not clear how to map co-iterators onto multiple components
                if rank in self.coiterators:
                    raise NotImplementedError

                self.coiterators[rank] = intersector

    def __build_eager_evicts(self) -> None:
        """
        Build a dictionary describing the ranks eager accesses will be evicted on

        self.eager_evicts: Dict[evict_rank, List[Tuple[tensor, root_rank]]]
        """
        einsum = self.program.get_equation().get_output().root_name()

        self.eager_evicts: Dict[str, List[Tuple[str, str]]] = {}
        for buffet in self.hardware.get_components(einsum, BuffetComponent):
            for binding in buffet.get_bindings()[einsum]:
                if binding["style"] != "eager":
                    continue

                evict_on = binding["evict-on"]
                if evict_on not in self.eager_evicts:
                    self.eager_evicts[evict_on] = []

                self.eager_evicts[evict_on].append(
                    (binding["tensor"], binding["root"]))

    def __build_fiber_traces(self) -> None:
        """
        Build the fiber traces

        self.fiber_traces: Dict[rank, Dict[tensor, Dict[is_read_trace, trace]]]
        self.coiter_traces: Dict[component, Dict[rank, List[trace]]]
        """
        part_ir = self.program.get_partitioning()
        einsum = self.program.get_equation().get_output().root_name()

        # Get the ranks/rank order of tensors during the loop nest
        tensors = self.program.get_equation().get_tensors()
        final_ranks = []
        for tensor in tensors:
            final_ranks.append(set(part_ir.partition_ranks(
                tensor.get_init_ranks(), part_ir.get_all_parts(), True, True)))

        # Get the tensors iterated on for each rank
        coiters: Dict[str, List[Tensor]] = {}
        for rank in self.program.get_loop_order().get_ranks():
            coiters[rank] = []
            for tensor, final in zip(tensors, final_ranks):
                if rank in final:
                    coiters[rank].append(tensor)

        # Get the corresponding traces
        self.fiber_traces: Dict[str, Dict[str, Dict[bool, str]]] = {}
        self.coiter_traces: Dict[str, Dict[str, List[str]]] = {}
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
                    # Not clear which intersection should performed
                    # with this component
                    if rank in self.coiterators and len(inputs) > 1:
                        raise NotImplementedError

                    # Reorganize the leader to be first
                    tensors = term.copy()
                    if rank in self.coiterators and isinstance(
                            self.coiterators[rank], LeaderFollowerComponent):
                        for binding in self.coiterators[rank].get_bindings()[
                                einsum]:
                            if binding["rank"] == rank:
                                leader = binding["leader"]
                                break

                        leader_tensor = self.program.get_equation().get_tensor(leader)
                        tensors.remove(leader_tensor)
                        tensors.insert(0, leader_tensor)

                    for j, tensor in enumerate(tensors[:-1]):
                        self.fiber_traces[rank][tensor.root_name()] = {
                            True: "intersect_" + str(next_label)}

                        if rank in self.coiterators and isinstance(
                                self.coiterators[rank], LeaderFollowerComponent) and j + 2 < len(tensors):
                            next_label += 1
                        else:
                            next_label += 2

                    self.fiber_traces[rank][tensors[-1].root_name()
                                            ] = {True: "intersect_" + str(next_label - 1)}

                    if rank in self.coiterators:
                        coiter = self.coiterators[rank]
                        if coiter.get_name() not in self.coiter_traces:
                            self.coiter_traces[coiter.get_name()] = {}
                        self.coiter_traces[coiter.get_name()][rank] = []

                        traces = self.coiter_traces[coiter.get_name()][rank]
                        if isinstance(coiter, LeaderFollowerComponent):
                            # TODO: Can the leader-follower component store
                            # this info itself
                            leader = ""
                            for binding in coiter.get_bindings()[einsum]:
                                if binding["rank"] == rank:
                                    leader = binding["leader"]
                                    break
                            traces.append(
                                self.fiber_traces[rank][leader][True])

                        else:
                            # Do not support tracing intersection of more than
                            # two components
                            if len(tensors) > 2:
                                raise NotImplementedError

                            for tensor in tensors:
                                traces.append(
                                    self.fiber_traces[rank][tensor.root_name()][True])

                if union_label is not None:
                    parent = "union_" + str(union_label + 1)
                    union_label = next_label
                    next_label += 2

    def __build_format_options(self) -> None:
        """
        Build a set of possible formats for each tensor

        self.format_options: Dict[tensor, List[format]]
        """
        self.format_options: Dict[str, List[str]] = {}
        for tensor_ir in self.program.get_equation().get_tensors():
            tensor = tensor_ir.root_name()
            self.format_options[tensor] = []

            spec = self.format.get_spec(tensor)

            # Identify the formats that can correspond to the iteration of this
            # loop nest
            loop_order = self.program.get_loop_order()
            for format_ in spec:
                format_ranks = spec[format_]["rank-order"]

                temp_tensor = Tensor(tensor, format_ranks)
                loop_order.apply(temp_tensor)

                if temp_tensor.get_ranks() == format_ranks:
                    self.format_options[tensor].append(format_)

    def __build_traffic_paths(self) -> None:
        """
        Build a dictionary of used loop formats:
        Dict[tensor, Tuple[format, Dict[rank, Tuple[coord_path, payload_path, elem_path]]]]
        """
        self.traffic_paths: Dict[str,
                                 Tuple[str,
                                       Dict[str,
                                            Tuple[List[Tuple[MemoryComponent, str]],
                                                  List[Tuple[MemoryComponent, str]],
                                                  List[Tuple[MemoryComponent, str]]]]]] = {}
        for tensor_ir in self.program.get_equation().get_tensors():
            tensor = tensor_ir.root_name()
            spec = self.format.get_spec(tensor)

            # Build the set of specs to collect
            einsum = self.program.get_equation().get_output().root_name()

            # TODO: What about eager binding?
            for format_ in self.format_options[tensor]:
                for rank in spec[format_]:
                    if rank == "rank-order":
                        continue

                    coord_path = self.hardware.get_traffic_path(
                        tensor, rank, "coord", format_)
                    payload_path = self.hardware.get_traffic_path(
                        tensor, rank, "payload", format_)
                    elem_path = self.hardware.get_traffic_path(
                        tensor, rank, "elem", format_)

                    if tensor in self.traffic_paths and self.traffic_paths[
                            tensor][0] != format_:
                        raise ValueError("Multiple potential formats " +
                                         str({self.traffic_paths[tensor][0], format_}) +
                                         " for tensor " +
                                         tensor +
                                         " in Einsum " +
                                         einsum)

                    if tensor not in self.traffic_paths:
                        self.traffic_paths[tensor] = (format_, {})

                    self.traffic_paths[tensor][1][rank] = (
                        coord_path, payload_path, elem_path)

    def __expand_eager(self):
        """
        Expand all eager bindings
        """
        einsum = self.program.get_equation().get_output().root_name()

        self.eager_write = False
        for tensor_ir in self.program.get_equation().get_tensors():
            tensor = tensor_ir.root_name()
            spec = self.format.get_spec(tensor)

            for format_ in self.format_options[tensor]:
                types = []
                for rank in spec[format_]["rank-order"]:
                    types.append([])
                    if "layout" in spec[format_][rank] and \
                            spec[format_][rank]["layout"] == "interleaved":
                        types[-1].append("elem")
                        continue

                    if "cbits" in spec[format_][rank] and \
                            spec[format_][rank]["cbits"] > 0:
                        types[-1].append("coord")

                    if "pbits" in spec[format_][rank] and \
                            spec[format_][rank]["pbits"] > 0:
                        types[-1].append("payload")

                for component in self.hardware.get_components(
                        einsum, BuffetComponent):

                    if tensor_ir.get_is_output():
                        for binding in component.get_bindings()[einsum]:
                            if binding["style"] == "eager":
                                self.eager_write = True

                    component.expand_eager(
                        einsum, tensor, format_, spec[format_]["rank-order"], types)
