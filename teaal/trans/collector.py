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

Translate the metrics collection
"""

from teaal.hifiber import *
from teaal.ir.component import *
from teaal.ir.metrics import Metrics
from teaal.ir.program import Program
from teaal.ir.tensor import Tensor
from teaal.trans.utils import TransUtils


class Collector:
    """
    Translate the metrics collection
    """

    def __init__(self, program: Program, metrics: Metrics) -> None:
        """
        Construct a collector object
        """
        self.program = program
        self.metrics = metrics

    def dump(self) -> Statement:
        """
        Dump metrics information
        """
        block = SBlock([])
        # If this is the first time, create a dictionary to store all
        # of the metrics information
        if self.program.get_einsum_ind() == 0:
            block.add(SAssign(AVar("metrics"), EDict({})))

        einsum = EString(self.program.get_equation().get_output().root_name())
        block.add(SAssign(AAccess(EVar("metrics"), einsum), EDict({})))

        # Add the memory traffic information
        for tensor in self.program.get_equation().get_tensors():
            # First revert the output to its loop nest form
            if tensor.get_is_output():
                tensor.reset()
                tensor.set_is_output(True)
                self.program.apply_all_partitioning(tensor)
                self.program.get_loop_order().apply(tensor)

            # Add the memory traffic information
            block.add(self.__mem_metrics(tensor))

            # Fix the output tensor
            if tensor.get_is_output():
                tensor.reset()
                tensor.set_is_output(True)

        # Add the compute information
        for compute in self.metrics.get_compute_components():
            block.add(self.__compute_metrics(compute))

        # Add the merger information
        for merge, name in self.metrics.get_merger_components():
            block.add(self.__merger_metrics(merge, name))

        return block

    @staticmethod
    def end() -> Statement:
        """
        End metrics collection
        """
        return SExpr(EMethod(EVar("Metrics"), "endCollect", []))

    def set_collecting(self, tensor_name: str, rank: str) -> Statement:
        """
        Collect the statistics about a tensor
        """
        tensor = self.program.get_equation().get_tensor(tensor_name)
        args = [AJust(EString(rank)), AJust(EBool(True))]
        call = EMethod(EVar(tensor.tensor_name()), "setCollecting", args)

        return SExpr(call)

    def start(self) -> Statement:
        """
        Start metrics collection
        """
        loop_order = self.program.get_loop_order()
        order = [EString(rank) for rank in loop_order.get_ranks()]
        call = EMethod(EVar("Metrics"), "beginCollect", [AJust(EList(order))])

        return SExpr(call)

    def __compute_metrics(self, component: FunctionalComponent) -> Statement:
        """
        Get the compute metrics for this hardware
        """
        einsum = self.program.get_equation().get_output().root_name()
        metrics = EAccess(EVar("metrics"), EString(einsum))
        block = SBlock([])

        for binding in component.get_bindings(einsum):
            if isinstance(component, LeaderFollowerComponent):
                rank = binding["rank"]
                leader = self.__get_leader(rank, binding["leader"])

                args = []
                args.append(AJust(EMethod(EVar("Metrics"), "dump", [])))
                args.append(AJust(EString(rank)))
                args.append(AJust(EInt(leader)))

                access = AAccess(metrics, EString(rank + " intersections"))
                count = EMethod(EVar("Compute"), "lfCount", args)
                block.add(SAssign(access, count))

            elif isinstance(component, SkipAheadComponent):
                rank = binding["rank"]

                args = []
                args.append(AJust(EMethod(EVar("Metrics"), "dump", [])))
                args.append(AJust(EString(rank)))

                access = AAccess(metrics, EString(rank + " intersections"))
                count = EMethod(EVar("Compute"), "skipCount", args)
                block.add(SAssign(access, count))

            else:
                op = binding["op"]

                args = []
                args.append(AJust(EMethod(EVar("Metrics"), "dump", [])))
                args.append(AJust(EString(op)))

                access = AAccess(metrics, EString(op))
                count = EMethod(EVar("Compute"), "opCount", args)
                block.add(SAssign(access, count))

        return block

    def __get_leader(self, rank: str, leader: str) -> int:
        """
        Get the index of the leader
        """
        i = 0
        for tensor in self.program.get_equation().get_tensors():
            if tensor.get_is_output():
                continue

            # TODO: Cover this when we allow more than two tensors
            # See test test_dump_leader_follower_not_intersected
            if rank not in tensor.get_ranks():
                continue  # pragma: no cover

            if tensor.root_name() == leader:
                return i

            i += 1

        raise ValueError("Tensor " + leader + " has no rank " + rank)

    def __mem_metrics(self, tensor: Tensor) -> Statement:
        """
        Get the memory metrics for a given tensor
        """
        block = SBlock([])

        # Dictionary accesses
        einsum = EString(self.program.get_equation().get_output().root_name())
        metrics = EAccess(EVar("metrics"), einsum)
        fp_access = (metrics, EString(tensor.root_name() + " footprint"))
        tf_access = (metrics, EString(tensor.root_name() + " traffic"))

        # No memory traffic if the tensor is not stored in DRAM
        if not self.metrics.in_dram(tensor):
            block.add(SAssign(AAccess(*fp_access), EInt(0)))
            block.add(SAssign(AAccess(*tf_access), EInt(0)))
            return block

        # Make a format for this tensor
        name = tensor.tensor_name()
        spec = TransUtils.build_expr(self.metrics.get_format(tensor))
        constr = EFunc("Format", [AJust(EVar(name)), AJust(spec)])
        format_ = name + "_format"
        block.add(SAssign(AVar(format_), constr))

        # Compute its memory footprint
        footprint = EMethod(EVar(format_), "getTensor", [])
        block.add(SAssign(AAccess(*fp_access), footprint))

        # If it is stationary, its footprint is its traffic, else compue
        # the traffic
        if self.metrics.on_chip_stationary(tensor):
            block.add(SAssign(AAccess(*tf_access), EAccess(*fp_access)))

        else:
            # First compute the traffic from loading the buffered subtrees
            traffic = self.__mem_traffic(tensor)

            # TODO: Make this more realistic
            # We assume that the other ranks are secretly buffered
            # somewhere else
            buffer_rank = self.metrics.get_on_chip_rank(tensor)
            prefix = tensor.get_prefix(buffer_rank)

            for rank in prefix:
                arg = AJust(EString(rank))
                rank_fp = EMethod(EVar(format_), "getRank", [arg])

            traffic = EBinOp(traffic, OAdd(), rank_fp)

            block.add(SAssign(AAccess(*tf_access), traffic))

        return block

    def __mem_traffic(self, tensor: Tensor) -> Expression:
        """
        Get the expression for computing the memory traffic for this tensor
        """
        buffer_ = self.metrics.get_on_chip_buffer(tensor)

        if isinstance(buffer_, BuffetComponent):
            args = []
            args.append(AJust(EVar(tensor.tensor_name())))
            args.append(AJust(EString(self.metrics.get_on_chip_rank(tensor))))
            args.append(AJust(EVar(tensor.tensor_name() + "_format")))

            return EMethod(EVar("Traffic"), "buffetTraffic", args)

        elif isinstance(buffer_, CacheComponent):
            capacity = buffer_.get_depth() * buffer_.get_width()

            args = []
            args.append(AJust(EVar(tensor.tensor_name())))
            args.append(AJust(EString(self.metrics.get_on_chip_rank(tensor))))
            args.append(AJust(EVar(tensor.tensor_name() + "_format")))
            args.append(AJust(EInt(capacity)))

            return EMethod(EVar("Traffic"), "cacheTraffic", args)

        else:
            # This error should be caught by the Hardware constructor
            raise ValueError(
                "Unknown MemoryComponent " +
                repr(buffer_))  # pragma: no cover

    def __merger_metrics(
            self,
            component: MergerComponent,
            binding: dict) -> Statement:
        """
        Get the merge metrics for this component
        """
        einsum = self.program.get_equation().get_output().root_name()
        metrics = EAccess(EVar("metrics"), EString(einsum))

        name = EVar(binding["tensor"] + "_" + "".join(binding["init_ranks"]))
        depth = EInt(binding["swap_depth"])

        # TODO: Use the correct merger parameters
        radix = TransUtils.build_expr(component.attrs["radix"])
        next_latency = TransUtils.build_expr(component.attrs["next_latency"])

        args = [AJust(arg) for arg in [name, depth, radix, next_latency]]

        access = AAccess(metrics, EString(name.gen() + " merge ops"))
        count = EMethod(EVar("Compute"), "swapCount", args)
        return SAssign(access, count)
