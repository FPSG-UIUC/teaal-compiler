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

Translate the header above the HiFiber loop nest
"""

from sympy import Symbol  # type: ignore
from typing import Iterable, Optional, Set

from teaal.hifiber import *
from teaal.ir.metrics import Metrics
from teaal.ir.program import Program
from teaal.ir.tensor import Tensor
from teaal.parse.utils import ParseUtils
from teaal.trans.graphics import Graphics
from teaal.trans.partitioner import Partitioner
from teaal.trans.utils import TransUtils


class Header:
    """
    Generate the HiFiber code for loop headers
    """

    def __init__(
            self,
            program: Program,
            metrics: Optional[Metrics],
            partitioner: Partitioner) -> None:
        """
        Construct a new Header object
        """
        self.program = program
        self.metrics = metrics
        self.partitioner = partitioner

    def make_get_payload(
            self,
            tensor: Tensor,
            ranks: Iterable[str]) -> Statement:
        """
        Make a call to getPayload() or getPayloadRef()
        """
        if tensor.get_is_output():
            func = "getPayloadRef"
        else:
            func = "getPayload"

        args: List[Argument] = [AJust(EVar(rank.lower())) for rank in ranks]
        if self.metrics:
            args.append(
                AParam(
                    "trace",
                    EString(
                        "get_payload_" +
                        tensor.root_name())))
        call = EMethod(EVar(tensor.fiber_name()), func, args)

        for _ in ranks:
            tensor.pop()

        return SAssign(AVar(tensor.fiber_name()), call)

    @staticmethod
    def make_get_root(tensor: Tensor) -> Statement:
        """
        Make a call to getRoot()
        """
        get_root_call = EMethod(EVar(tensor.tensor_name()), "getRoot", [])
        fiber_name = AVar(tensor.fiber_name())
        return SAssign(fiber_name, get_root_call)

    def make_output(self) -> Statement:
        """
        Given an output tensor, generate the constructor
        """
        tensor = self.program.get_equation().get_output()
        self.program.apply_all_partitioning(tensor)
        self.program.get_loop_order().apply(tensor)

        arg0 = TransUtils.build_rank_ids(tensor)
        arg1 = AParam("name", EString(tensor.root_name()))
        args = self.__make_shape([arg0, arg1])
        constr = EFunc("Tensor", args)
        return SAssign(AVar(tensor.tensor_name()), constr)

    def make_swizzle(
            self,
            tensor: Tensor,
            ranks: List[str],
            type_: str) -> Statement:
        """
        Make call to swizzleRanks() (as necessary)
        """
        # Swizzle for a concordant traversal
        old_name = tensor.tensor_name()

        if type_ == "loop-order":
            self.program.get_loop_order().apply(tensor)
        elif type_ == "partitioning":
            self.program.apply_partition_swizzling(tensor)
        elif type_ == "metrics":
            tensor.swizzle(ranks)
        else:
            raise ValueError("Unknown swizzling reason: " + type_)

        new_name = tensor.tensor_name()

        # Emit code to perform the swizzle if necessary
        if old_name == new_name:
            return SBlock([])
        else:
            return TransUtils.build_swizzle(tensor, old_name, new_name)

    @staticmethod
    def make_tensor_from_fiber(tensor: Tensor) -> Statement:
        """
        Get a tensor from the current fiber
        """
        tensor.from_fiber()

        ranks = [EString(rank) for rank in tensor.get_ranks()]
        args = [AParam("rank_ids", EList(ranks)),
                AParam("fiber", EVar(tensor.fiber_name())),
                AParam("name", EString(tensor.root_name()))]

        from_fiber = EMethod(EVar("Tensor"), "fromFiber", args)
        tensor_name = AVar(tensor.tensor_name())
        return SAssign(tensor_name, from_fiber)

    def __make_shape(self, args: List[Argument]) -> List[Argument]:
        """
        Add the shape argument to a tensor if necessary (i.e. no input tensor
        has at least one rank of the output)
        """
        output = self.program.get_equation().get_output()
        part = self.program.get_partitioning()
        loop_order = self.program.get_loop_order()
        order = loop_order.get_ranks()

        # ranks = output.get_init_ranks()
        ranks = output.get_ranks()
        avail = [False for _ in ranks]

        final_pos = {}
        i = 0
        for pos in range(len(order)):
            if i == len(ranks):
                break

            if loop_order.is_ready(
                part.get_final_rank_id(
                    output.get_init_ranks(), ranks[i]), pos):
                final_pos[ranks[i]] = pos

                i += 1

        for tensor in self.program.get_equation().get_tensors():
            # Skip the output
            if tensor.get_is_output():
                continue

            # Mark all ranks in the input tensor available
            tranks = part.partition_ranks(
                tensor.get_ranks(), part.get_all_parts(), True, True)

            for trank in tranks:
                for i, rank in enumerate(ranks):
                    if loop_order.is_ready(trank, final_pos[rank]):
                        avail[i] = True

        # If at least one rank is not available, we need an explicit shape
        if not all(avail) or self.metrics is not None:
            # TODO: Test that this removes the partitioning
            unpart_ranks = [part.get_root_name(
                rank) for rank in output.get_ranks()]
            args.append(TransUtils.build_shape(unpart_ranks))

        return args
