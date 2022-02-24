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

Translate the header above the HFA loop nest
"""

from typing import cast, Set

from es2hfa.hfa import *
from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.utils import ParseUtils
from es2hfa.trans.graphics import Graphics
from es2hfa.trans.partitioner import Partitioner
from es2hfa.trans.utils import TransUtils


class Header:
    """
    Generate the HFA code for loop headers
    """

    def __init__(self, program: Program, partitioner: Partitioner) -> None:
        """
        Construct a new Header object
        """
        self.program = program
        self.partitioner = partitioner

    def make_global_header(self, graphics: Graphics) -> Statement:
        """
        Create the header for a given einsum

        Expects the Einsum to have already been added to the program, but
        modifies the tensors in the program for the current einsum
        """
        header = SBlock([])

        # Prepare to partition all static ranks
        partitioning = self.program.get_partitioning().get_static_parts()
        for rank in partitioning:
            self.program.start_partitioning(rank)

        # Configure the output tensor
        output = self.program.get_output()
        self.program.apply_all_partitioning(output)
        self.program.get_loop_order().apply(output)

        header.add(cast(Statement, self.make_output()))

        # Get the tensors we need to generate headers for
        tensors = self.program.get_tensors().copy()
        tensors.remove(output)

        # Modify each of the input tensors if necessary
        for tensor in tensors:
            header.add(self.__format_tensor(tensor, set(partitioning.keys())))

        # Emit code to get the root fiber
        for tensor in self.program.get_tensors():
            header.add(Header.__get_root(tensor))

        # Generate graphics if needed
        header.add(graphics.make_header())

        return cast(Statement, header)

    def make_loop_header(self, rank: str) -> Statement:
        """
        Create an individual loop header
        """
        header = SBlock([])

        # If this rank is not dynamically partitioned, we are done
        dyn_parts = self.program.get_partitioning().get_dyn_parts()
        rank = cast(
            str, self.program.get_partitioning().get_curr_rank_id(rank))
        if rank not in dyn_parts.keys():
            return cast(Statement, header)

        # Otherwise, we assume leader-follower style partitioning with a clear
        # leader
        self.program.start_partitioning(rank)
        leader_name = ParseUtils.find_str(dyn_parts[rank][0], "leader")
        leader = self.program.get_tensor(leader_name)

        # Partition the leader first
        header.add(self.make_tensor_from_fiber(leader))
        header.add(self.__format_tensor(leader, {rank}))
        header.add(self.__get_root(leader))

        # Find all follower tensors
        tensors = self.program.get_tensors().copy()
        tensors.remove(leader)
        tensors.remove(self.program.get_output())

        # Partition the follower tensors
        for tensor in tensors:
            if rank in tensor.get_ranks():
                header.add(self.make_tensor_from_fiber(tensor))
                header.add(self.__format_tensor(tensor, {rank}))
                header.add(self.__get_root(tensor))

        return cast(Statement, header)

    def make_output(self) -> Statement:
        """
        Given an output tensor, generate the constructor
        """
        tensor = self.program.get_output()
        name = cast(Assignable, AVar(tensor.tensor_name()))
        arg = TransUtils.build_rank_ids(tensor)
        constr = cast(Expression, EFunc("Tensor", [arg]))
        return cast(Statement, SAssign(name, constr))

    def make_swizzle_root(self, tensor: Tensor) -> Statement:
        """
        Make calls to swizzleRanks() (as necessary) and getRoot()
        """
        block = SBlock([])

        # Swizzle for a concordant traversal
        old_name = tensor.tensor_name()
        self.program.get_loop_order().apply(tensor)
        new_name = tensor.tensor_name()

        # Emit code to perform the swizzle if necessary
        if old_name != new_name:
            block.add(TransUtils.build_swizzle(tensor, old_name))

        # Add the call to getRoot()
        get_root_call = EMethod(tensor.tensor_name(), "getRoot", [])
        call_expr = cast(Expression, get_root_call)
        fiber_name = cast(Assignable, AVar(tensor.fiber_name()))
        block.add(cast(Statement, SAssign(fiber_name, call_expr)))

        return cast(Statement, block)

    @staticmethod
    def make_tensor_from_fiber(tensor: Tensor) -> Statement:
        """
        Get a tensor from the current fiber
        """
        ranks = [cast(Expression, EString(rank))
                 for rank in tensor.get_ranks()]
        ranks_list = cast(Expression, EList(ranks))
        fiber_name = cast(Expression, EVar(tensor.fiber_name()))
        args = [cast(Argument, AParam("rank_ids", ranks_list)),
                cast(Argument, AParam("fiber", fiber_name))]

        from_fiber = EMethod("Tensor", "fromFiber", args)
        from_fiber_expr = cast(Expression, from_fiber)

        tensor_name = cast(Assignable, AVar(tensor.tensor_name()))
        return cast(Statement, SAssign(tensor_name, from_fiber_expr))

    def __format_tensor(self, tensor: Tensor, ranks: Set[str]) -> Statement:
        """
        Partition and then swizzle as necessary
        """
        header = SBlock([])

        # Partition if necessary
        header.add(self.partitioner.partition(tensor, ranks))

        # Swizzle for a concordant traversal
        old_name = tensor.tensor_name()
        self.program.get_loop_order().apply(tensor)
        new_name = tensor.tensor_name()

        # Emit code to perform the swizzle if necessary
        if old_name != new_name:
            header.add(TransUtils.build_swizzle(tensor, old_name))

        return cast(Statement, header)

    @staticmethod
    def __get_root(tensor: Tensor) -> Statement:
        """
        Get the root fiber for the given tensor
        """
        get_root_call = EMethod(tensor.tensor_name(), "getRoot", [])
        call_expr = cast(Expression, get_root_call)
        fiber_name = cast(Assignable, AVar(tensor.fiber_name()))
        return cast(Statement, SAssign(fiber_name, call_expr))
