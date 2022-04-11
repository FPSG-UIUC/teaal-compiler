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

Translate the partitiong specification
"""
from typing import Set

from lark.tree import Tree

from es2hfa.hfa import *
from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.utils import ParseUtils
from es2hfa.trans.utils import TransUtils


class Partitioner:
    """
    Generate the HFA code for the partitioning information
    """

    def __init__(self, program: Program, trans_utils: TransUtils) -> None:
        """
        Create a partitioner for a given program
        """
        self.program = program
        self.trans_utils = trans_utils

    def partition(self, tensor: Tensor, ranks: Set[str]) -> Statement:
        """
        Partition the given tensor according to the stored program
        """
        # Check if we need to partition at all
        part_ir = self.program.get_partitioning()
        partitioning = part_ir.get_tensor_spec(tensor.get_ranks(), ranks)
        if not partitioning:
            return SBlock([])

        # We will build a block with the partitioning code
        block = SBlock([])

        # Rename the variable
        next_tmp = AVar(self.trans_utils.next_tmp())
        old_name = tensor.tensor_name()
        block.add(SAssign(next_tmp, EVar(old_name)))

        # Emit the partitioning code
        for i, rank in reversed(list(enumerate(tensor.get_ranks()))):
            # Continue if no partitioning across this rank
            if rank not in partitioning.keys():
                continue

            first = True
            for j, part in enumerate(partitioning[rank]):
                if part.data == "nway_shape":
                    block.add(self.__nway_shape(rank, part, i))

                elif part.data == "uniform_occupancy":
                    # The dynamic partitioning must be of the current top rank
                    if not first:
                        break

                    block.add(self.__uniform_occupancy(tensor, part))

                elif part.data == "uniform_shape":
                    block.add(self.__uniform_shape(part, i + j))

                else:
                    # Note: there is no good way to test this error. Bad
                    # partitioning styles should be caught by the
                    # Partitioning
                    raise ValueError(
                        "Unknown partitioning style: " +
                        part.data)  # pragma: no cover

                first = False

            # Finally, update the tensor with this partition
            self.program.apply_partitioning(tensor, rank)

        # Rename the tensor
        part_name = AVar(tensor.tensor_name())
        tmp_expr = EVar(self.trans_utils.curr_tmp())
        block.add(SAssign(part_name, tmp_expr))

        # Rename the rank_ids
        block.add(TransUtils.build_set_rank_ids(tensor))

        return block

    def unpartition(self, tensor: Tensor) -> Statement:
        """
        Unpartition the given tensor
        """
        # Get the tensor names
        part_name = tensor.tensor_name()
        tensor.reset()

        # Get the partitioning
        part_ir = self.program.get_partitioning()
        ranks = set(part_ir.get_all_parts().keys())
        partitioning = part_ir.get_tensor_spec(tensor.get_ranks(), ranks)

        # If there was no partitioning, there is nothing to undo
        block = SBlock([])
        if not partitioning:
            return block

        # Switch to name tmp
        next_tmp = AVar(self.trans_utils.next_tmp())
        block.add(SAssign(next_tmp, EVar(part_name)))

        # For each rank
        for i, rank in enumerate(tensor.get_ranks()):
            if rank not in partitioning.keys():
                continue

            # Flatten the rank
            curr_tmp = self.trans_utils.curr_tmp()
            next_tmp = AVar(self.trans_utils.next_tmp())
            arg1 = AParam("depth", EInt(i))
            arg2 = AParam("levels", EInt(len(partitioning[rank])))
            arg3 = AParam("coord_style", EString("absolute"))

            flat_call = EMethod(curr_tmp, "flattenRanks", [arg1, arg2, arg3])
            block.add(SAssign(next_tmp, flat_call))

        # Switch back to tensor name and rename the rank_ids
        new_name = AVar(tensor.tensor_name())
        tmp_name_expr = EVar(self.trans_utils.curr_tmp())
        block.add(SAssign(new_name, tmp_name_expr))
        block.add(TransUtils.build_set_rank_ids(tensor))

        return block

    def __nway_shape(self, rank: str, part: Tree, depth: int) -> Statement:
        """
        Partition into the given number of partitions in coordinate space
        """
        # Build the step
        parts = ParseUtils.next_int(part)

        # Ceiling divide: (rank - 1) // parts + 1
        parens = EParens(EBinOp(EVar(rank), OSub(), EInt(1)))
        fdiv = EBinOp(parens, OFDiv(), EInt(parts))
        step = EBinOp(fdiv, OAdd(), EInt(1))

        # Build the splitUniform
        return self.__split_uniform(step, depth)

    def __split_equal(self, size: int) -> Statement:
        """
        Build call to splitEqual
        """
        arg = AJust(EInt(size))
        part_call = EMethod(self.trans_utils.curr_tmp(), "splitEqual", [arg])

        next_tmp = AVar(self.trans_utils.next_tmp())
        return SAssign(next_tmp, part_call)

    def __split_follower(self, leader: str) -> Statement:
        """
        Build a call to splitNonUniform
        """
        fiber = EVar(self.program.get_tensor(leader).fiber_name())
        curr_tmp = self.trans_utils.curr_tmp()
        part_call = EMethod(curr_tmp, "splitNonUniform", [AJust(fiber)])

        next_tmp = AVar(self.trans_utils.next_tmp())
        return SAssign(next_tmp, part_call)

    def __split_uniform(self, step: Expression, depth: int) -> Statement:
        """
        Build a call to splitUniform
        """
        # Build the depth and the arguments
        arg1 = AJust(step)
        arg2 = AParam("depth", EInt(depth))

        # Build the call to splitUniform()
        curr_tmp = self.trans_utils.curr_tmp()
        part_call = EMethod(curr_tmp, "splitUniform", [arg1, arg2])

        next_tmp = AVar(self.trans_utils.next_tmp())
        return SAssign(next_tmp, part_call)

    def __uniform_occupancy(self, tensor: Tensor, part: Tree) -> Statement:
        """
        Partition with a uniform occupancy
        """
        leader = self.program.get_partitioning().get_leader(part)
        size = ParseUtils.find_int(part, "size")

        if tensor.root_name() == leader:
            return self.__split_equal(size)
        else:
            return self.__split_follower(leader)

    def __uniform_shape(self, part: Tree, depth: int) -> Statement:
        """
        Partition with a uniform shape
        """
        # Build the step
        step = ParseUtils.next_int(part)

        # Build the splitUniform()
        return self.__split_uniform(EInt(step), depth)
