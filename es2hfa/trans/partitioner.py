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
from lark.tree import Tree
from sympy import Symbol
from typing import List, Set

from es2hfa.hfa import *
from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.utils import ParseUtils
from es2hfa.trans.coord_access import CoordAccess
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

    def partition(self, tensor: Tensor, rank: str) -> Statement:
        """
        Partition the given tensor according to the stored program
        """
        part_ir = self.program.get_partitioning()
        part_rank = part_ir.partition_rank(rank)

        # We will build a block with the partitioning code
        block = SBlock([])

        if part_rank is None:
            return block

        partitioning = part_ir.get_tensor_spec(tensor.get_ranks(), {part_rank})

        # Rename the variable
        next_tmp = AVar(self.trans_utils.next_tmp())
        old_name = tensor.tensor_name()
        block.add(SAssign(next_tmp, EVar(old_name)))

        # Emit the partitioning code
        i = tensor.get_ranks().index(rank)

        first = True
        for j, part in enumerate(partitioning[part_rank]):
            if part.data == "nway_shape":
                # If j != 0, then the rank we are partitioning is already in
                # the part_rank space
                if j == 0:
                    block.add(self.__nway_shape(rank, part_rank, part, i))
                else:
                    block.add(self.__nway_shape(part_rank, part_rank, part, i))

            elif part.data == "uniform_occupancy":
                # The dynamic partitioning must be of the current top rank
                if not first:
                    break

                block.add(
                    self.__uniform_occupancy(
                        rank, part_rank, tensor, part))

            elif part.data == "uniform_shape":
                block.add(self.__uniform_shape(rank, part_rank, part, i + j))

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

            flat_call = EMethod(
                EVar(curr_tmp), "flattenRanks", [
                    arg1, arg2, arg3])
            block.add(SAssign(next_tmp, flat_call))

        # Switch back to tensor name and rename the rank_ids
        new_name = AVar(tensor.tensor_name())
        tmp_name_expr = EVar(self.trans_utils.curr_tmp())
        block.add(SAssign(new_name, tmp_name_expr))
        block.add(TransUtils.build_set_rank_ids(tensor))

        return block

    def __build_halo(self, rank: str, part_rank: str) -> Expression:
        """
        Build halo expression
        """
        sexpr = self.program.get_coord_math().get_eqn_exprs()[
            Symbol(rank.lower())]
        part = self.program.get_partitioning()
        part_symbol = Symbol(part_rank.lower())

        replace = {}
        for symbol in sexpr.atoms(Symbol):
            final = str(symbol).upper()

            # Replace each value with the largest value it can take on
            if symbol != part_symbol:
                replace[symbol] = Symbol(final) - 1

        for symbol, max_ in replace.items():
            sexpr = sexpr.subs(symbol, max_)

        sexpr -= part_symbol
        if part_symbol in sexpr.atoms(Symbol):
            raise ValueError("Non-constant halo partitioning rank " + rank)

        return CoordAccess.build_expr(sexpr)

    def __nway_shape(
            self,
            rank: str,
            part_rank: str,
            part: Tree,
            depth: int) -> Statement:
        """
        Partition into the given number of partitions in coordinate space
        """
        # Build the step
        parts = ParseUtils.next_int(part)

        # Ceiling divide: (rank - 1) // parts + 1
        parens = EParens(EBinOp(EVar(part_rank), OSub(), EInt(1)))
        fdiv = EBinOp(parens, OFDiv(), EInt(parts))
        step = EBinOp(fdiv, OAdd(), EInt(1))

        # Build the splitUniform
        return self.__split_uniform(rank, part_rank, step, depth)

    def __split_equal(self, rank: str, part_rank: str, size: int) -> Statement:
        """
        Build call to splitEqual
        """
        args: List[Argument] = [AJust(EInt(size))]
        if rank != part_rank:
            args.append(AParam("halo", self.__build_halo(rank, part_rank)))

        curr_tmp = self.trans_utils.curr_tmp()
        part_call = EMethod(EVar(curr_tmp), "splitEqual", args)

        next_tmp = AVar(self.trans_utils.next_tmp())
        return SAssign(next_tmp, part_call)

    def __split_follower(
            self,
            rank: str,
            part_rank: str,
            leader: str) -> Statement:
        """
        Build a call to splitNonUniform
        """
        fiber = EVar(self.program.get_tensor(leader).fiber_name())
        args: List[Argument] = [AJust(fiber)]
        if rank != part_rank:
            args.append(AParam("halo", self.__build_halo(rank, part_rank)))

        curr_tmp = self.trans_utils.curr_tmp()
        part_call = EMethod(EVar(curr_tmp), "splitNonUniform", args)

        next_tmp = AVar(self.trans_utils.next_tmp())
        return SAssign(next_tmp, part_call)

    def __split_uniform(
            self,
            rank: str,
            part_rank: str,
            step: Expression,
            depth: int) -> Statement:
        """
        Build a call to splitUniform
        """
        # Build the depth and the arguments
        args: List[Argument] = []
        args.append(AJust(step))
        args.append(AParam("depth", EInt(depth)))

        if rank != part_rank:
            args.append(AParam("halo", self.__build_halo(rank, part_rank)))

        # Build the call to splitUniform()
        curr_tmp = self.trans_utils.curr_tmp()
        part_call = EMethod(EVar(curr_tmp), "splitUniform", args)

        next_tmp = AVar(self.trans_utils.next_tmp())
        return SAssign(next_tmp, part_call)

    def __uniform_occupancy(
            self,
            rank: str,
            part_rank: str,
            tensor: Tensor,
            part: Tree) -> Statement:
        """
        Partition with a uniform occupancy
        """
        leader = self.program.get_partitioning().get_leader(part)
        size = ParseUtils.find_int(part, "size")

        if tensor.root_name() == leader:
            return self.__split_equal(rank, part_rank, size)
        else:
            return self.__split_follower(rank, part_rank, leader)

    def __uniform_shape(
            self,
            rank: str,
            part_rank: str,
            part: Tree,
            depth: int) -> Statement:
        """
        Partition with a uniform shape
        """
        # Build the step
        step = ParseUtils.next_int(part)

        # Build the splitUniform()
        return self.__split_uniform(rank, part_rank, EInt(step), depth)
