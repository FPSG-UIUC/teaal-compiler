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
from sympy import Add, Basic, Expr, Mul, Number, Symbol # type: ignore
from typing import cast, List, Optional, Set, Tuple, Union

from teaal.hifiber import *
from teaal.ir.program import Program
from teaal.ir.tensor import Tensor
from teaal.parse.utils import ParseUtils
from teaal.trans.coord_access import CoordAccess
from teaal.trans.utils import TransUtils


class Partitioner:
    """
    Generate the HiFiber code for the partitioning information
    """

    def __init__(self, program: Program, trans_utils: TransUtils) -> None:
        """
        Create a partitioner for a given program
        """
        self.program = program
        self.trans_utils = trans_utils

    def partition(self, tensor: Tensor, ranks: Tuple[str, ...]) -> Statement:
        """
        Partition the given tensor according to the stored program
        """
        part_ir = self.program.get_partitioning()
        part_ranks = part_ir.partition_rank(ranks)

        # We will build a block with the partitioning code
        block = SBlock([])

        if part_ranks is None:
            return block

        # Rename the variable
        next_tmp = AVar(self.trans_utils.next_tmp())
        old_name = tensor.tensor_name()
        block.add(SAssign(next_tmp, EVar(old_name)))

        if len(ranks) > 1:
            block.add(self.__apply_flatten(tensor, ranks))
        else:
            block.add(self.__apply_split(tensor, ranks[0], part_ranks[0]))

        # Rename the tensor
        part_name = tensor.tensor_name()
        tmp_expr = EVar(self.trans_utils.curr_tmp())
        block.add(SAssign(AVar(part_name), tmp_expr))

        # Rename the rank_ids
        block.add(TransUtils.build_set_rank_ids(tensor, part_name))

        return block

    def unpartition(self, tensor: Tensor) -> Statement:
        """
        Unpartition the given tensor
        """
        block = SBlock([])
        part_ir = self.program.get_partitioning()

        # First, undo swizzling
        curr_name = tensor.tensor_name()
        curr_ranks = tensor.get_ranks()

        # Compute the transformations that the tensor goes through
        tensor.reset()
        tensor.set_is_output(True)

        # Build a list of the transformations that the tensor will go through
        trans: List[Tuple[Union[str, Tuple[str, ...]], List[str]]] = []
        old_ranks = None
        new_ranks = tensor.get_ranks()
        while old_ranks != new_ranks:
            old_ranks = new_ranks
            swizzled_ranks = part_ir.swizzle_for_flattening(new_ranks)
            if swizzled_ranks != new_ranks:
                trans.append(("swizzle", new_ranks))
            tensor.swizzle(swizzled_ranks)

            valid_parts = part_ir.get_valid_parts(
                swizzled_ranks, part_ir.get_all_parts(), False)
            for part in valid_parts:
                trans.append((part, tensor.get_ranks()))
                self.program.apply_partitioning(tensor, part)

            new_ranks = tensor.get_ranks()

        # Undo the loop-order swizzle
        if curr_ranks != new_ranks:
            trans.append(("swizzle", new_ranks))

        # Return if there is nothing to do
        if not trans:
            return block

        # Start the temporary
        next_tmp = self.trans_utils.next_tmp()
        block.add(SAssign(AVar(next_tmp), EVar(curr_name)))

        rename_ranks = False
        for info, ranks in reversed(trans):
            curr_tmp = next_tmp

            if isinstance(info, tuple):
                # Undo split with flatten
                if len(info) == 1:
                    # Skip temporary ranks (created by occupancy-based
                    # partitioning)
                    _, suffix = part_ir.split_rank_name(info[0])
                    if suffix and suffix[-1] == "I":
                        continue

                    block.add(
                        self.__build_flatten(
                            "merge", ranks.index(
                                info[0]), len(
                                part_ir.get_part_spec(info)), "absolute"))
                    next_tmp = self.trans_utils.curr_tmp()

                # Otherwise unflatten
                else:
                    next_tmp = self.trans_utils.next_tmp()

                    # Build arguments
                    args = []
                    args.append(AParam("depth", EInt(ranks.index(info[0]))))
                    args.append(AParam("levels", EInt(len(info) - 1)))

                    # Build the call
                    call = EMethod(EVar(curr_tmp), "unflattenRanks", args)
                    block.add(SAssign(AVar(next_tmp), call))

                tensor.update_ranks(ranks)
                rename_ranks = True

            # Otherwise, we have a swizzle
            else:
                # First ensure the ranks have the correct name
                next_tmp = self.trans_utils.next_tmp()
                if rename_ranks:
                    block.add(TransUtils.build_set_rank_ids(tensor, curr_tmp))
                    rename_ranks = False

                tensor.swizzle(ranks)
                block.add(TransUtils.build_swizzle(tensor, curr_tmp, next_tmp))

        if rename_ranks:
            block.add(TransUtils.build_set_rank_ids(tensor, next_tmp))

        # Copy the tensor back
        block.add(SAssign(AVar(tensor.tensor_name()), EVar(next_tmp)))
        return block

    def __apply_flatten(self, tensor: Tensor,
                        ranks: Tuple[str, ...]) -> Statement:
        """
        Apply a flattening (as opposed to a split)
        """
        part_ir = self.program.get_partitioning()

        # Ensure that the ranks are in the correct order in the tensor
        i = -1
        for j, rank in enumerate(ranks):
            if i == -1:
                i = tensor.get_ranks().index(rank)

            if tensor.get_ranks()[i + j] != rank:
                raise ValueError("Cannot flatten together " +
                                 str(ranks) +
                                 " on tensor with ranks " +
                                 str(tensor.get_ranks()))

        assign = self.__build_flatten("flatten", i, len(ranks) - 1, "tuple")

        self.program.apply_partitioning(tensor, ranks)
        return assign

    def __apply_split(
            self,
            tensor: Tensor,
            rank: str,
            part_rank: str) -> Statement:
        """
        Apply a split (as opposed to a flattening)
        """
        part_ir = self.program.get_partitioning()
        block = SBlock([])

        partitioning = part_ir.get_part_spec((part_rank,))

        # Emit the partitioning code
        i = tensor.get_ranks().index(rank)

        first = True
        root_name = part_ir.get_root_name(part_rank)
        for j, part in enumerate(partitioning):
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

                part_num = len(partitioning) - j
                block.add(
                    self.__uniform_occupancy(
                        rank,
                        part_rank,
                        root_name +
                        str(part_num),
                        tensor,
                        part))

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
            self.program.apply_partitioning(tensor, (rank,))

        return block

    def __build_flatten(
            self,
            type_: str,
            depth: int,
            levels: int,
            style: str) -> Statement:
        """
        Build a call to the flattenRanks() function

        type_ should be either "flatten" (for flattenRanks) or "merge" (for mergeRanks)
        """
        # Build arguments
        args = []
        args.append(AParam("depth", EInt(depth)))
        args.append(AParam("levels", EInt(levels)))
        args.append(AParam("coord_style", EString(style)))

        # Build the call
        curr_tmp = self.trans_utils.curr_tmp()
        flat_call = EMethod(EVar(curr_tmp), type_ + "Ranks", args)
        next_tmp = AVar(self.trans_utils.next_tmp())
        return SAssign(next_tmp, flat_call)

    def __build_halo(self,
                     rank: str,
                     part_rank: str) -> Tuple[Optional[Expression],
                                              Optional[Expression]]:
        """
        Build halo expression; returns (pre_halo, post_halo) if they exist
        """
        root = self.program.get_partitioning().get_root_name(rank)
        proot = self.program.get_partitioning().get_root_name(part_rank)

        trans = self.program.get_coord_math().get_cond_expr(
            root.lower(), lambda expr: Symbol(
                proot.lower()) in expr.atoms(Symbol))
        trans = trans.subs(Symbol(proot.lower()), 0)

        if trans == 0:
            return None, None

        if trans.func == Add:
            terms = list(trans.args)
        else:
            terms = [trans]

        # Separate terms that will go in the prehalo and terms that will go
        # in the post_halo
        sym_pre_halo: Expr = Number(0)
        sym_post_halo: Expr = Number(0)
        for term in terms:
            if term.func == Mul:
                coeffs = term.atoms(Number)
                assert len(coeffs) == 1
                coeff = next(iter(coeffs))

                if coeff > 0:
                    sym_post_halo = sym_post_halo + term

                else:
                    sym_pre_halo = sym_pre_halo + -1 * cast(Expr, term)

            else:
                sym_post_halo = sym_post_halo + term

        # If there is a halo, substitute in the halo rank shapes
        def subs_halo_shapes(sym_halo: Expr) -> Optional[Expression]:
            if not sym_halo:
                return None

            halo = sym_halo
            for symbol in sym_halo.atoms(Symbol):
                halo = halo.subs(symbol, Symbol(str(symbol).upper()) - 1)

            return CoordAccess.build_expr(halo)

        return subs_halo_shapes(sym_pre_halo), subs_halo_shapes(sym_post_halo)

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
        parts: Expression
        if list(part.find_data("int_sz")):
            parts = EInt(ParseUtils.find_int(part, "int_sz"))
        else:
            parts = EVar(ParseUtils.find_str(part, "str_sz"))

        # Ceiling divide: (rank - 1) // parts + 1
        parens = EParens(EBinOp(EVar(part_rank), OSub(), EInt(1)))
        fdiv = EBinOp(parens, OFDiv(), parts)
        step = EBinOp(fdiv, OAdd(), EInt(1))

        # Build the splitUniform
        return self.__split_uniform(rank, part_rank, step, depth)

    def __split_equal(self, rank: str, part_rank: str,
                      size: Expression) -> Statement:
        """
        Build call to splitEqual
        """
        args: List[Argument] = [AJust(size)]
        pre_halo, post_halo = self.__build_halo(rank, part_rank)
        if pre_halo:
            args.append(AParam("pre_halo", pre_halo))

        if post_halo:
            args.append(AParam("post_halo", post_halo))

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
        # Make sure there is no translation needed between the leader and
        # follower tensors' ranks
        leader_tensor = self.program.get_equation().get_tensor(leader)
        leader_rank = leader_tensor.peek_clean()
        lroot = self.program.get_partitioning().get_root_name(leader_rank)
        root = self.program.get_partitioning().get_root_name(rank)
        if root != lroot:
            raise ValueError(
                "Cannot partition rank " +
                rank +
                " with a leader of a different rank (" +
                leader_rank.upper() +
                ")")

        fiber = EVar(leader_tensor.fiber_name())
        args: List[Argument] = [AJust(fiber)]
        pre_halo, post_halo = self.__build_halo(rank, part_rank)
        if pre_halo:
            args.append(AParam("pre_halo", pre_halo))

        if post_halo:
            args.append(AParam("post_halo", post_halo))

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

        expr = self.program.get_coord_math().get_cond_expr(
            rank, lambda expr: Symbol(part_rank.lower()) in expr.atoms(Symbol))
        sym_step = CoordAccess.build_expr(
            CoordAccess.isolate_rank(expr, part_rank))
        rank_step = cast(
            Expression, TransUtils.sub_hifiber(
                sym_step, EVar(
                    part_rank.lower()), step))

        args.append(AJust(rank_step))
        args.append(AParam("depth", EInt(depth)))

        # Add the halos
        pre_halo, post_halo = self.__build_halo(rank, part_rank)
        if pre_halo:
            args.append(AParam("pre_halo", pre_halo))
        if post_halo:
            args.append(AParam("post_halo", post_halo))

        # Build the call to splitUniform()
        curr_tmp = self.trans_utils.curr_tmp()
        part_call = EMethod(EVar(curr_tmp), "splitUniform", args)

        next_tmp = AVar(self.trans_utils.next_tmp())
        return SAssign(next_tmp, part_call)

    def __uniform_occupancy(
            self,
            rank: str,
            src_rank: str,
            dst_rank: str,
            tensor: Tensor,
            part: Tree) -> Statement:
        """
        Partition with a uniform occupancy
        """
        leader = self.program.get_partitioning().get_leader(src_rank, dst_rank)
        size: Expression
        if list(part.find_data("int_sz")):
            size = EInt(ParseUtils.find_int(part, "int_sz"))
        else:
            size = EVar(ParseUtils.find_str(part, "str_sz"))

        if tensor.root_name() == leader:
            return self.__split_equal(rank, src_rank, size)
        else:
            return self.__split_follower(rank, src_rank, leader)

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
        step: Expression
        if list(part.find_data("int_sz")):
            step = EInt(ParseUtils.find_int(part, "int_sz"))
        else:
            step = EVar(ParseUtils.find_str(part, "str_sz"))

        # Build the splitUniform()
        return self.__split_uniform(rank, part_rank, step, depth)
