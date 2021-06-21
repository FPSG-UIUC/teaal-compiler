"""
Translate the partitiong specification
"""
from typing import cast, Generator

from lark.tree import Tree

from es2hfa.hfa.arg import AJust, AParam
from es2hfa.hfa.base import Argument, Expression, Operator, Statement
from es2hfa.hfa.expr import EBinOp, EInt, EMethod, EString, EVar
from es2hfa.hfa.op import OFDiv
from es2hfa.hfa.stmt import SAssign, SBlock
from es2hfa.ir.mapping import Mapping
from es2hfa.ir.tensor import Tensor
from es2hfa.trans.utils import Utils


class Partitioner:
    """
    Generate the HFA code for the partitioning information
    """

    def __init__(self, mapping: Mapping) -> None:
        """
        Create a partitioner for a given mapping
        """
        self.mapping = mapping

    def partition(self, tensor: Tensor) -> Statement:
        """
        Partition the given tensor according to the stored mapping
        """
        # Check if we need to partition at all
        partitioning = self.mapping.get_partitioning(tensor)
        if not partitioning:
            return cast(Statement, SBlock([]))

        # We will build a block with the partitioning code
        block = SBlock([])

        # Rename the variable
        old_name = tensor.tensor_name()
        old_expr = cast(Expression, EVar(old_name))
        block.add(cast(Statement, SAssign("tmp", old_expr)))

        # Emit the partitioning code
        for i, ind in reversed(list(enumerate(tensor.get_inds()))):
            # Continue if no partitioning across this dimension
            if ind not in partitioning.keys():
                continue

            for j, part in enumerate(partitioning[ind]):
                if part.data == "uniform_shape":
                    block.add(self._uniform_shape(part, i + j))
                elif part.data == "nway_shape":
                    block.add(self._nway_shape(ind, part, i))
                else:
                    raise ValueError(
                        "Unknown partitioning style: " + part.data)

        # Rename the tensor
        self.mapping.apply_partitioning(tensor)
        part_name = tensor.tensor_name()
        tmp_expr = cast(Expression, EVar("tmp"))
        block.add(cast(Statement, SAssign(part_name, tmp_expr)))

        # Rename the rank_ids
        block.add(Utils.build_set_rank_ids(tensor))

        return cast(Statement, block)

    def unpartition(self, tensor: Tensor) -> Statement:
        """
        Unpartition the given tensor
        """
        # Get the tensor names
        part_name = tensor.tensor_name()
        tensor.reset()
        new_name = tensor.tensor_name()

        # Get the partitioning
        partitioning = self.mapping.get_partitioning(tensor)

        # If there was no partitioning, there is nothing to undo
        block = SBlock([])
        if not partitioning:
            return cast(Statement, block)

        # Switch to name tmp
        part_name_expr = cast(Expression, EVar(part_name))
        block.add(cast(Statement, SAssign("tmp", part_name_expr)))

        # For each dimension
        for i, ind in enumerate(tensor.get_inds()):
            if ind not in partitioning.keys():
                continue

            # TODO: Replace with a single call
            # Flatten the rank if necessary
            for _ in range(len(partitioning[ind])):
                arg1 = AParam("depth", cast(Expression, EInt(i)))
                arg2 = AParam("levels", cast(Expression, EInt(1)))
                arg3 = AParam(
                    "coord_style", cast(
                        Expression, EString("absolute")))
                args = [cast(Argument, arg) for arg in [arg1, arg2, arg3]]

                flat_call = EMethod("tmp", "flattenRanks", args)
                flat_assn = SAssign("tmp", cast(Expression, flat_call))

                block.add(cast(Statement, flat_assn))

        # Switch back to tensor name and rename the rank_ids
        tmp_name_expr = cast(Expression, EVar("tmp"))
        block.add(cast(Statement, SAssign(new_name, tmp_name_expr)))
        block.add(Utils.build_set_rank_ids(tensor))

        return cast(Statement, block)

    def _nway_shape(self, dim: str, part: Tree, depth: int) -> Statement:
        """
        Partition into the given number of partitions in coordinate space

        TODO: strange behavior if number of partitions is not a multiple of the
        total size of the dimension
        """
        # Build the step
        parts = next(cast(Generator, part.scan_values(lambda _: True)))
        step = EBinOp(
            cast(
                Expression, EVar(dim)), cast(
                Operator, OFDiv()), cast(
                Expression, EInt(parts)))

        # Build the splitUniform
        return self._split_uniform(cast(Expression, step), depth)

    def _split_uniform(self, step: Expression, depth: int) -> Statement:
        """
        Build a call to splitUniform
        """
        # Build the depth and the arguments
        arg1 = AJust(step)
        arg2 = AParam("depth", cast(Expression, EInt(depth)))
        args = [cast(Argument, arg1), cast(Argument, arg2)]

        # Build the call to splitUniform()
        part_call = EMethod("tmp", "splitUniform", args)
        part_assn = SAssign("tmp", cast(Expression, part_call))

        return cast(Statement, part_assn)

    def _uniform_shape(self, part: Tree, depth: int) -> Statement:
        """
        Partition with a uniform shape
        """
        # Build the step
        step = next(cast(Generator, part.scan_values(lambda _: True)))

        # Build the splitUniform()
        return self._split_uniform(cast(Expression, EInt(step)), depth)
