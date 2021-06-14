"""
Translate the partitiong specification
"""
from typing import cast, Generator

from lark.tree import Tree

from es2hfa.hfa.arg import AJust, AParam
from es2hfa.hfa.base import Argument, Expression, Statement
from es2hfa.hfa.expr import EInt, EMethod, EVar
from es2hfa.hfa.stmt import SAssign, SBlock
from es2hfa.ir.mapping import Mapping
from es2hfa.ir.tensor import Tensor


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

        # Finally, rename the tensor
        self.mapping.apply_partitioning(tensor)
        part_name = tensor.tensor_name()
        tmp_expr = cast(Expression, EVar("tmp"))
        block.add(cast(Statement, SAssign(part_name, tmp_expr)))

        return cast(Statement, block)

    def _uniform_shape(self, part: Tree, depth: int) -> Statement:
        """
        Partition with a uniform shape
        """
        # Build the shape
        dim = cast(Generator, part.scan_values(lambda _: True))
        arg1 = AJust(cast(Expression, EInt(next(dim))))

        # Build the depth and the arguments
        arg2 = AParam("depth", cast(Expression, EInt(depth)))
        args = [cast(Argument, arg1), cast(Argument, arg2)]

        # Build the call to splitUniform()
        part_call = EMethod("tmp", "splitUniform", args)
        part_assn = SAssign("tmp", cast(Expression, part_call))

        return cast(Statement, part_assn)
