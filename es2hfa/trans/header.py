"""
Translate the header above the HFA loop nest
"""

from typing import cast

from es2hfa.hfa.arg import AParam
from es2hfa.hfa.base import Argument, Expression, Statement
from es2hfa.hfa.expr import EFunc, EList, EMethod, EString
from es2hfa.hfa.stmt import SAssign, SBlock
from es2hfa.ir.mapping import Mapping
from es2hfa.ir.tensor import Tensor
from es2hfa.trans.partitioning import Partitioner


class Header:
    """
    Generate the HFA code for the header above the loop nest
    """

    @staticmethod
    def make_header(mapping: Mapping) -> Statement:
        """
        Create the header for a given einsum

        Expects the Einsum to have already been added to the mapping, but
        modifies the tensors in the mapping for the current einsum
        """
        header = SBlock([])

        # First, create the output tensor
        output = mapping.get_output()
        out_constr = cast(
            Expression, EFunc(
                "Tensor", [
                    Header._build_rank_ids(output)]))
        out_assn = SAssign(output.tensor_name(), out_constr)
        header.add(cast(Statement, out_assn))

        # Create a partitioner
        partitioner = Partitioner(mapping)

        # Get the tensors we need to generate headers for
        tensors = mapping.get_tensors()

        # Generate the header for each tensor
        for tensor in tensors:
            # Partition if necessary
            header.add(partitioner.partition(tensor))

            # Swizzle for a concordant traversal
            old_name = tensor.tensor_name()
            mapping.apply_loop_order(tensor)
            new_name = tensor.tensor_name()

            # Emit code to perform the swizzle if necessary
            if old_name != new_name:
                arg = Header._build_rank_ids(tensor)
                swizzle_call = cast(
                    Expression, EMethod(
                        old_name, "swizzleRanks", [arg]))
                header.add(cast(Statement, SAssign(new_name, swizzle_call)))

            # Emit code to get the root fiber
            get_root_call = cast(Expression, EMethod(new_name, "getRoot", []))
            header.add(
                cast(
                    Statement,
                    SAssign(
                        tensor.fiber_name(),
                        get_root_call)))

        return cast(Statement, header)

    @staticmethod
    def _build_rank_ids(tensor: Tensor) -> Argument:
        """
        Build the rank_ids argument
        """
        ranks = [cast(Expression, EString(ind)) for ind in tensor.get_inds()]
        arg = AParam("rank_ids", cast(Expression, EList(ranks)))
        return cast(Argument, arg)
