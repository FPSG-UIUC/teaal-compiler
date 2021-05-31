"""
Translate the header above the HFA loop nest
"""
from typing import cast

from es2hfa.hfa.arg import AJust
from es2hfa.hfa.base import Argument, Expression, Statement
from es2hfa.hfa.expr import EList, EMethod, EString
from es2hfa.hfa.stmt import SAssign, SBlock
from es2hfa.ir.mapping import Mapping


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
        # Get the tensors we need to generate headers for
        tensors = mapping.get_tensors()

        # Generate the header for each tensor
        header = []
        for tensor in tensors:
            # First, swizzle for a concordant traversal
            old_name = tensor.tensor_name()
            mapping.apply_loop_order(tensor)
            new_name = tensor.tensor_name()

            # Emit code to perform the swizzle if necessary
            if old_name != new_name:
                ranks = [cast(Expression, EString(ind))
                         for ind in tensor.get_inds()]
                arg = cast(Argument, AJust(cast(Expression, EList(ranks))))
                swizzle_call = cast(
                    Expression, EMethod(
                        old_name, "swizzleRanks", [arg]))
                header.append(cast(Statement, SAssign(new_name, swizzle_call)))

            # Emit code to get the root fiber
            get_root_call = cast(Expression, EMethod(new_name, "getRoot", []))
            header.append(
                cast(
                    Statement,
                    SAssign(
                        tensor.fiber_name(),
                        get_root_call)))

        return cast(Statement, SBlock(header))
