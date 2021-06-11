"""
Translate the header above the HFA loop nest
"""
from typing import cast, Generator

from es2hfa.hfa.arg import AJust, AParam
from es2hfa.hfa.base import Argument, Expression, Statement
from es2hfa.hfa.expr import EInt, EList, EMethod, EString, EVar
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
            # Get the partitioning information
            old_name = tensor.tensor_name()
            partitioning = mapping.get_partitioning(tensor)

            # Emit partitioning code if necessary
            if partitioning:
                # Rename the variable
                old_expr = cast(Expression, EVar(old_name))
                header.append(cast(Statement, SAssign("tmp", old_expr)))

                # Emit the partitioning code
                for i, ind in reversed(list(enumerate(tensor.get_inds()))):
                    # Continue if no partitioning across this dimension
                    if ind not in partitioning.keys():
                        continue

                    for j, part in enumerate(partitioning[ind]):
                        if part.data == "uniform_shape":
                            dim = cast(
                                Generator, part.scan_values(
                                    lambda _: True))
                            arg1 = AJust(cast(Expression, EInt(next(dim))))
                            arg2 = AParam(
                                "depth", cast(
                                    Expression, EInt(
                                        i + j)))
                            args = [cast(Argument, arg1), cast(Argument, arg2)]

                            part_call = EMethod("tmp", "splitUniform", args)
                            part_assn = SAssign(
                                "tmp", cast(Expression, part_call))

                            header.append(cast(Statement, part_assn))

                # Finally, rename the tensor
                mapping.apply_partitioning(tensor)
                part_name = tensor.tensor_name()
                tmp_expr = cast(Expression, EVar("tmp"))
                header.append(cast(Statement, SAssign(part_name, tmp_expr)))

            else:
                part_name = old_name

            # Swizzle for a concordant traversal
            mapping.apply_loop_order(tensor)
            new_name = tensor.tensor_name()

            # Emit code to perform the swizzle if necessary
            if part_name != new_name:
                ranks = [cast(Expression, EString(ind))
                         for ind in tensor.get_inds()]
                arg = cast(Argument, AJust(cast(Expression, EList(ranks))))
                swizzle_call = cast(
                    Expression, EMethod(
                        part_name, "swizzleRanks", [arg]))
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
