"""
Translate the footer below the loop nest
"""
from typing import cast

from es2hfa.hfa.arg import AParam
from es2hfa.hfa.base import Argument, Expression, Statement
from es2hfa.hfa.expr import EInt, EMethod, EString, EVar
from es2hfa.hfa.stmt import SAssign, SBlock
from es2hfa.ir.mapping import Mapping
from es2hfa.trans.utils import Utils


class Footer:
    """
    Generate the HFA code for the footer below the loop nest
    """

    @staticmethod
    def make_footer(mapping: Mapping) -> Statement:
        """
        Create the footer for the given einsum

        The footer must return the output tensor to its desired shape
        """
        footer = SBlock([])
        output = mapping.get_output()

        # First, undo swizzling
        curr_name = output.tensor_name()

        # To do this, we need to reset, and then re-apply partitioning, to get
        # the unswizzled name
        output.reset()
        mapping.apply_partitioning(output)
        part_name = output.tensor_name()

        # Generate undo swizzle code if necessary
        if curr_name != part_name:
            footer.add(Utils.build_swizzle(output, curr_name))

        # Now, undo partitioning, using a temporary to remove naming confusion
        output.reset()
        new_name = output.tensor_name()
        partitioning = mapping.get_partitioning(output)

        if partitioning:
            # Switch to name tmp
            part_name_expr = cast(Expression, EVar(part_name))
            footer.add(cast(Statement, SAssign("tmp", part_name_expr)))

            # For each dimension
            for i, ind in enumerate(output.get_inds()):
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

                    footer.add(cast(Statement, flat_assn))

            # Switch back to tensor name and rename the rank_ids
            tmp_name_expr = cast(Expression, EVar("tmp"))
            footer.add(cast(Statement, SAssign(new_name, tmp_name_expr)))
            footer.add(Utils.build_set_rank_ids(output))

        # After reseting the output tensor, make sure that it still knows that
        # it is the output
        output.set_is_output(True)

        return cast(Statement, footer)
