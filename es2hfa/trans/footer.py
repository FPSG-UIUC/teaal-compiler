"""
Translate the footer below the loop nest
"""
from typing import cast

from es2hfa.hfa.arg import AParam
from es2hfa.hfa.base import Argument, Expression, Statement
from es2hfa.hfa.expr import EInt, EMethod, EString, EVar
from es2hfa.hfa.stmt import SAssign, SBlock
from es2hfa.ir.mapping import Mapping
from es2hfa.trans.partitioning import Partitioner
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

        # Now, undo partitioning
        partitioner = Partitioner(mapping)
        footer.add(partitioner.unpartition(mapping.get_output()))

        # After reseting the output tensor, make sure that it still knows that
        # it is the output
        output.set_is_output(True)

        return cast(Statement, footer)
