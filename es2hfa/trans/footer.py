"""
Translate the footer below the loop nest
"""
from typing import cast

from es2hfa.hfa.base import Statement
from es2hfa.hfa.stmt import SBlock
from es2hfa.ir.mapping import Mapping
from es2hfa.trans.canvas import Canvas
from es2hfa.trans.partitioning import Partitioner
from es2hfa.trans.utils import Utils


class Footer:
    """
    Generate the HFA code for the footer below the loop nest
    """

    @staticmethod
    def make_footer(mapping: Mapping, canvas: Canvas) -> Statement:
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

        # Display the canvas if necessary
        if canvas.displayable():
            footer.add(canvas.display_canvas())

        return cast(Statement, footer)
