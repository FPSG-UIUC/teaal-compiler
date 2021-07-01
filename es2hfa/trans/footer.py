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
    def make_footer(
            mapping: Mapping,
            canvas: Canvas,
            utils: Utils) -> Statement:
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
        partitioner = Partitioner(mapping, utils)
        footer.add(partitioner.unpartition(mapping.get_output()))

        # After reseting the output tensor, make sure that it still knows that
        # it is the output
        output.set_is_output(True)

        # Display the canvas if necessary
        if canvas.displayable():
            footer.add(canvas.display_canvas())

        return cast(Statement, footer)
