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

from es2hfa.hfa import *
from es2hfa.ir.program import Program
from es2hfa.trans.graphics import Graphics
from es2hfa.trans.partitioner import Partitioner
from es2hfa.trans.utils import TransUtils


class Footer:
    """
    Generate the HFA code for the footer below the loop nest
    """

    @staticmethod
    def make_footer(
            program: Program,
            graphics: Graphics,
            partitioner: Partitioner) -> Statement:
        """
        Create the footer for the given einsum

        The footer must return the output tensor to its desired shape
        """
        footer = SBlock([])
        output = program.get_output()

        # First, undo swizzling
        curr_name = output.tensor_name()

        # To do this, we need to reset, and then re-apply partitioning, to get
        # the unswizzled name
        output.reset()
        program.apply_all_partitioning(output)
        part_name = output.tensor_name()

        # Generate undo swizzle code if necessary
        if curr_name != part_name:
            footer.add(TransUtils.build_swizzle(output, curr_name))
        footer.add(partitioner.unpartition(program.get_output()))

        # After resetting the output tensor, make sure that it still knows that
        # it is the output
        output.set_is_output(True)

        # Display the graphics if necessary
        footer.add(graphics.make_footer())

        return cast(Statement, footer)
