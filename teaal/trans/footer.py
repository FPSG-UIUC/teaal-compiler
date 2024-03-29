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

from teaal.hifiber import *
from teaal.ir.program import Program
from teaal.trans.graphics import Graphics
from teaal.trans.partitioner import Partitioner
from teaal.trans.utils import TransUtils


class Footer:
    """
    Generate the HiFiber code for the footer below the loop nest
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
        output = program.get_equation().get_output()

        footer.add(partitioner.unpartition(output))

        # After resetting the output tensor, make sure that it still knows that
        # it is the output
        output.set_is_output(True)

        # Display the graphics if necessary
        footer.add(graphics.make_footer())

        return footer
