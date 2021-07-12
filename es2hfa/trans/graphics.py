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

Translate all relevant graphics information
"""

from typing import cast

from es2hfa.hfa.base import Statement
from es2hfa.hfa.stmt import SBlock
from es2hfa.ir.program import Program
from es2hfa.trans.canvas import Canvas


class Graphics:
    """
    Generate the HFA code for displaying tensors
    """

    def __init__(self, program: Program) -> None:
        """
        Construct a graphics object
        """
        self.program = program
        self.canvas = Canvas(program)

    def make_body(self) -> Statement:
        """
        Create the code for adding computations inside the loopnest
        """
        if self.__displayable():
            return self.canvas.add_activity()
        else:
            return cast(Statement, SBlock([]))

    def make_footer(self) -> Statement:
        """
        Create the loop footer for graphics
        """
        if self.__displayable():
            return self.canvas.display_canvas()
        else:
            return cast(Statement, SBlock([]))

    def make_header(self) -> Statement:
        """
        Create the loop header for graphics
        """
        if self.__displayable():
            return self.canvas.create_canvas()
        else:
            return cast(Statement, SBlock([]))

    def __displayable(self) -> bool:
        """
        Returns True if the program contains the information necessary to
        display the Einsum
        """
        spacetime = self.program.get_spacetime()
        return spacetime is not None
