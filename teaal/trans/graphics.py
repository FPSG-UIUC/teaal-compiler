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

from teaal.hfa import *
from teaal.ir.program import Program
from teaal.ir.spacetime import SpaceTime
from teaal.trans.canvas import Canvas


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
        body = SBlock([])
        spacetime = self.program.get_spacetime()

        if spacetime is not None:
            # If we are using slip, increment the timestamp
            if spacetime.get_slip():

                # If this is the first time we are seeing the space stamp
                keys = EMethod(EVar("timestamps"), "keys", [])
                space_tup = self.canvas.get_space_tuple()
                cond = EBinOp(space_tup, OIn(), keys)

                # Then add 1
                acc = AAccess(EVar("timestamps"), space_tup)
                then = SIAssign(acc, OAdd(), EInt(1))

                # Otherwise add it to the dictionary
                else_ = SAssign(acc, EInt(1))

                if_ = SIf((cond, then), [], else_)
                body.add(if_)

            body.add(self.canvas.add_activity())

        return body

    def make_footer(self) -> Statement:
        """
        Create the loop footer for graphics
        """
        spacetime = self.program.get_spacetime()
        if spacetime is not None:
            return self.canvas.display_canvas()
        else:
            return SBlock([])

    def make_header(self) -> Statement:
        """
        Create the loop header for graphics
        """
        header = SBlock([])

        # If displayable, add the graphics information
        spacetime = self.program.get_spacetime()
        if spacetime is not None:
            header.add(self.canvas.create_canvas())

            # Create the timestamp dictionary if we want slip
            if spacetime.get_slip():
                assign = SAssign(AVar("timestamps"), EDict({}))
                header.add(assign)

        return header
