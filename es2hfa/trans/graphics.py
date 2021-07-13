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

from es2hfa.hfa.base import Expression, Operator, Statement
from es2hfa.hfa.expr import EBinOp, EDict, EInt, EMethod
from es2hfa.hfa.op import OAdd, OIn
from es2hfa.hfa.stmt import SAssign, SBlock, SIAssign, SIf
from es2hfa.ir.program import Program
from es2hfa.ir.spacetime import SpaceTime
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
        body = SBlock([])
        spacetime = self.program.get_spacetime()

        if spacetime is not None:
            # If we are using slip, increment the timestamp
            if spacetime.get_slip():

                # If this is the first time we are seeing the space stamp
                keys = cast(Expression, EMethod("timestamps", "keys", []))
                in_ = cast(Operator, OIn())
                space_tup = self.canvas.get_space_tuple()
                cond = cast(Expression, EBinOp(space_tup, in_, keys))

                # Then add 1
                # TODO: This is a hack!!!!!
                acc = "timestamps[" + space_tup.gen() + "]"
                add = cast(Operator, OAdd())
                one = cast(Expression, EInt(1))
                then = cast(Statement, SIAssign(acc, add, one))

                # Otherwise add it to the dictionary
                else_ = cast(Statement, SAssign(acc, one))

                if_ = cast(Statement, SIf((cond, then), [], else_))
                body.add(if_)

            body.add(self.canvas.add_activity())

        return cast(Statement, body)

    def make_footer(self) -> Statement:
        """
        Create the loop footer for graphics
        """
        spacetime = self.program.get_spacetime()
        if spacetime is not None:
            return self.canvas.display_canvas()
        else:
            return cast(Statement, SBlock([]))

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
                dict_ = cast(Expression, EDict({}))
                assign = cast(Statement, SAssign("timestamps", dict_))
                header.add(assign)

        return cast(Statement, header)
