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

Translate the canvas spacetime information
"""
from typing import cast, List, Optional

from es2hfa.hfa import *
from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor


class Canvas:
    """
    Generate the HFA code for the tensor graphics
    """

    def __init__(self, program: Program) -> None:
        """
        Construct a Canvas
        """
        self.program = program
        self.tensors: Optional[List[Tensor]] = None

    def add_activity(self) -> Statement:
        """
        Add activity to the canvas (i.e. show a computation)
        """
        if self.tensors is None:
            raise ValueError(
                "Unconfigured canvas. Make sure to first call create_canvas()")

        spacetime = self.program.get_spacetime()
        if spacetime is None:
            raise ValueError("SpaceTime information unspecified")

        # Create tensor rank arguments
        args = []
        for tensor in self.tensors:
            ranks = [self.program.get_partitioning().get_dyn_rank(rank)
                     for rank in tensor.get_access()]
            access = [cast(Expression, EVar(rank)) for rank in ranks]
            arg = AJust(cast(Expression, ETuple(access)))
            args.append(cast(Argument, arg))

        # Get the space and time tuples
        space = self.get_space_tuple()

        # If slip, the timestamp is actually (timestamps[space] - 1,)
        if spacetime.get_slip():
            acc = cast(Expression, EAccess("timestamps", space))
            sub = cast(Operator, OSub())
            one = cast(Expression, EInt(1))
            bop = cast(Expression, EBinOp(acc, sub, one))
            time = cast(Expression, ETuple([bop]))

        # Otherwise, it is just the time tuple
        else:
            time = self.get_time_tuple()

        # Add the space-time arguments
        spacetime_tup = cast(Expression, ETuple([space, time]))
        args.append(cast(Argument, AParam("spacetime", spacetime_tup)))

        # Create call to addActivity
        add = cast(Expression, EMethod("canvas", "addActivity", args))

        # Create the corresponding statement
        return cast(Statement, SExpr(add))

    def create_canvas(self) -> Statement:
        """
        Create a canvas
        """
        # Create a list of tensors in the order they should be displayed,
        # specifically, we want the output tensor to be at the end
        self.tensors = []
        for tensor in self.program.get_tensors():
            if tensor != self.program.get_output():
                self.tensors.append(tensor)
        self.tensors.append(self.program.get_output())

        # Build the args
        args = []
        for tensor in self.tensors:
            arg = AJust(cast(Expression, EVar(tensor.tensor_name())))
            args.append(cast(Argument, arg))

        # Build the call to createCanvas()
        create = cast(Expression, EFunc("createCanvas", args))

        # Build the assignment
        canvas_name = cast(Assignable, AVar("canvas"))
        return cast(Statement, SAssign(canvas_name, create))

    def display_canvas(self) -> Statement:
        """
        Display the canvas
        """
        if self.tensors is None:
            raise ValueError(
                "Unconfigured canvas. Make sure to first call create_canvas()")

        # Create call displayCanvas(canvas)
        canvas = cast(Argument, AJust(cast(Expression, EVar("canvas"))))
        call = cast(Expression, EFunc("displayCanvas", [canvas]))
        return cast(Statement, SExpr(call))

    def get_space_tuple(self) -> Expression:
        """
        Get the space stamp tuple for this mapping
        """
        spacetime = self.program.get_spacetime()
        if spacetime is None:
            raise ValueError("SpaceTime information unspecified")

        return cast(Expression, ETuple(
            [self.__rel_coord(rank) for rank in spacetime.get_space()]))

    def get_time_tuple(self) -> Expression:
        """
        Get the time stamp tuple for this mapping
        """
        spacetime = self.program.get_spacetime()
        if spacetime is None:
            raise ValueError("SpaceTime information unspecified")

        return cast(Expression, ETuple(
            [self.__rel_coord(rank) for rank in spacetime.get_time()]))

    def __rel_coord(self, rank: str) -> Expression:
        """
        Get the relative coordinate for this rank (important for PE distribution)
        """
        spacetime = self.program.get_spacetime()

        # Make sure we have a spacetime (needed to unwrap the Optional)
        if spacetime is None:
            raise ValueError("SpaceTime information unspecified")

        rank_str = rank[0].lower() + rank[1:]
        if spacetime.get_style(rank) == "coord":
            # Check if we already have the absolute coordinate
            offset = spacetime.get_offset(rank)
            if offset is None:
                return cast(Expression, EVar(rank_str))

            # rank - offset
            rank_expr = cast(Expression, EVar(rank_str))
            offset_expr = cast(Expression,
                               EVar(offset[0].lower() + offset[1:]))
            sub = EBinOp(rank_expr, cast(Operator, OSub()), offset_expr)

            return cast(Expression, sub)

        elif spacetime.get_style(rank) == "pos":
            # rank_pos
            return cast(Expression, EVar(rank_str + "_pos"))

        else:
            # Note: there is no good way to test this error. Bad spacetime styles
            # should be caught by the SpaceTime
            raise ValueError("Unknown spacetime style")  # pragma: no cover
