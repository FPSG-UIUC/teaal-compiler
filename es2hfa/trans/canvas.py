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

Translate the canvas display information
"""
from typing import cast, List, Optional

from es2hfa.hfa.arg import AJust, AParam
from es2hfa.hfa.base import Argument, Expression, Operator, Statement
from es2hfa.hfa.expr import EBinOp, EFunc, EMethod, ETuple, EVar
from es2hfa.hfa.op import OSub
from es2hfa.hfa.stmt import SAssign, SExpr
from es2hfa.ir.mapping import Mapping
from es2hfa.ir.tensor import Tensor


class Canvas:
    """
    Generate the HFA code for the tensor graphics
    """

    def __init__(self, mapping: Mapping) -> None:
        """
        Construct a Canvas
        """
        self.mapping = mapping
        self.tensors: Optional[List[Tensor]] = None

    def add_activity(self) -> Statement:
        """
        Add activity to the canvas (i.e. show a computation)
        """
        if self.tensors is None:
            raise ValueError(
                "Unconfigured canvas. Make sure to first call create_canvas()")

        display = self.mapping.get_display()
        if display is None:
            raise ValueError("Display information unspecified")

        # Create tensor index arguments
        args = []
        for tensor in self.tensors:
            access = [cast(Expression, EVar(ind))
                      for ind in tensor.get_access()]
            arg = AJust(cast(Expression, ETuple(access)))
            args.append(cast(Argument, arg))

        # Add the space-time arguments
        space_tup = cast(Expression, ETuple(
            [self.__rel_coord(ind) for ind in display.get_space()]))
        time_tup = cast(Expression, ETuple(
            [self.__rel_coord(ind) for ind in display.get_time()]))
        spacetime = cast(Expression, ETuple([space_tup, time_tup]))
        args.append(cast(Argument, AParam("spacetime", spacetime)))

        # Create call to add activity
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
        for tensor in self.mapping.get_tensors():
            if tensor != self.mapping.get_output():
                self.tensors.append(tensor)
        self.tensors.append(self.mapping.get_output())

        # Build the args
        args = []
        for tensor in self.tensors:
            arg = AJust(cast(Expression, EVar(tensor.tensor_name())))
            args.append(cast(Argument, arg))

        # Build the call to createCanvas()
        create = cast(Expression, EFunc("createCanvas", args))

        # Build the assignment
        return cast(Statement, SAssign("canvas", create))

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

    def displayable(self) -> bool:
        """
        Returns True if the mapping contains the information necessary to
        display the Einsum
        """
        display = self.mapping.get_display()
        return display is not None

    def __rel_coord(self, ind: str) -> Expression:
        """
        Get the relative coordinate for this index (important for PE distribution)
        """
        display = self.mapping.get_display()

        # Make sure we have a display (needed to unwrap the Optional)
        if display is None:
            raise ValueError("Display information unspecified")

        # Check if we already have the absolute coordinate
        base = display.get_base(ind)
        ind_str = ind[0].lower() + ind[1:]
        if base is None:
            return cast(Expression, EVar(ind_str))

        if display.get_style() == "shape":
            # ind - base
            ind_expr = cast(Expression, EVar(ind_str))
            base_expr = cast(Expression, EVar(base[0].lower() + base[1:]))
            sub = EBinOp(ind_expr, cast(Operator, OSub()), base_expr)

            return cast(Expression, sub)

        elif display.get_style() == "occupancy":
            # ind_pos
            return cast(Expression, EVar(ind_str + "_pos"))

        else:
            # Note: there is no good way to test this error. Bad display styles
            # should be caught by the Display
            raise ValueError("Unknown display style")  # pragma: no cover
