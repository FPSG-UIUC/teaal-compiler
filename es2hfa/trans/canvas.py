"""
Translate the canvas display information
"""
from typing import cast, List, Optional

from es2hfa.hfa.arg import AJust, AParam
from es2hfa.hfa.base import Argument, Expression, Statement
from es2hfa.hfa.expr import EFunc, EMethod, ETuple, EVar
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
        space = [EVar(ind[0].lower() + ind[1:]) for ind in display["space"]]
        space_tup = cast(Expression, ETuple(
            [cast(Expression, ind) for ind in space]))
        time = [EVar(ind[0].lower() + ind[1:]) for ind in display["time"]]
        time_tup = cast(Expression, ETuple(
            [cast(Expression, ind) for ind in time]))
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
        return display is not None and \
            "space" in display.keys() and \
            "time" in display.keys()
