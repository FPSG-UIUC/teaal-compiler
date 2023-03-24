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
from copy import deepcopy

from sympy import Symbol
from typing import List, Optional

from teaal.hifiber import *
from teaal.ir.program import Program
from teaal.ir.tensor import Tensor
from teaal.trans.coord_access import CoordAccess


class Canvas:
    """
    Generate the HiFiber code for the tensor graphics
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
        args: List[Argument] = []
        for tensor in self.tensors:
            access = [self.__build_access(rank)
                      for rank in tensor.get_access()]
            args.append(AJust(ETuple(access)))

        # Get the space and time tuples
        space = self.get_space_tuple()

        # If slip, the timestamp is actually (timestamps[space] - 1,)
        time: Expression
        if spacetime.get_slip():
            bop = EBinOp(EAccess(EVar("timestamps"), space), OSub(), EInt(1))
            time = ETuple([bop])

        # Otherwise, it is just the time tuple
        else:
            time = self.get_time_tuple()

        # Add the space-time arguments
        args.append(AParam("spacetime", ETuple([space, time])))

        # Create call to addActivity
        add = EMethod(EVar("canvas"), "addActivity", args)

        # Create the corresponding statement
        return SExpr(add)

    def create_canvas(self) -> Statement:
        """
        Create a canvas
        """
        # Create a list of tensors in the order they should be displayed,
        # specifically, we want the output tensor to be at the end
        self.tensors = []
        for tensor in self.program.get_tensors():
            if tensor != self.program.get_equation().get_output():
                self.tensors.append(deepcopy(tensor))
        self.tensors.append(self.program.get_equation().get_output())

        # Build the args
        args = [AJust(EVar(tensor.tensor_name())) for tensor in self.tensors]

        # Build the call to createCanvas()
        create = EFunc("createCanvas", args)

        # Build the assignment
        return SAssign(AVar("canvas"), create)

    def display_canvas(self) -> Statement:
        """
        Display the canvas
        """
        if self.tensors is None:
            raise ValueError(
                "Unconfigured canvas. Make sure to first call create_canvas()")

        # Create call displayCanvas(canvas)
        return SExpr(EFunc("displayCanvas", [AJust(EVar("canvas"))]))

    def get_space_tuple(self) -> Expression:
        """
        Get the space stamp tuple for this mapping
        """
        spacetime = self.program.get_spacetime()
        if spacetime is None:
            raise ValueError("SpaceTime information unspecified")

        return ETuple([self.__rel_coord(rank)
                      for rank in spacetime.get_space()])

    def get_time_tuple(self) -> Expression:
        """
        Get the time stamp tuple for this mapping
        """
        spacetime = self.program.get_spacetime()
        if spacetime is None:
            raise ValueError("SpaceTime information unspecified")

        return ETuple([self.__rel_coord(rank)
                      for rank in spacetime.get_time()])

    def __build_access(self, rank: str) -> Expression:
        """
        Build an expression to describe the relevant rank
        """
        part_ir = self.program.get_partitioning()
        root = part_ir.get_root_name(rank.upper())
        suffix = rank[len(root):]

        # This is not the innermost rank
        if len(suffix) > 0 and suffix != "0" and suffix[-1] != "i":
            return EVar(rank)

        # If this rank is the result of flattening, then build the access
        # as a tuple of the constituent ranks
        if part_ir.is_flattened(rank.upper()):
            flat_ranks = self.program.get_loop_order().get_iter_ranks(rank.upper())
            return ETuple([EVar(frank.lower()) for frank in flat_ranks])

        # Otherwise, this is the innermost rank; so translate
        sexpr = self.program.get_coord_math().get_trans(root.lower())

        # Now, we need to replace the roots with their dynamic names
        for symbol in sexpr.atoms(Symbol):
            # Fix dynamic partitioning variable name
            new = part_ir.get_dyn_rank(str(symbol).upper()).lower()

            # Fix static partitioning variable name
            if (root,) in part_ir.get_static_parts():
                new += "0"

            sexpr = sexpr.subs(symbol, Symbol(new))

        return CoordAccess.build_expr(sexpr)

    def __rel_coord(self, rank: str) -> Expression:
        """
        Get the relative coordinate for this rank (important for PE distribution)
        """
        spacetime = self.program.get_spacetime()

        # Make sure we have a spacetime (needed to unwrap the Optional)
        if spacetime is None:
            raise ValueError("SpaceTime information unspecified")

        rank_str = rank.lower()
        if spacetime.get_style(rank) == "coord":
            # Check if we already have the absolute coordinate
            offset = spacetime.get_offset(rank)
            if offset is None:
                return EVar(rank_str)

            # rank - offset
            return EBinOp(EVar(rank_str), OSub(), EVar(offset.lower()))

        elif spacetime.get_style(rank) == "pos":
            # rank_pos
            return EVar(rank_str + "_pos")

        else:
            # Note: there is no good way to test this error. Bad spacetime styles
            # should be caught by the SpaceTime
            raise ValueError("Unknown spacetime style")  # pragma: no cover
