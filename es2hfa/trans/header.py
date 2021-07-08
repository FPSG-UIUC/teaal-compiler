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

Translate the header above the HFA loop nest
"""

from typing import cast

from es2hfa.hfa.base import Expression, Statement
from es2hfa.hfa.expr import EFunc, EMethod
from es2hfa.hfa.stmt import SAssign, SBlock
from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor
from es2hfa.trans.canvas import Canvas
from es2hfa.trans.partitioning import Partitioner
from es2hfa.trans.utils import Utils


class Header:
    """
    Generate the HFA code for the header above the loop nest
    """

    @staticmethod
    def make_header(
            program: Program,
            canvas: Canvas,
            utils: Utils) -> Statement:
        """
        Create the header for a given einsum

        Expects the Einsum to have already been added to the program, but
        modifies the tensors in the program for the current einsum
        """
        header = SBlock([])

        # First, create the output tensor
        output = program.get_output()
        out_arg = Utils.build_rank_ids(output)
        out_constr = cast(Expression, EFunc("Tensor", [out_arg]))
        out_assn = SAssign(output.tensor_name(), out_constr)
        header.add(cast(Statement, out_assn))

        # Create a partitioner
        partitioner = Partitioner(program, utils)

        # Get the tensors we need to generate headers for
        tensors = program.get_tensors()

        # Generate the header for each tensor
        for tensor in tensors:
            # Partition if necessary
            header.add(partitioner.partition(tensor))

            # Swizzle for a concordant traversal
            old_name = tensor.tensor_name()
            program.apply_loop_order(tensor)
            new_name = tensor.tensor_name()

            # Emit code to perform the swizzle if necessary
            if old_name != new_name:
                header.add(Utils.build_swizzle(tensor, old_name))

            # Emit code to get the root fiber
            get_root_call = cast(Expression, EMethod(new_name, "getRoot", []))
            fiber_name = tensor.fiber_name()
            header.add(cast(Statement, SAssign(fiber_name, get_root_call)))

        # Generate canvas creation if needed
        if canvas.displayable():
            header.add(canvas.create_canvas())

        return cast(Statement, header)
