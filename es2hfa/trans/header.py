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

from es2hfa.hfa import *
from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor
from es2hfa.trans.graphics import Graphics
from es2hfa.trans.partitioner import Partitioner
from es2hfa.trans.utils import TransUtils


class Header:
    """
    Generate the HFA code for the header above the loop nest
    """

    @staticmethod
    def make_header(
            program: Program,
            graphics: Graphics,
            partitioner: Partitioner) -> Statement:
        """
        Create the header for a given einsum

        Expects the Einsum to have already been added to the program, but
        modifies the tensors in the program for the current einsum
        """
        header = SBlock([])

        # Configure the output tensor
        output = program.get_output()
        # program.apply_all_partitioning(output)
        # program.apply_final_loop_order

        # Create the output tensor
        out_name = cast(Assignable, AVar(output.tensor_name()))
        out_arg = TransUtils.build_rank_ids(output)
        out_constr = cast(Expression, EFunc("Tensor", [out_arg]))
        out_assn = SAssign(out_name, out_constr)
        header.add(cast(Statement, out_assn))

        # Get the tensors we need to generate headers for
        tensors = program.get_tensors()

        # Prepare to partition all static dimensions
        for ind in program.get_partitioning().get_static_parts():
            program.start_partitioning(ind)

        # Generate the header for each tensor
        for tensor in tensors:
            # Partition if necessary
            header.add(partitioner.partition(tensor))

            # Swizzle for a concordant traversal
            old_name = tensor.tensor_name()
            program.apply_curr_loop_order(tensor)
            new_name = tensor.tensor_name()

            # Emit code to perform the swizzle if necessary
            if old_name != new_name:
                header.add(TransUtils.build_swizzle(tensor, old_name))

            # Emit code to get the root fiber
            get_root_call = cast(Expression, EMethod(new_name, "getRoot", []))
            fiber_name = cast(Assignable, AVar(tensor.fiber_name()))
            header.add(cast(Statement, SAssign(fiber_name, get_root_call)))

        # Generate graphics if needed
        header.add(graphics.make_header())

        return cast(Statement, header)
