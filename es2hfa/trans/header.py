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

from typing import cast, Set

from es2hfa.hfa import *
from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor
from es2hfa.trans.graphics import Graphics
from es2hfa.trans.partitioner import Partitioner
from es2hfa.trans.utils import TransUtils


class Header:
    """
    Generate the HFA code for loop headers
    """

    def __init__(self, program: Program, partitioner: Partitioner) -> None:
        """
        Construct a new Header object
        """
        self.program = program
        self.partitioner = partitioner

    # def make_dyn_part_header(self, ind: str) -> Statement:
    #     """
    #     Create the loop header for a dynamically partitioned dimension
    #     """


    def make_global_header(self, graphics: Graphics) -> Statement:
        """
        Create the header for a given einsum

        Expects the Einsum to have already been added to the program, but
        modifies the tensors in the program for the current einsum
        """
        header = SBlock([])

        # Configure the output tensor
        output = self.program.get_output()
        self.program.apply_all_partitioning(output)
        self.program.apply_final_loop_order(output)

        # Create the output tensor
        out_name = cast(Assignable, AVar(output.tensor_name()))
        out_arg = TransUtils.build_rank_ids(output)
        out_constr = cast(Expression, EFunc("Tensor", [out_arg]))
        out_assn = SAssign(out_name, out_constr)
        header.add(cast(Statement, out_assn))

        # Get the tensors we need to generate headers for
        tensors = self.program.get_tensors().copy()
        tensors.remove(output)

        # Prepare to partition all static dimensions
        partitioning = self.program.get_partitioning().get_static_parts()
        for ind in partitioning:
            self.program.start_partitioning(ind)

        # Modify each of the input tensors if necessary
        for tensor in tensors:
            header.add(self.__format_tensor(tensor, set(partitioning.keys())))

        # Emit code to get the root fiber
        for tensor in self.program.get_tensors():
            header.add(Header.__get_root(tensor))

        # Generate graphics if needed
        header.add(graphics.make_header())

        return cast(Statement, header)

    def __format_tensor(self, tensor: Tensor, inds: Set[str]) -> Statement:
        """
        Format the tensor according
        """
        header = SBlock([])

        # Partition if necessary
        header.add(self.partitioner.partition(tensor, inds))

        # Swizzle for a concordant traversal
        old_name = tensor.tensor_name()
        self.program.apply_curr_loop_order(tensor)
        new_name = tensor.tensor_name()

        # Emit code to perform the swizzle if necessary
        if old_name != new_name:
            header.add(TransUtils.build_swizzle(tensor, old_name))

        return cast(Statement, header)

    @staticmethod
    def __get_root(tensor: Tensor) -> Statement:
        """
        Get the root fiber for the given tensor
        """
        get_root_call = EMethod(tensor.tensor_name(), "getRoot", [])
        call_expr = cast(Expression, get_root_call)
        fiber_name = cast(Assignable, AVar(tensor.fiber_name()))
        return cast(Statement, SAssign(fiber_name, call_expr))
