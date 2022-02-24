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
from es2hfa.parse.utils import ParseUtils
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

    def make_output(self) -> Statement:
        """
        Given an output tensor, generate the constructor
        """
        tensor = self.program.get_output()
        self.program.apply_all_partitioning(tensor)
        self.program.get_loop_order().apply(tensor)

        name = cast(Assignable, AVar(tensor.tensor_name()))
        arg = TransUtils.build_rank_ids(tensor)
        constr = cast(Expression, EFunc("Tensor", [arg]))
        return cast(Statement, SAssign(name, constr))

    def make_swizzle_root(self, tensor: Tensor) -> Statement:
        """
        Make calls to swizzleRanks() (as necessary) and getRoot()
        """
        block = SBlock([])

        # Swizzle for a concordant traversal
        old_name = tensor.tensor_name()
        self.program.get_loop_order().apply(tensor)
        new_name = tensor.tensor_name()

        # Emit code to perform the swizzle if necessary
        if old_name != new_name:
            block.add(TransUtils.build_swizzle(tensor, old_name))

        # Add the call to getRoot()
        get_root_call = EMethod(tensor.tensor_name(), "getRoot", [])
        call_expr = cast(Expression, get_root_call)
        fiber_name = cast(Assignable, AVar(tensor.fiber_name()))
        block.add(cast(Statement, SAssign(fiber_name, call_expr)))

        return cast(Statement, block)

    @staticmethod
    def make_tensor_from_fiber(tensor: Tensor) -> Statement:
        """
        Get a tensor from the current fiber
        """
        tensor.from_fiber()

        ranks = [cast(Expression, EString(rank))
                 for rank in tensor.get_ranks()]
        ranks_list = cast(Expression, EList(ranks))
        fiber_name = cast(Expression, EVar(tensor.fiber_name()))
        args = [cast(Argument, AParam("rank_ids", ranks_list)),
                cast(Argument, AParam("fiber", fiber_name))]

        from_fiber = EMethod("Tensor", "fromFiber", args)
        from_fiber_expr = cast(Expression, from_fiber)

        tensor_name = cast(Assignable, AVar(tensor.tensor_name()))
        return cast(Statement, SAssign(tensor_name, from_fiber_expr))
