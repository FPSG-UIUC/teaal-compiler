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

Translate the metrics collection
"""

from es2hfa.hfa import *
from es2hfa.ir.metrics import Metrics
from es2hfa.ir.program import Program


class Collector:
    """
    Translate the metrics collection
    """

    def __init__(self, program: Program, metrics: Metrics) -> None:
        """
        Construct a collector object
        """
        self.program = program
        self.metrics = metrics

    def dump(self) -> Statement:
        """
        Dump metrics information
        """
        # TODO
        return SBlock([])

    @staticmethod
    def end() -> Statement:
        """
        End metrics collection
        """
        return SExpr(EMethod(EVar("Metrics"), "endCollect", []))

    def set_collecting(self, tensor_name: str, rank: str) -> Statement:
        """
        Collect the statistics about a tensor
        """
        tensor = self.program.get_tensor(tensor_name)
        args = [AJust(EString(rank)), AJust(EBool(True))]
        call = EMethod(EVar(tensor.tensor_name()), "setCollecting", args)

        return SExpr(call)

    def start(self) -> Statement:
        """
        Start metrics collection
        """
        loop_order = self.program.get_loop_order()
        order = [EString(rank) for rank in loop_order.get_ranks()]
        call = EMethod(EVar("Metrics"), "beginCollect", [AJust(EList(order))])

        return SExpr(call)
