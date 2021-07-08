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

Represetation of an iteration graph
"""

from typing import cast, List, Optional, Tuple

from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor


class IterationGraph:
    """
    A graph storing the tensor IR used to build the loop nests
    """

    def __init__(self, program: Program) -> None:
        """
        Construct a new IterationGraph
        """
        # Add a None to the end of the loop_order for the bottom of the loop
        # nest
        self.loop_order = cast(List[Optional[str]],
                               program.get_loop_order().copy()) + [None]

        # Place the tensors in the appropriate locations in the iteration graph
        self.graph: List[List[Tensor]] = []
        for ind in self.loop_order:
            self.graph.append([])
            for tensor in program.get_tensors():
                if tensor.peek() == ind:
                    self.graph[-1].append(tensor)

    def peek(self) -> Tuple[Optional[str], List[Tensor]]:
        """
        Peek at the next iteration
        """
        ind = self.loop_order[0]
        return ind, self.graph[0]

    def pop(self) -> Tuple[Optional[str], List[Tensor]]:
        """
        Pop the next iteration off the graph
        """
        # Pop off the next iteration
        ind = self.loop_order.pop(0)
        tensors = self.graph.pop(0)

        # Update each of the tensors and re-insert them into the graph at the
        # appropriate location
        for tensor in tensors:
            tensor.pop()
            self.graph[self.loop_order.index(tensor.peek())].append(tensor)

        return ind, tensors
