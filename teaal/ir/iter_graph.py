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

Representation of an iteration graph
"""

from typing import cast, List, Optional, Tuple

from teaal.ir.program import Program
from teaal.ir.tensor import Tensor


class IterationGraph:
    """
    A graph storing the tensor IR used to build the loop nests
    """

    def __init__(self, program: Program) -> None:
        """
        Construct a new IterationGraph
        """
        self.program = program

        # Track the current location in the iteration graph
        self.pos = 0

        # Configure the iteration graph
        loop_order = self.program.get_loop_order().get_ranks()
        self.loop_order = cast(List[Optional[str]], loop_order.copy()) + [None]

        # Get the set of ranks available at this point in the loop order
        rank_avail = []
        for rank in loop_order:
            rank_avail.append(
                self.program.get_partitioning().get_available(rank))

        # Get the set of ranks available up to (and including) this point in
        # the loop order
        self.avail = [set.union(*rank_avail[:i + 1])
                      for i in range(len(rank_avail))]

    def peek_concord(self) -> Tuple[Optional[str], List[Tensor]]:
        """
        Peek at the next loop iteration
        """
        tensors = []
        for tensor in self.program.get_tensors():
            if self.__ready(tensor):
                tensors.append(tensor)

        return self.loop_order[self.pos], tensors

    def peek_discord(self) -> List[Tuple[Tuple[str, ...], Tensor]]:
        """
        Peek at the tensors that need to be accessed discordantly
        """
        if self.pos < 1:
            raise ValueError(
                "Can only perform a discordant traversal inside the loop nest")

        tensors = []
        # For now, the only reason something would need to be accessed
        # discordantly is if it was not included in flattening
        for tensor in self.program.get_tensors():
            lower_rank = tensor.peek()
            if lower_rank is None:
                continue

            if lower_rank.upper() in self.avail[self.pos - 1]:
                tensors.append(tensor)

        print(self.pos, self.avail[self.pos - 1], tensors)

        info = []
        for tensor in tensors:
            ranks = []
            for rank in tensor.peek_rest():
                if rank in self.avail[self.pos - 1]:
                    ranks.append(rank)
                else:
                    break

            info.append((tuple(ranks), tensor))

        return info

    def pop_concord(self) -> Tuple[Optional[str], List[Tensor]]:
        """
        Pop the next loop iteration off the graph
        """
        rank, tensors = self.peek_concord()

        # Update each of the tensors
        for tensor in tensors:
            tensor.pop()

        # Update the position in the iteration graph
        self.pos += 1

        return rank, tensors

    def __ready(self, tensor: Tensor) -> bool:
        """
        Returns true if all coords are available to iterate over this tensor
        """
        rank = self.loop_order[self.pos]
        # If we are at the end, all tensors are involved
        if rank is None:
            return True

        # We already know rank is not None
        trank = tensor.peek()
        if trank is None:
            return False

        return self.program.get_loop_order().is_ready(trank.upper(), self.pos)
