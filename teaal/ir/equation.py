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

Representation of the Einsum equation
"""
from lark.tree import Tree
from typing import Any, Dict, Iterable, List, Tuple

from teaal.ir.tensor import Tensor
from teaal.parse.utils import ParseUtils


class Equation:
    """
    Representation of the Einsum equation
    """

    def __init__(self, equation: Tree, tensors: Dict[str, Tensor]) -> None:
        """
        Construct a new  representation
        """
        self.equation = equation
        self.tensors = tensors

    def get_output(self) -> Tuple[Tensor, Tree]:
        """
        Construct the output tensor
        """
        output_tree = next(self.equation.find_data("output"))
        output = self.__get_tensor(output_tree)
        output.set_is_output(True)
        return output, output_tree

    def get_tensors(self) -> List[Tuple[Tensor, Tree]]:
        """
        Construct a list of input tensors
        """
        tensors_trees: List[Tuple[Tensor, Tree]] = []
        for tensor_tree in self.equation.find_data("tensor"):
            tensors_trees.append((self.__get_tensor(tensor_tree), tensor_tree))

        return tensors_trees

    def __get_tensor(self, tensor: Tree) -> Tensor:
        """
        Given a parse tree, get the appropriate tensor
        """
        name = ParseUtils.next_str(tensor)
        if name not in self.tensors.keys():
            raise ValueError("Undeclared tensor: " + name)

        return self.tensors[name]

    def __eq__(self, other: object) -> bool:
        """
        The == operator for Equations
        """
        if isinstance(other, type(self)):
            for field1, field2 in zip(self.__key(), other.__key()):
                if field1 != field2:
                    return False
            return True
        return False

    def __key(self) -> Iterable[Any]:
        """
        Get the fields of the Equation
        """
        return self.equation, self.tensors
