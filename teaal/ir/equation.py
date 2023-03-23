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
from itertools import chain

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
        self.__compute_einsum_ranks()
        self.__compute_tensors_trees()

    def get_output(self) -> Tensor:
        """
        Get the output tensor
        """
        return self.es_tensors[0]

    def get_tensors(self) -> List[Tensor]:
        """
        Get the list of tensors
        """
        return self.es_tensors

    def get_trees(self) -> List[Tree]:
        """
        Get the parse trees corresponding to the list of tensors
        """
        return self.es_trees

    def get_einsum_ranks(self) -> List[str]:
        """
        Get the ranks used in the Einsum
        """
        return self.einsum_ranks

    def __compute_einsum_ranks(self) -> None:
        """
        Compute the ranks used by the Einsum

        Note: returns the output ranks first
        """
        term_iter = chain(
            self.equation.find_data("times"),
            self.equation.find_data("take"))

        # Get the ranks in a term of inputs
        term_ranks = Equation.__get_term_ranks(next(term_iter))

        # Do some error checking
        for term in term_iter:
            check_ranks = Equation.__get_term_ranks(term)

            if check_ranks != term_ranks:
                raise ValueError(
                    "Malformed einsum: ensure all terms iterate over all ranks")

        # Find the ranks of the output
        output_ranks = next(
            next(self.equation.find_data("output")).find_data("ranks"))
        self.einsum_ranks = Equation.__get_tensor_ranks(output_ranks)
        for rank in term_ranks:
            if rank not in self.einsum_ranks:
                self.einsum_ranks.append(rank)

    def __compute_tensors_trees(self) -> None:
        """
        Find the tensors used in this Einsum and their corresponding parse trees
        """
        self.es_trees = []
        self.es_tensors = []

        output_tree = next(self.equation.find_data("output"))
        self.es_trees.append(output_tree)

        output = self.__get_tensor(output_tree)
        output.set_is_output(True)
        self.es_tensors.append(output)

        for tensor_tree in self.equation.find_data("tensor"):
            self.es_trees.append(tensor_tree)
            self.es_tensors.append(self.__get_tensor(tensor_tree))

    def __get_tensor(self, tensor: Tree) -> Tensor:
        """
        Given a parse tree, get the appropriate tensor
        """
        name = ParseUtils.next_str(tensor)
        if name not in self.tensors.keys():
            raise ValueError("Undeclared tensor: " + name)

        return self.tensors[name]

    @staticmethod
    def __get_tensor_ranks(ranks: Tree) -> List[str]:
        """
        Return a list of ranks in the tensor
        """
        str_ranks = []
        for ijust in ranks.find_data("ijust"):
            rank = ParseUtils.next_str(ijust).upper()
            str_ranks.append(rank)

        for itimes in ranks.find_data("itimes"):
            rank = str(itimes.children[1]).upper()
            str_ranks.append(rank)

        return str_ranks

    @staticmethod
    def __get_term_ranks(term: Tree) -> List[str]:
        """
        Get the ranks in a term
        """
        term_ranks = list()
        for ranks in term.find_data("ranks"):
            for rank in Equation.__get_tensor_ranks(ranks):
                if rank not in term_ranks:
                    term_ranks.append(rank)

        return term_ranks

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
