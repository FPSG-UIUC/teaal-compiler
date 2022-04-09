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

Representation of the tensor indices and their relationships
"""

from functools import reduce

from lark.lexer import Token
from lark.tree import Tree
from sympy import Symbol
from sympy.core.expr import Expr
from sympy.solvers import solve
from typing import Any, Dict, Iterable, List, Optional

from es2hfa.ir.partitioning import Partitioning
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.utils import ParseUtils


class IndexMath:
    """
    Store the relationships between tensor indices
    """

    def __init__(self) -> None:
        """
        Construct the metadata for index math
        """
        self.all_exprs: Dict[Symbol, List[Expr]] = {}
        self.trans: Optional[Dict[Symbol, Expr]] = None

    def add(self, tensor: Tensor, tranks: Tree) -> None:
        """
        Add the information for a given set of ranks
        """
        for rank, expr in zip(tensor.get_ranks(), tranks.children):
            if not isinstance(expr, Tree):
                raise ValueError("Unknown index tree: " + repr(expr))

            terms = []
            for term in expr.children:

                if not isinstance(term, Tree):
                    raise ValueError("Unknown index term: " + repr(term))

                # Extract a term of the form "x"
                if term.data == "ijust":
                    terms.append(Symbol(ParseUtils.next_str(term)))

                # Extract a term of the form "2 * y"
                elif term.data == "itimes":

                    tokens = []
                    for token in term.children:
                        if not isinstance(token, Token):
                            raise ValueError(
                                "Unknown index factor: " + repr(token))

                        tokens.append(token)

                    if len(tokens) != 2:
                        raise ValueError("Unknown index term: " + repr(term))

                    terms.append(int(tokens[0]) * Symbol(tokens[1]))

                else:
                    raise ValueError("Unknown index term: " + term.data)

            # The relationship between the symbols given by this relation
            # (when set to 0)
            init_ind = Symbol(rank.lower())
            full_expr = reduce(lambda x, y: x + y, terms) - init_ind

            symbols = full_expr.atoms(Symbol)

            # All indices map to themselves
            for ind in symbols | {init_ind}:
                if ind not in self.all_exprs.keys():
                    self.all_exprs[ind] = [ind]

            for ind in symbols:
                self.all_exprs[ind] += solve(full_expr, ind)

    def get_all_exprs(self, ind: str) -> List[Expr]:
        """
        Get expressions corresponding to the different ways to represent a
        a given index

        TODO: currently only used for testing
        """
        return self.all_exprs[Symbol(ind)]

    def get_trans(self, ind: str) -> Expr:
        """
        Get the expression corresponding to the index with the current loop order
        """
        if self.trans is None:
            raise ValueError("Unconfigured index math. First call prune()")

        return self.trans[Symbol(ind)]

    def prune(self, loop_order: List[str], partitioning: Partitioning) -> None:
        """
        Prune out all index translations not available with this loop order
        """
        self.trans = {}

        # Build the set of symbols available
        def trans_name(r): return partitioning.get_root_name(r).lower()
        avail = {Symbol(trans_name(rank)) for rank in loop_order}

        # Prune unnecessary translations
        for ind, exprs in self.all_exprs.items():
            for expr in exprs:
                if len(expr.atoms(Symbol) - avail) == 0:
                    self.trans[ind] = expr

    def __key(self) -> Iterable[Any]:
        """
        Get all properties of the IndexMath
        """
        return self.all_exprs, self.trans

    def __eq__(self, other) -> bool:
        """
        Test IndexMath equality
        """
        if not isinstance(other, IndexMath):
            return False

        return self.__key() == other.__key()
