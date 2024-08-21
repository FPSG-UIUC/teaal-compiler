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

Representation of the tensor coordinates and their relationships
"""

from functools import reduce

from lark.lexer import Token
from lark.tree import Tree
from sympy import Basic, solve, Symbol  # type: ignore
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union

from teaal.ir.tensor import Tensor
from teaal.parse.utils import ParseUtils


class CoordMath:
    """
    Store the relationships between tensor coordinates
    """

    def __init__(self) -> None:
        """
        Construct the metadata for coord math
        """
        self.all_exprs: Dict[Symbol, List[Basic]] = {}
        self.eqn_exprs: Dict[Symbol, Basic] = {}
        self.trans: Optional[Dict[Symbol, Basic]] = None

    def add(self, tensor: Tensor, ranks: Tree) -> None:
        """
        Add the information for a given set of ranks
        """
        for rank, expr in zip(tensor.get_ranks(), ranks.children):
            if not isinstance(expr, Tree):
                raise ValueError("Unknown coord tree: " + repr(expr))

            terms = []
            for term in expr.children:

                if not isinstance(term, Tree):
                    raise ValueError("Unknown coord term: " + repr(term))

                # Extract a term of the form "x"
                if term.data == "ijust":
                    terms.append(Symbol(ParseUtils.next_str(term)))

                # Extract a term of the form "2 * y"
                elif term.data == "itimes":

                    tokens = []
                    for token in term.children:
                        if not isinstance(token, Token):
                            raise ValueError(
                                "Unknown coord factor: " + repr(token))

                        tokens.append(token)

                    if len(tokens) != 2:
                        raise ValueError("Unknown coord term: " + repr(term))

                    terms.append(int(tokens[0]) * Symbol(tokens[1]))

                else:
                    raise ValueError("Unknown coord term: " + term.data)

            # The relationship between the symbols given by this relation
            # (when set to 0)
            init_ind = Symbol(rank.lower())
            eqn_expr = reduce(lambda x, y: x + y, terms)
            self.eqn_exprs[init_ind] = eqn_expr

            full_expr = eqn_expr - init_ind
            symbols = full_expr.atoms(Symbol)

            # All coordinates map to themselves
            for ind in symbols | {init_ind}:
                if ind not in self.all_exprs.keys():
                    self.all_exprs[ind] = [ind]

            for ind in symbols:
                self.all_exprs[ind] += solve(full_expr, ind)

    def get_all_exprs(self, ind: str) -> List[Basic]:
        """
        Get expressions corresponding to the different ways to represent a
        a given coordinate
        """
        sym = Symbol(ind)
        if sym in self.all_exprs:
            return self.all_exprs[sym]
        # If the symbol is not here (e.g. because of flattening) there is no
        # translation
        return [sym]

    def get_cond_expr(self, ind: str, cond: Callable[[Basic], bool]) -> Basic:
        """
        Get an expression to translate the given index variable, provided it
        meets the condition

        Note: Exactly one expression must meet this condition
        """
        exprs = {
            expr for expr in self.get_all_exprs(
                ind.lower()) if cond(expr)}

        if not exprs:
            raise ValueError(
                "No matching expression for index variable " + ind)
        if len(exprs) > 1:
            raise ValueError(
                "Multiple expressions match for index variable " +
                ind +
                ": " +
                str(exprs))

        return next(iter(exprs))

    def get_trans(self, ind: Union[str, Symbol]) -> Basic:
        """
        Get the expression corresponding to the coord with the current loop order
        """
        if self.trans is None:
            raise ValueError("Unconfigured coord math. First call prune()")

        if isinstance(ind, str):
            ind = Symbol(ind)

        return self.trans[ind]

    def prune(self, avail_roots: Set[str]) -> None:
        """
        Prune out all coord translations not available with this loop order
        """
        self.trans = {}

        avail = set(Symbol(root.lower()) for root in avail_roots)

        # Prune unnecessary translations
        for ind, exprs in self.all_exprs.items():
            for expr in exprs:
                if not (expr.atoms(Symbol) - avail):
                    self.trans[ind] = expr

    def __key(self) -> Iterable[Any]:
        """
        Get all properties of the CoordMath
        """
        return self.all_exprs, self.trans

    def __eq__(self, other) -> bool:
        """
        Test CoordMath equality
        """
        if not isinstance(other, CoordMath):
            return False

        return self.__key() == other.__key()
