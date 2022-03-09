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

Representation of how tensors and variables are combined
"""

from typing import cast, Dict, List, Optional

from lark.lexer import Token
from lark.tree import Tree

from es2hfa.hfa import *
from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.utils import ParseUtils


class Equation:
    """
    A representation of how fibers should be combined, as well as the final
    equation at the bottom of the loop nest
    """

    def __init__(self, program: Program) -> None:
        """
        Construct a new Equation
        """
        self.program = program
        einsum = self.program.get_einsum()

        # First find all terms (terminals multiplied or otherwise intersected
        # together)
        self.terms: List[List[str]] = []
        self.vars: List[List[str]] = []

        self.factor_order: Dict[str, Tuple[int, int]] = {}
        self.in_update: List[List[bool]] = []
        for i, term in enumerate(einsum.find_data("times")):
            self.terms.append(cast(List[str], []))
            self.vars.append([])

            self.in_update.append(cast(List[bool], []))

            # Find all unfiltered factors
            for single in term.find_data("single"):
                for var in single.find_data("var"):
                    self.vars[-1].append(ParseUtils.next_str(var))
                    self.factor_order[self.vars[-1][-1]
                                      ] = (i, len(self.in_update[-1]))
                    self.in_update[-1].append(True)

                for tensor in single.find_data("tensor"):
                    self.terms[-1].append(ParseUtils.next_str(tensor))
                    self.factor_order[self.terms[-1][-1]
                                      ] = (i, len(self.in_update[-1]))
                    self.in_update[-1].append(True)

            # Find all factors intersected together
            in_update_off = len(self.in_update[-1])
            for dot in term.find_data("dot"):
                for child in dot.children:
                    if isinstance(child, Tree):
                        if child.data == "var":
                            self.vars[-1].append(ParseUtils.next_str(child))
                            self.factor_order[self.vars[-1][-1]
                                              ] = (i, len(self.in_update[-1]))
                            self.in_update[-1].append(False)

                        elif child.data == "tensor":
                            self.terms[-1].append(ParseUtils.next_str(child))
                            self.factor_order[self.terms[-1][-1]
                                              ] = (i, len(self.in_update[-1]))
                            self.in_update[-1].append(False)

                        else:
                            # Note: there is no way to test this error, bad
                            # factors should be caught by the parser
                            raise ValueError(
                                "Unknown factor")  # pragma: no cover

                    elif isinstance(child, Token):
                        self.in_update[-1][in_update_off +
                                           int(child.value)] = True

        # Now create the reverse dictionary of factors to term #
        self.term_dict: Dict[str, int] = {}
        for i, factors in enumerate(self.terms):
            for factor in factors:
                if factor in self.term_dict.keys():
                    raise ValueError(
                        factor + " appears multiple times in the einsum")
                self.term_dict[factor] = i

        # Finally, get the name of the output
        self.output = ParseUtils.find_str(einsum, "output")
        if self.output in self.term_dict.keys():
            raise ValueError(self.output +
                             " appears multiple times in the einsum")

    def make_iter_expr(self, rank: str, tensors: List[Tensor]) -> Expression:
        """
        Given a list of tensors, make the expression used to combine them
        """
        # Make sure that we have at least one tensor
        if not tensors:
            raise ValueError("Must iterate over at least one tensor")

        # Separate the tensors into terms
        terms = self.__separate_terms(tensors)
        output_tensor = self.__get_output_tensor(tensors)

        # If there are no input tensors, we just need to iterShape on the
        # output
        if not terms and output_tensor:
            call = EMethod(output_tensor.fiber_name(), "iterShape", [])
            return cast(Expression, call)

        # Combine terms with intersections
        intersections = []
        for term in terms:
            expr = cast(Expression, EVar(term[-1].fiber_name()))
            for factor in reversed(term[:-1]):
                expr = Equation.__add_operator(
                    cast(
                        Expression, EVar(
                            factor.fiber_name())), cast(
                        Operator, OAnd()), expr)
            intersections.append(expr)

        # Combine intersections with a union
        expr = intersections[-1]
        for intersection in reversed(intersections[:-1]):
            expr = Equation.__add_operator(
                intersection, cast(Operator, OOr()), expr)

        # Finally, add in the output
        if output_tensor:
            expr = Equation.__add_operator(
                cast(
                    Expression, EVar(
                        output_tensor.fiber_name())), cast(
                    Operator, OLtLt()), expr)

        # If the spacetime style is occupancy, we also need to enumerate the
        # iterations
        spacetime = self.program.get_spacetime()
        if spacetime is not None and spacetime.emit_pos(rank):
            arg = cast(Argument, AJust(expr))
            expr = cast(Expression, EFunc("enumerate", [arg]))

        return expr

    def make_payload(self, rank: str, tensors: List[Tensor]) -> Payload:
        """
        Given a list of tensors, construct the corresponding payload
        """
        # Make sure that we have at least one tensor
        if not tensors:
            raise ValueError(
                "Must have at least one tensor to make the payload")

        # Separate the tensors into terms
        terms = self.__separate_terms(tensors)
        output_tensor = self.__get_output_tensor(tensors)

        if terms:
            # Construct the term payloads
            term_payloads = []
            for term in terms:
                payload = cast(Payload, PVar(term[-1].fiber_name()))
                for factor in reversed(term[:-1]):
                    payload = Equation.__add_pvar(factor.fiber_name(), payload)
                term_payloads.append(payload)

            # Construct the entire expression payload
            payload = term_payloads[-1]
            for term_payload in reversed(term_payloads[:-1]):
                payload = cast(Payload, PTuple(
                    [cast(Payload, PVar("_")), term_payload, payload]))

            # Put the output on the outside
            if output_tensor:
                payload = Equation.__add_pvar(
                    output_tensor.fiber_name(), payload)

        elif output_tensor:
            payload = cast(Payload, PVar(output_tensor.fiber_name()))

        else:
            # We should never get to this state
            raise ValueError("Something is wrong...")  # pragma: no cover

        # Add the rank variable
        rank_var = rank.lower()
        payload = Equation.__add_pvar(rank_var, payload)

        # If the spacetime style is occupancy, we also need to enumerate the
        # iterations
        spacetime = self.program.get_spacetime()
        if spacetime is not None and spacetime.emit_pos(rank):
            payload = Equation.__add_pvar(rank_var + "_pos", payload)

        return payload

    def make_update(self) -> Statement:
        """
        Construct the statement that will actually update the output tensor
        """
        # Combine the factors within a term
        products = []
        for i, term in enumerate(self.terms):
            factors = [var for var in self.vars[i] if self.__in_update(var)] + \
                [tensor.lower() + "_val" for tensor in term
                    if self.__in_update(tensor)]
            product = cast(Expression, EVar(factors[0]))
            for factor in factors[1:]:
                product = cast(Expression, EBinOp(product, cast(
                    Operator, OMul()), cast(Expression, EVar(factor))))
            products.append(product)

        # Combine the terms
        sum_ = products[0]
        for product in products[1:]:
            sum_ = cast(
                Expression,
                EBinOp(
                    sum_,
                    cast(
                        Operator,
                        OAdd()),
                    product))

        # Create the final statement
        out_name = self.output.lower() + "_ref"
        out_var = cast(Assignable, AVar(out_name))
        return cast(Statement, SIAssign(out_var, cast(Operator, OAdd()), sum_))

    @staticmethod
    def __add_operator(
            expr1: Expression,
            op: Operator,
            expr2: Expression) -> Expression:
        """
        Combine two expressions with an operator
        """
        exprs = [expr1, expr2]
        for i, expr in enumerate(exprs):
            if isinstance(expr, EBinOp):
                exprs[i] = cast(Expression, EParens(expr))

        return cast(Expression, EBinOp(exprs[0], op, exprs[1]))

    @staticmethod
    def __add_pvar(var: str, payload: Payload) -> Payload:
        """
        Add a var to the front of a payload
        """
        pvar = cast(Payload, PVar(var))
        return cast(Payload, PTuple([pvar, payload]))

    def __get_output_tensor(self, tensors: List[Tensor]) -> Optional[Tensor]:
        """
        Get the output tensor if it exists
        """
        output_tensor = [
            tensor for tensor in tensors if tensor.root_name() == self.output]
        if output_tensor:
            return output_tensor[0]
        else:
            return None

    def __in_update(self, factor: str) -> bool:
        """
        Returns true if the factor should be included in the update
        """
        i, j = self.factor_order[factor]
        return self.in_update[i][j]

    def __separate_terms(self, tensors: List[Tensor]) -> List[List[Tensor]]:
        """
        Separate a list of tensors according to which term they belong to
        """
        # Separate the tensors
        terms: List[List[Tensor]] = [[] for _ in self.terms]
        for tensor in tensors:
            if tensor.get_is_output():
                continue

            terms[self.term_dict[tensor.root_name()]].append(tensor)

        # Remove any empty lists
        return [term for term in terms if term]
