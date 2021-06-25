"""
Representation of how tensors and variables are combined
"""

from lark.tree import Tree
from typing import cast, Dict, Generator, List, Optional

from es2hfa.hfa.base import Expression, Operator, Payload, Statement
from es2hfa.hfa.expr import EBinOp, EParens, EVar
from es2hfa.hfa.op import *
from es2hfa.hfa.payload import *
from es2hfa.hfa.stmt import SIAssign
from es2hfa.ir.tensor import Tensor


class Equation:
    """
    A representation of how fibers should be combined, as well as the final
    equation at the bottom of the loop nest
    """

    def __init__(self, einsum: Tree) -> None:
        """
        Construct a new Equation
        """
        # Make sure we are starting with the full einsum
        if einsum.data != "einsum":
            raise ValueError("Input parse tree must be an einsum")

        # First find all terms (terminals multiplied together)
        self.terms: List[List[str]] = []
        self.vars: List[List[str]] = []
        for term in einsum.find_data("times"):
            self.terms.append(cast(List[str], []))
            self.vars.append([])
            for var in term.find_data("var"):
                self.vars[-1].append(next(cast(Generator,
                                               var.scan_values(lambda _: True))))
            for tensor in term.find_data("tensor"):
                self.terms[-1].append(next(cast(Generator,
                                           tensor.scan_values(lambda _: True))))

        # Now create the reverse dictionary of factors to term #
        self.term_dict: Dict[str, int] = {}
        for i, factors in enumerate(self.terms):
            for factor in factors:
                if factor in self.term_dict.keys():
                    raise ValueError(
                        factor + " appears multiple times in the einsum")
                self.term_dict[factor] = i

        # Finally, get the name of the output
        self.output = next(
            cast(
                Generator, next(
                    einsum.find_data("output")).scan_values(
                    lambda _: True)))
        if self.output in self.term_dict.keys():
            raise ValueError(self.output +
                             " appears multiple times in the einsum")

    def make_iter_expr(self, tensors: List[Tensor]) -> Expression:
        """
        Given a list of tensors, make the expression used to combine them
        """
        # Make sure that we have at least one tensor
        if not tensors:
            raise ValueError("Must iterate over at least one tensor")

        # Separate the tensors into terms
        terms = self.__separate_terms(tensors)

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
        output_tensor = self.__get_output_tensor(tensors)
        if output_tensor:
            expr = Equation.__add_operator(
                cast(
                    Expression, EVar(
                        output_tensor.fiber_name())), cast(
                    Operator, OLtLt()), expr)

        return expr

    def make_payload(self, tensors: List[Tensor]) -> Payload:
        """
        Given a list of tensors, construct the corresponding payload
        """
        # Make sure that we have at least one tensor
        if not tensors:
            raise ValueError(
                "Must have at least one tensor to make the payload")

        # Separate the tensors into terms
        terms = self.__separate_terms(tensors)

        # Construct the term payloads
        term_payloads = []
        for term in terms:
            payload = cast(Payload, PVar(term[-1].fiber_name()))
            for factor in reversed(term[:-1]):
                payload = cast(Payload, PTuple(
                    [cast(Payload, PVar(factor.fiber_name())), payload]))
            term_payloads.append(payload)

        # Construct the entire expression payload
        payload = term_payloads[-1]
        for term_payload in reversed(term_payloads[:-1]):
            payload = cast(Payload, PTuple(
                [cast(Payload, PVar("_")), term_payload, payload]))

        # Finally, put the output on the outside
        output_tensor = self.__get_output_tensor(tensors)
        if output_tensor:
            payload = cast(Payload, PTuple(
                [cast(Payload, PVar(output_tensor.fiber_name())), payload]))

        return payload

    def make_update(self) -> Statement:
        """
        Construct the statement that will actually update the output tensor
        """
        # Combine the factors within a term
        products = []
        for i, term in enumerate(self.terms):
            factors = [var for var in self.vars[i]] + \
                [tensor[0].lower() + tensor[1:] + "_val" for tensor in term]
            product = cast(Expression, EVar(factors[0]))
            for factor in factors[1:]:
                product = cast(
                    Expression, EBinOp(
                        product, cast(
                            Operator, OMul()), cast(
                            Expression, EVar(factor))))
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
        return cast(Statement, SIAssign(self.output[0].lower(
        ) + self.output[1:] + "_ref", cast(Operator, OAdd()), sum_))

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

    def __separate_terms(self, tensors: List[Tensor]) -> List[List[Tensor]]:
        """
        Separate a list of tensors according to which term they belong to
        """
        # Separate the tensors
        terms: List[List[Tensor]] = [[] for _ in self.terms]
        for tensor in tensors:
            if tensor.root_name() in self.term_dict.keys():
                terms[self.term_dict[tensor.root_name()]].append(tensor)

        # Remove any empty lists
        return [term for term in terms if term]
