"""
Representation of how payloads are combined
"""

from typing import cast, Dict, Generator, List, Optional

from lark.tree import Tree

from es2hfa.hfa.base import Expression, Operator, Payload
from es2hfa.hfa.expr import EBinOp, EParens, EVar
from es2hfa.hfa.op import OAnd, OLtLt, OOr
from es2hfa.hfa.payload import *
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
        terms: List[List[str]] = []
        for term in einsum.find_data("times"):
            terms.append(cast(List[str], []))
            for var in term.find_data("var"):
                terms[-1].append(next(cast(Generator,
                                 var.scan_values(lambda _: True))))
            for tensor in term.find_data("tensor"):
                terms[-1].append(next(cast(Generator,
                                           tensor.scan_values(lambda _: True))))
        self.num_terms = len(terms)

        # Now create the reverse dictionary of factors to term #
        self.terms: Dict[str, int] = {}
        for i, factors in enumerate(terms):
            for factor in factors:
                if factor in self.terms.keys():
                    raise ValueError(
                        factor + " appears multiple times in the einsum")
                self.terms[factor] = i

        # Finally, get the name of the output
        self.output = next(
            cast(
                Generator, next(
                    einsum.find_data("output")).scan_values(
                    lambda _: True)))
        if self.output in self.terms.keys():
            raise ValueError(self.output +
                             " appears multiple times in the einsum")

    def make_payload(self, tensors: List[Tensor]) -> Payload:
        """
        Given a list of tensors, construct the corresponding payload
        """
        # Make sure that we have at least one tensor
        if not tensors:
            raise ValueError(
                "Must have at least one tensor to make the payload")

        # Get the tensors that will be used in this payload
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

    def make_iter_expr(self, tensors: List[Tensor]) -> Expression:
        """
        Given a list of tensors, make the expression used to combine them
        """
        # Make sure that we have at least one tensor
        if not tensors:
            raise ValueError("Must iterate over at least one tensor")

        # Get the tensors that will be used in this payload
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

    def __separate_terms(self, tensors: List[Tensor]) -> List[List[Tensor]]:
        """
        Separate a list of tensors according to which term they belong to
        """
        # Separate the tensors
        terms: List[List[Tensor]] = [[] for _ in range(self.num_terms)]
        for tensor in tensors:
            if tensor.root_name() in self.terms.keys():
                terms[self.terms[tensor.root_name()]].append(tensor)

        # Remove any empty lists
        return [term for term in terms if term]

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
