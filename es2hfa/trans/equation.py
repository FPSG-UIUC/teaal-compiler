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

Translation for how tensors and variables are combined
"""

from lark.lexer import Token
from lark.tree import Tree
from sympy import Add, Basic, Integer, Mul, Rational, solve, Symbol
from typing import cast, Dict, List, Optional, Type

from es2hfa.hfa import *
from es2hfa.ir.program import Program
from es2hfa.ir.tensor import Tensor
from es2hfa.parse.utils import ParseUtils
from es2hfa.trans.coord_access import CoordAccess


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

    def make_eager_inputs(self, rank: str, inputs: List[str]) -> Statement:
        """
        Given a rank to make eager inputs out of and a list of tensors, combine them
        """
        tensors = [self.program.get_tensor(input_) for input_ in inputs]
        iter_expr = self.__make_input_iter_expr(rank, tensors)

        # If there is only one fiber, its already eager
        if len(inputs) == 1:
            return SAssign(AVar("inputs_" + rank.lower()), iter_expr)

        # Otherwise, use Fiber.fromLazy()
        method_call = EMethod(EVar("Fiber"), "fromLazy", [AJust(iter_expr)])
        return SAssign(AVar("inputs_" + rank.lower()), method_call)

    def make_interval(self, rank: str) -> Statement:
        """
        Make the interval to project over: [rank_start, rank_end)
        """
        root = self.program.get_partitioning().get_root_name(rank)

        # This should be the bottom of the partition
        if rank[len(root):] != "0":
            raise ValueError("Interval not necessary for rank " + rank)

        rank = rank.lower()
        root = root.lower()

        # Set rank_start
        pos = EVar(root + "1_pos")
        start_cond = EBinOp(pos, OEqEq(), EInt(0))
        start_then = SAssign(AVar(rank + "_start"), EInt(0))
        start_else = SAssign(AVar(rank + "_start"), EVar(root + "1"))
        start_if = SIf((start_cond, start_then), [], start_else)

        # Set rank_end
        # root1_pos + 1 < len(inputs_root1)
        inputs = EVar("inputs_" + root + "1")
        len_call = EFunc("len", [AJust(inputs)])
        pos_plus_1 = EBinOp(pos, OAdd(), EInt(1))
        end_cond = EBinOp(pos_plus_1, OLt(), len_call)

        # rank_end = inputs_root1.getCoords()[root1_pos + 1]
        coords = EMethod(inputs, "getCoords", [])
        bound = EAccess(coords, EBinOp(pos, OAdd(), EInt(1)))
        end_then = SAssign(AVar(rank + "_end"), bound)

        # rank_end = ROOT
        end_else = SAssign(AVar(rank + "_end"), EVar(root.upper()))
        end_if = SIf((end_cond, end_then), [], end_else)

        return SBlock([start_if, end_if])

    def make_iter_expr(self, rank: str, tensors: List[Tensor]) -> Expression:
        """
        Given a list of tensors, make the expression used to combine them
        """
        # Make sure that we have at least one tensor
        if not tensors:
            raise ValueError("Must iterate over at least one tensor")

        # If there are no input tensors, we just need to iterShapeRef on the
        # output
        output_tensor = self.__get_output_tensor(tensors)
        if len(tensors) == 1 and output_tensor:
            out_name = output_tensor.fiber_name()
            iter_shape = EMethod(EVar(out_name), "iterShapeRef", [])
            return self.__add_enumerate(rank, iter_shape)

        # Build the expression of the inputs
        expr = self.__make_input_iter_expr(rank, tensors)

        # Finally, add in the output
        if output_tensor:
            trank = output_tensor.peek()
            if trank is not None:
                trank = trank.upper()

            if trank != rank:
                raise ValueError(
                    "Cannot project into the output tensor. Replace " +
                    rank +
                    " with " +
                    str(trank) +
                    " in the loop order")

            expr = Equation.__add_operator(
                EVar(output_tensor.fiber_name()), OLtLt(), expr)

        return self.__add_enumerate(rank, expr)

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

        payload: Payload
        if terms:
            # Construct the term payloads
            term_payloads = []
            for term in terms:
                payload = PVar(term[-1].fiber_name())
                for factor in reversed(term[:-1]):
                    payload = PTuple([PVar(factor.fiber_name()), payload])
                term_payloads.append(payload)

            # Construct the entire expression payload
            payload = term_payloads[-1]
            for term_payload in reversed(term_payloads[:-1]):
                payload = PTuple([PVar("_"), term_payload, payload])

            # Put the output on the outside
            if output_tensor:
                payload = PTuple([PVar(output_tensor.fiber_name()), payload])

        elif output_tensor:
            payload = PVar(output_tensor.fiber_name())

        else:
            # We should never get to this state
            raise ValueError("Something is wrong...")  # pragma: no cover

        # Add the rank variable
        rank_var = rank.lower()
        payload = PTuple([PVar(rank_var), payload])

        # If the spacetime style is occupancy, we also need to enumerate the
        # iterations
        if self.__need_enumerate(rank):
            payload = PTuple([PVar(rank_var + "_pos"), payload])

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

            product: Expression = EVar(factors[0])
            for factor in factors[1:]:
                product = EBinOp(product, OMul(), EVar(factor))
            products.append(product)

        # Combine the terms
        sum_ = products[0]
        for product in products[1:]:
            sum_ = EBinOp(sum_, OAdd(), product)

        # Create the final statement
        out_name = self.output.lower() + "_ref"
        return SIAssign(AVar(out_name), OAdd(), sum_)

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
                exprs[i] = EParens(expr)

        return EBinOp(exprs[0], op, exprs[1])

    def __add_enumerate(self, rank: str, expr: Expression) -> Expression:
        """
        Enumerate the iterations if necessary
        """
        if self.__need_enumerate(rank):
            expr = EFunc("enumerate", [AJust(expr)])
        return expr

    def __need_enumerate(self, rank: str) -> bool:
        """
        Returns True if new need to enumerate over this rank
        """
        # Check the interval
        enum_int = False
        root = self.program.get_partitioning().get_root_name(rank)
        # If this is not right before the bottom rank, we don't need to worry
        # about it
        if rank[len(root):] == "1":
            coord_math = self.program.get_coord_math()
            root_symbol = Symbol(root.lower())

            # Get all possibly affected symbols
            symbols = set()
            for trans in coord_math.get_all_exprs(root.lower()):
                if trans != root_symbol:
                    symbols.update(trans.atoms(Symbol))

            # If this rank is in any of the translations of these symbols,
            # then we will need to project that symbol, meaning we need the
            # enumerate
            for symbol in symbols:
                if root_symbol in coord_math.get_trans(symbol).atoms(Symbol):
                    enum_int = True

        # Check the spacetime
        spacetime = self.program.get_spacetime()
        enum_st = spacetime is not None and spacetime.emit_pos(rank)

        return enum_int or enum_st

    @staticmethod
    def __frac_coords(sexpr: Basic) -> bool:
        """
        Return True if fractional coordinates will be generated
        """
        if not isinstance(sexpr, Integer) and isinstance(sexpr, Rational):
            return True

        return any(Equation.__frac_coords(arg) for arg in sexpr.args)

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

    def __iter_fiber(self, rank: str, tensor: Tensor) -> Expression:
        """
        Get fiber for iteration (may involve projection)
        """
        trank = tensor.peek()
        if trank is None:
            raise ValueError(
                "Cannot iterate over payload " +
                tensor.fiber_name())

        # If this fiber is already over the correct rank, we can iterate over it
        # directly
        if trank == rank.lower():
            return EVar(tensor.fiber_name())

        # Otherwise, we need to project
        math = self.program.get_coord_math().get_trans(trank)
        sexpr = solve(math - Symbol(trank), Symbol(rank.lower()))[0]

        args = [AParam("trans_fn", ELambda([trank], CoordAccess.build_expr(
            sexpr))), AParam("interval", ETuple([EInt(0), EVar(rank)]))]
        project = EMethod(EVar(tensor.fiber_name()), "project", args)

        # If there are no fractional coordinates, we are done
        if not Equation.__frac_coords(sexpr):
            return project

        # Otherwise, we need to prune out the fractional coordinates
        # We do this by pruning on the function c % 1 == 0
        int_test = EBinOp(EBinOp(EVar("c"), OMod(), EInt(1)), OEqEq(), EInt(0))
        trans_fn = AParam("trans_fn", ELambda(["i", "c", "p"], int_test))

        return EMethod(project, "prune", [trans_fn])

    def __make_input_iter_expr(
            self,
            rank: str,
            tensors: List[Tensor]) -> Expression:
        """
        Make the iteration expression for the inputs
        """
        # Separate the tensors into terms
        terms = self.__separate_terms(tensors)

        # Combine terms with intersections
        intersections = []
        for term in terms:
            expr = self.__iter_fiber(rank, term[-1])
            for factor in reversed(term[:-1]):
                fiber = self.__iter_fiber(rank, factor)
                expr = Equation.__add_operator(fiber, OAnd(), expr)
            intersections.append(expr)

        # Combine intersections with a union
        expr = intersections[-1]
        for intersection in reversed(intersections[:-1]):
            expr = Equation.__add_operator(intersection, OOr(), expr)

        return expr

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
