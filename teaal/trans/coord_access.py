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

Translation from a symbolic representation of an index access to the
corresonding HiFiber
"""

from sympy import Add, Basic, Integer, Mul, Rational, Symbol # type: ignore
from typing import Type

from teaal.hifiber import *


class CoordAccess:
    """
    A translator from the symbolic representation of an expression to the
    corresponding HiFiber program
    """
    @staticmethod
    def build_expr(sexpr: Basic) -> Expression:
        """
        Build an HiFiber expression from a SymPy expression
        """
        if isinstance(sexpr, Symbol):
            return EVar(str(sexpr))

        elif isinstance(sexpr, Integer):
            return EInt(int(sexpr))

        elif isinstance(sexpr, Rational):
            return EBinOp(EInt(sexpr.p), ODiv(), EInt(sexpr.q))

        elif isinstance(sexpr, Add):
            return CoordAccess.__combine(sexpr, OAdd)

        elif isinstance(sexpr, Mul):
            return CoordAccess.__combine(sexpr, OMul)

        else:
            raise ValueError("Unable to translate operator " + str(sexpr.func))

    @staticmethod
    def isolate_rank(expr: Basic, rank: str):
        """
        Substitute 0 for all Symbols that are not the given rank
        """
        for atom in expr.atoms(Symbol) - {Symbol(rank.lower())}:
            expr = expr.subs(atom, 0)

        return expr

    @staticmethod
    def __combine(sexpr: Basic, op: Type[Operator]) -> Expression:
        """
        Fold together an expression
        """
        hexprs = [CoordAccess.build_expr(arg) for arg in sexpr.args]
        bexpr = EBinOp(hexprs[-2], op(), hexprs[-1])

        for hexpr in reversed(hexprs[:-2]):
            bexpr = EBinOp(hexpr, op(), bexpr)

        return bexpr
