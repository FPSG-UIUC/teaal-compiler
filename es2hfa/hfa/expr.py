"""
HFA AST and code generation for HFA expressions
"""

import abc
from typing import List

from es2hfa.hfa.base import Expression, Operator


@Expression.register
class EBinOp:
    """
    An HFA binary operation
    """

    def __init__(
            self,
            expr1: Expression,
            op: Operator,
            expr2: Expression) -> None:
        self.expr1 = expr1
        self.op = op
        self.expr2 = expr2

    def gen(self) -> str:
        """
        Generate the HFA code for an EBinOp
        """
        return self.expr1.gen() + " " + self.op.gen() + " " + self.expr2.gen()


@Expression.register
class EList:
    """
    An HFA list
    """

    def __init__(self, list_: List[Expression]) -> None:
        self.list = list_

    def gen(self) -> str:
        """
        Generate the HFA code for an EList
        """
        return "[" + ", ".join([e.gen() for e in self.list]) + "]"


@Expression.register
class EParens:
    """
    An HFA expression surroounded by parentheses
    """

    def __init__(self, expr: Expression) -> None:
        self.expr = expr

    def gen(self) -> str:
        """
        Generate the HFA code for an EParens
        """
        return "(" + self.expr.gen() + ")"


@Expression.register
class EString:
    """
    A string in HFA
    """

    def __init__(self, string: str) -> None:
        self.string = string

    def gen(self) -> str:
        """
        Generate the HFA code for an EString
        """
        return "\"" + self.string + "\""


@Expression.register
class EVar:
    """
    An HFA variable
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def gen(self) -> str:
        """
        Generate the HFA code for an EVar
        """
        return self.name
