"""
HFA AST and code generation for HFA expressions
"""

from typing import List

from es2hfa.hfa.base import Argument, Expression, Operator


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
class EField:
    """
    An HFA object field access
    """

    def __init__(self, obj: str, field: str):
        self.obj = obj
        self.field = field

    def gen(self) -> str:
        """
        Generate the HFA code for an EField
        """
        return self.obj + "." + self.field


@Expression.register
class EFunc:
    """
    An HFA function call
    """

    def __init__(self, name: str, args: List[Argument]) -> None:
        self.name = name
        self.args = args

    def gen(self) -> str:
        """
        Generate the HFA code for an EFunc
        """
        return self.name + \
            "(" + ", ".join([a.gen() for a in self.args]) + ")"


@Expression.register
class EInt:
    """
    An HFA integer
    """

    def __init__(self, int_: int) -> None:
        self.int = int_

    def gen(self) -> str:
        """
        Generate HFA code for an EInt
        """
        return str(self.int)


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
class EMethod:
    """
    An HFA method call
    """

    def __init__(self, obj: str, name: str, args: List[Argument]) -> None:
        self.obj = obj
        self.name = name
        self.args = args

    def gen(self) -> str:
        """
        Generate the HFA code for an EMethod
        """
        return self.obj + "." + self.name + \
            "(" + ", ".join([a.gen() for a in self.args]) + ")"


@Expression.register
class EParens:
    """
    An HFA expression surrounded by parentheses
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
