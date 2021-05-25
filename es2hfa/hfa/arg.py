"""
HFA AST and code generation for HFA expressions
"""

import abc

from es2hfa.hfa.base import Argument, Expression


@Argument.register
class AJust:
    """
    An unparameterized argument to an HFA function
    """

    def __init__(self, expr: Expression) -> None:
        self.expr = expr

    def gen(self) -> str:
        """
        Generate the HFA code for an AJust
        """
        return self.expr.gen()


@Argument.register
class AParam:
    """
    A parameterized argument to an HFA function
    """

    def __init__(self, name: str, expr: Expression) -> None:
        self.name = name
        self.expr = expr

    def gen(self) -> str:
        """
        Generate the HFA code for an AParam
        """
        return self.name + "=" + self.expr.gen()
