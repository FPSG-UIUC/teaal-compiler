"""
HFA AST and code generation for HFA statements
"""

from typing import List

from es2hfa.hfa.base import Expression, Statement


@Statement.register
class SAssign:
    """
    A variable assignment
    """

    def __init__(self, var: str, expr: Expression) -> None:
        self.var = var
        self.expr = expr

    def gen(self) -> str:
        """
        Generate the HFA output for an SAssign
        """
        return self.var + " = " + self.expr.gen()


@Statement.register
class SBlock:
    """
    A block of statements
    """

    def __init__(self, stmts: List[Statement]) -> None:
        self.stmts = stmts

    def gen(self) -> str:
        """
        Generate the HFA output for an SBlock
        """
        return "\n".join([s.gen() for s in self.stmts])
