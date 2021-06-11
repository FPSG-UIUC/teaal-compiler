"""
HFA AST and code generation for HFA statements
"""

from typing import List

from es2hfa.hfa.base import Expression, Operator, Payload, Statement
from es2hfa.hfa.expr import EVar


@Statement.register
class SAssign:
    """
    A variable assignment
    """

    def __init__(self, var: str, expr: Expression) -> None:
        self.var = var
        self.expr = expr

    def gen(self, depth: int) -> str:
        """
        Generate the HFA output for an SAssign
        """
        return "    " * depth + self.var + " = " + self.expr.gen()


@Statement.register
class SBlock:
    """
    A block of statements
    """

    def __init__(self, stmts: List[Statement]) -> None:
        self.stmts = stmts

    def gen(self, depth: int) -> str:
        """
        Generate the HFA output for an SBlock
        """
        return "\n".join([s.gen(depth) for s in self.stmts])

    def add(self, stmt: Statement) -> None:
        """
        Add a statement onto the end of the SBlock, combine if the new
        statement is also an SBlock
        """
        if isinstance(stmt, SBlock):
            self.stmts.extend(stmt.stmts)
        else:
            self.stmts.append(stmt)


@Statement.register
class SExpr:
    """
    A statement that is an expression (usually because the expression has side effects)
    """

    def __init__(self, expr: Expression) -> None:
        self.expr = expr

    def gen(self, depth: int) -> str:
        """
        Generate the HFA output for an SExpr
        """
        return "    " * depth + self.expr.gen()


@Statement.register
class SFor:
    """
    A for loop for iterating over fibers in HFA
    """

    def __init__(
            self,
            var: str,
            payload: Payload,
            expr: Expression,
            stmt: Statement) -> None:
        self.var = var
        self.payload = payload
        self.expr = expr
        self.stmt = stmt

    def gen(self, depth: int) -> str:
        """
        Generate the HFA output for an SFor
        """
        return "    " * depth + "for " + self.var + ", " + \
            self.payload.gen() + " in " + self.expr.gen() + ":\n" + self.stmt.gen(depth + 1)


@Statement.register
class SFunc:
    """
    A function definition
    """

    def __init__(self, name: str, args: List[EVar], body: Statement) -> None:
        self.name = name
        self.args = args
        self.body = body

    def gen(self, depth: int) -> str:
        """
        Generate the HFA output for an SFunc
        """
        args = ", ".join([arg.gen() for arg in self.args])
        header = "def " + self.name + "(" + args + "):\n"
        return "    " * depth + header + self.body.gen(depth + 1)


@Statement.register
class SIAssign:
    """
    A variable assignment that updates a variable in place, e.g. i += j
    """

    def __init__(self, var: str, op: Operator, expr: Expression) -> None:
        self.var = var
        self.op = op
        self.expr = expr

    def gen(self, depth: int) -> str:
        """
        Generate the HFA output for an SIAssign
        """
        return "    " * depth + self.var + " " + self.op.gen() + "= " + self.expr.gen()


@Statement.register
class SReturn:
    """
    A return statement for the end of a function
    """

    def __init__(self, expr: Expression) -> None:
        self.expr = expr

    def gen(self, depth: int) -> str:
        """
        Generate the HFA output for an SReturn
        """
        return "    " * depth + "return " + self.expr.gen()
