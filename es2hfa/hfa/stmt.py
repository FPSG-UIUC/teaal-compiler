"""
HFA AST and code generation for HFA statements
"""

from typing import List

from es2hfa.hfa.base import Expression, Operator, Payload, Statement


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
