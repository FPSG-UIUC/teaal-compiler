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

HFA AST and code generation for HFA statements
"""

from typing import List, Optional, Tuple

from es2hfa.hfa.base import Assignable, Expression, Operator, Payload, Statement
from es2hfa.hfa.expr import EVar


@Statement.register
class SAssign:
    """
    An assignment
    """

    def __init__(self, assn: Assignable, expr: Expression) -> None:
        self.assn = assn
        self.expr = expr

    def gen(self, depth: int) -> str:
        """
        Generate the HFA output for an SAssign
        """
        return "    " * depth + self.assn.gen() + " = " + self.expr.gen()


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
            payload: Payload,
            expr: Expression,
            stmt: Statement) -> None:
        self.payload = payload
        self.expr = expr
        self.stmt = stmt

    def gen(self, depth: int) -> str:
        """
        Generate the HFA output for an SFor
        """
        return "    " * depth + "for " + \
            self.payload.gen(False) + " in " + self.expr.gen() + ":\n" + self.stmt.gen(depth + 1)


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
    An assignment that updates an object in place, e.g. i += j
    """

    def __init__(
            self,
            assn: Assignable,
            op: Operator,
            expr: Expression) -> None:
        self.assn = assn
        self.op = op
        self.expr = expr

    def gen(self, depth: int) -> str:
        """
        Generate the HFA output for an SIAssign
        """
        return "    " * depth + self.assn.gen() + " " + self.op.gen() + \
            "= " + self.expr.gen()


@Statement.register
class SIf:
    """
    An if statement
    """

    def __init__(self,
                 if_: Tuple[Expression,
                            Statement],
                 elifs: List[Tuple[Expression,
                                   Statement]],
                 else_: Optional[Statement]) -> None:
        self.if_ = if_
        self.elifs = elifs
        self.else_ = else_

    def gen(self, depth: int) -> str:
        """
        Generate the HFA output for an SIf
        """
        out = "    " * depth
        out += "if " + self.if_[0].gen() + ":\n" + self.if_[1].gen(depth + 1)

        for cond, stmt in self.elifs:
            out += "\n" + "    " * depth
            out += "elif " + cond.gen() + ":\n" + stmt.gen(depth + 1)

        if self.else_ is not None:
            out += "\n" + "    " * depth
            out += "else:\n" + self.else_.gen(depth + 1)

        return out


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
