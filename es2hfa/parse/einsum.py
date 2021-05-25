"""
Lexing and parsing for a single Einsum
"""

from lark import Lark
from lark.tree import Tree


class EinsumParser:
    """
    A lexer and parser for a single Einsum
    """
    grammar = """
        ?start: tensor "=" expr
              | tensor "=" sum

        ?expr: NAME
             | "(" expr ")"
             | expr op expr
             | tensor

        ?inds: [NAME ("," NAME)*]

        ?op: "+" -> plus
           | "-" -> minus
           | "*" -> times

        ?sum: "sum(" inds ")." expr

        ?tensor: NAME "[" inds "]"

        %import common.CNAME -> NAME
        %import common.WS_INLINE

        %ignore WS_INLINE
    """
    parser = Lark(grammar)

    @staticmethod
    def parse(einsum: str) -> Tree:
        return EinsumParser.parser.parse(einsum)
