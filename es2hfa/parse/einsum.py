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
        ?start: output "=" expr -> einsum
              | output "=" sum -> einsum

        ?expr: expr "+" expr -> plus
             | expr "-" expr -> minus
             | term

        ?factor: NAME
               | tensor

        ?output: NAME "[" tinds "]"

        ?sum: "sum(" sinds ")." factor
            | "sum(" sinds ").(" expr ")"

        ?sinds: [NAME ("," NAME)*] -> sinds

        ?tensor: NAME "[" tinds "]"

        ?term: term "*" term -> times
             | factor

        ?tinds: [NAME ("," NAME)*]

        %import common.CNAME -> NAME
        %import common.WS_INLINE

        %ignore WS_INLINE
    """
    parser = Lark(grammar)

    @staticmethod
    def parse(einsum: str) -> Tree:
        return EinsumParser.parser.parse(einsum)
