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
             | term

        ?factor: NAME -> var
               | tensor

        ?output: NAME "[" tinds "]"

        ?sum: "sum(" sinds ")." factor
            | "sum(" sinds ").(" expr ")"

        ?sinds: [NAME ("," NAME)*] -> sinds

        ?tensor: NAME "[" tinds "]"

        ?term: (factor "*")* factor -> times

        ?tinds: [NAME ("," NAME)*] -> tinds

        %import common.CNAME -> NAME
        %import common.WS_INLINE

        %ignore WS_INLINE
    """
    parser = Lark(grammar)

    @staticmethod
    def parse(einsum: str) -> Tree:
        return EinsumParser.parser.parse(einsum)
