"""
Lexing and parsing for a tensor

Used for the declaration and the rank order information
"""

from lark import Lark
from lark.tree import Tree


class TensorParser:
    """
    A lexer and parser for a tensor
    """
    grammar = """
        ?start: NAME "[" tinds "]" -> tensor

        ?tinds: [NAME ("," NAME)*] -> tinds

        %import common.CNAME -> NAME
        %import common.WS_INLINE

        %ignore WS_INLINE
    """
    parser = Lark(grammar)

    @staticmethod
    def parse(tensor: str) -> Tree:
        return TensorParser.parser.parse(tensor)
