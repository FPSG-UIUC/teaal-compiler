"""
Lexing and parsing for a single Einsum
"""

from lark import Lark

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

        ?op: "+"
           | "-"
           | "*"

        ?sum: "sum(" inds ")." expr

        ?tensor: NAME "[" inds "]"

        %import common.CNAME -> NAME
    """

    def __init__(self) -> None:
        """
        Construct the par
        """
        self.parser = Lark(EinsumParser.grammar)

    def parse(self, einsum: str) -> None:
        out = self.parser.parse(einsum)
        print(out)
        print(type(out))
