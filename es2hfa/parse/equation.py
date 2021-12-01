"""
Lexing and parsing for a single Einsum
"""

from lark import Lark
from lark.tree import Tree


class EquationParser:
    """
    A lexer and parser for a single Einsum
    """
    grammar = """
        ?start: output "=" expr -> einsum
              | output "=" sum -> einsum

        ?expr: (term "+")* term -> plus

        ?factor: NAME -> var
               | tensor

        ?filter: factor -> single
               | "dot(" (factor ",")* factor "," NUMBER ")" -> dot

        ?output: NAME "[" tinds "]"

        ?sum: "sum(" sinds ")." factor
            | "sum(" sinds ").(" expr ")"

        ?sinds: [NAME ("," NAME)*] -> sinds

        ?tensor: NAME "[" tinds "]"

        ?term: (filter "*")* filter -> times

        ?tinds: [NAME ("," NAME)*] -> tinds

        %import common.CNAME -> NAME
        %import common.NUMBER -> NUMBER
        %import common.WS_INLINE

        %ignore WS_INLINE
    """
    parser = Lark(grammar)

    @staticmethod
    def parse(equation: str) -> Tree:
        return EquationParser.parser.parse(equation)
