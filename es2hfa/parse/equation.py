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

        ?output: NAME "[" tranks "]"

        ?sum: "sum(" sranks ")." factor
            | "sum(" sranks ").(" expr ")"

        ?sranks: [NAME ("," NAME)*] -> sranks

        ?tensor: NAME "[" tranks "]"

        ?term: (filter "*")* filter -> times

        ?tranks: [NAME ("," NAME)*] -> tranks

        %import common.CNAME -> NAME
        %import common.NUMBER -> NUMBER
        %import common.WS_INLINE

        %ignore WS_INLINE
    """
    parser = Lark(grammar)

    @staticmethod
    def parse(equation: str) -> Tree:
        return EquationParser.parser.parse(equation)
