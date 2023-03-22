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

        ?expr: (term "+")* term -> plus

        ?factor: NAME -> var
               | tensor

        ?iexpr: (iterm "+")* iterm -> iplus

        ?iterm: NAME -> ijust
              | NUMBER "*" NAME -> itimes

        ?output: NAME "[" ranks "]"

        ?tensor: NAME "[" ranks "]"

        ?term: (factor "*")* factor -> times
               | "take(" (factor ",")* factor "," NUMBER ")" -> take

        ?ranks: [iexpr ("," iexpr)*] -> ranks

        %import common.CNAME -> NAME
        %import common.NUMBER -> NUMBER
        %import common.WS_INLINE

        %ignore WS_INLINE
    """
    parser = Lark(grammar)

    @staticmethod
    def parse(equation: str) -> Tree:
        return EquationParser.parser.parse(equation)
