"""
Lexing and parsing for a single Einsum
"""

from lark import Lark, Token
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
              | num "*" NAME -> itimes

        ?num: NUMBER -> pos
            | "-" NUMBER -> neg

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
        tree = EquationParser.parser.parse(equation)

        # Parse both positive and negative numbers
        for itimes in tree.find_data("itimes"):
            num = itimes.children[0]
            assert isinstance(num, Tree)

            if num.data == "pos":
                itimes.children[0] = num.children[0]

            # Otherwise, it is a negative
            else:
                pos = num.children[0]
                assert isinstance(pos, Token)
                itimes.children[0] = Token("NUMBER", str(-1 * int(pos)))

        return tree
