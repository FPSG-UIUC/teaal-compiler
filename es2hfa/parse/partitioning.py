"""
Lexing and parsing for the partitioning mapping information
"""

from lark import Lark
from lark.tree import Tree


class PartitioningParser:
    """
    Lexing and parsing for partitioning mapping information
    """
    grammar = """
        ?start: "uniform_shape(" NUMBER ")" -> uniform_shape
              | "divide_uniform(" NUMBER ")" -> divide_uniform

        %import common.SIGNED_NUMBER -> NUMBER
        %import common.WS_INLINE

        %ignore WS_INLINE
    """
    parser = Lark(grammar)

    @staticmethod
    def parse(info: str) -> Tree:
        return PartitioningParser.parser.parse(info)
