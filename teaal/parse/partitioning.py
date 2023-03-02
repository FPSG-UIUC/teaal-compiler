"""
MIT License

Copyright (c) 2021 University of Illinois

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Lexing and parsing for the partitioning mapping information
"""

from lark import Lark
from lark.tree import Tree


class PartitioningParser:
    """
    Lexing and parsing for partitioning mapping information
    """
    partitioning_grammar = """
        ?start: "nway_shape(" size ")" -> nway_shape
              | "uniform_occupancy(" leader "." size ")" -> uniform_occupancy
              | "uniform_shape(" size ")" -> uniform_shape
              | "flatten(" ")" -> flatten

        ?leader: NAME -> leader

        ?size: NUMBER -> int_sz
             | NAME -> str_sz

        %import common.CNAME -> NAME
        %import common.NUMBER -> NUMBER
        %import common.WS_INLINE

        %ignore WS_INLINE
    """

    ranks_grammar = """
        ?start: NAME -> rank
              | "(" NAME ("," NAME)+ ")" -> ranks

        %import common.CNAME -> NAME
        %import common.WS_INLINE

        %ignore WS_INLINE
    """

    partitioning_parser = Lark(partitioning_grammar)
    ranks_parser = Lark(ranks_grammar)

    @staticmethod
    def parse_partitioning(info: str) -> Tree:
        return PartitioningParser.partitioning_parser.parse(info)

    @staticmethod
    def parse_ranks(info: str) -> Tree:
        return PartitioningParser.ranks_parser.parse(info)
