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

Representations of all of the nodes in the partitioning graph
"""

import abc
from typing import Any, Iterable, Tuple

from teaal.ir.node import Node


class FlattenNode(Node):
    """
    A node that represents multiple ranks being combined together for
    flattening
    """

    def __init__(self, ranks: Tuple[str, ...]) -> None:
        """
        Construct a flatten node for partitioning
        """
        self.ranks = ranks

    def get_ranks(self) -> Tuple[str, ...]:
        """
        Accessor for the rank
        """
        return self.ranks

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a RankNode
        """
        return self.ranks,


class RankNode(Node):
    """
    A node that represents a rank in the partitioning graph
    """

    def __init__(self, rank: str) -> None:
        """
        Construct a rank node for partitioning
        """
        self.rank = rank

    def get_rank(self) -> str:
        """
        Accessor for the rank
        """
        return self.rank

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a RankNode
        """
        return self.rank,
