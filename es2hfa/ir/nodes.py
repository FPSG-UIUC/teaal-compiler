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

Representations of all of the nodes in the FlowGraph
"""

import abc
from typing import Any, cast, Iterable, List, Tuple


class Node(metaclass=abc.ABCMeta):
    """
    FlowGraph node interface
    """

    def __eq__(self, other: object) -> bool:
        """
        The == operator for nodes

        """
        if isinstance(other, type(self)):
            return self.__key() == other.__key()
        return False

    def __hash__(self) -> int:
        """
        Hash the node (needed to insert it into the graph)
        """
        return hash(repr(self))

    def __key(self) -> Iterable[Any]:
        """
        A tuple of all fields of a node
        """
        return ()

    def __repr__(self) -> str:
        """
        A string representation of the node for hashing
        """
        strs = [key if isinstance(key, str) else repr(key)
                for key in self.__key()]
        return "(" + type(self).__name__ + ", " + ", ".join(strs) + ")"


@Node.register
class FiberNode(Node):
    """
    A Node representing a fiber
    """

    def __init__(self, fiber: str):
        """
        Construct a FiberNode
        """
        self.fiber = fiber

    def get_fiber(self) -> str:
        """
        Accessor for the fiber
        """
        return self.fiber

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a FiberNode
        """
        return self.fiber,


@Node.register
class FromFiberNode(Node):
    """
    A Node representing a call to Tensor.fromFiber()
    """

    def __init__(self, tensor: str, rank: str):
        """
        Construct a FromFiberNode
        """
        self.tensor = tensor
        self.rank = rank

    def get_rank(self) -> str:
        """
        Accessor for the rank
        """
        return self.rank

    def get_tensor(self) -> str:
        """
        Accessor for the tensor
        """
        return self.tensor

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a FromFiberNode
        """
        return self.tensor, self.rank


@Node.register
class LoopNode(Node):
    """
    A Node representing a loop
    """

    def __init__(self, rank: str):
        """
        Construct a LoopNode
        """
        self.rank = rank

    def get_rank(self) -> str:
        """
        Accessor for the rank
        """
        return self.rank

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a LoopNode
        """
        return self.rank,


@Node.register
class OtherNode(Node):
    """
    Another type of node
    """

    def __init__(self, type_: str):
        """
        Construct another type of node
        Should be used for required, one-off nodes
        """
        self.type = type_

    def get_type(self) -> str:
        """
        Accessor for the type
        """
        return self.type

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of the fields of an OtherNode
        """
        return self.type,


@Node.register
class PartNode(Node):
    """
    A Node representing a partitioning function
    """

    def __init__(self, tensor: str, ranks: Tuple[str]):
        """
        Build a partitioning node for a given tensor and ranks
        """
        self.tensor = tensor
        self.ranks = ranks

    def get_ranks(self) -> Tuple[str]:
        """
        Accessor for the ranks
        """
        return self.ranks

    def get_tensor(self) -> str:
        """
        Accessor for the tensor
        """
        return self.tensor

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a PartNode
        """
        return self.tensor, self.ranks


@Node.register
class RankNode(Node):
    """
    A Node representing a rank
    """

    def __init__(self, tensor: str, rank: str):
        """
        Construct a node for a rank name, tagged with its tensor
        """
        self.tensor = tensor
        self.rank = rank

    def get_rank(self) -> str:
        """
        Accessor for the rank
        """
        return self.rank

    def get_tensor(self) -> str:
        """
        Accessor for the tensor
        """
        return self.tensor

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a PartNode
        """
        return self.tensor, self.rank


@Node.register
class SRNode(Node):
    """
    A Node representing a swizzleRanks and getRoot
    """

    def __init__(self, tensor: str, ranks: List[str]):
        """
        Construct a swizzleRanks and getRoot node
        """
        self.tensor = tensor
        self.ranks = ranks

    def get_ranks(self) -> List[str]:
        """
        Accessor for the ranks
        """
        return self.ranks

    def get_tensor(self) -> str:
        """
        Accessor for the tensor
        """
        return self.tensor

    def _Node__key(self) -> Iterable[Any]:
        """
        Iterable of fields of a PartNode
        """
        return self.tensor, self.ranks
